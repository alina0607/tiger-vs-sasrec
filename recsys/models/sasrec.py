"""SASRec (Kang & McAuley, ICDM 2018): self-attention over a user's item history.

`SASRec` *is a* `ScoringRecommender` and *has a* `SASRecNet` (an `nn.Module`).
Inheriting from both would mix two `__init__` and `__setattr__` contracts, and
the popularity baseline shows a recommender needs no network at all.
"""

import copy
import logging
import math
import random
from collections.abc import Sequence

import torch
from torch import nn

from recsys.base import History, ScoringRecommender
from recsys.config import SASRecConfig
from recsys.data import Split
from recsys.decorators import seeded, timed
from recsys.evaluate import evaluate

logger = logging.getLogger(__name__)


class SASRecNet(nn.Module):
    """The network: item and position embeddings, causal self-attention blocks.

    Item `n_items` is the padding id; it has an embedding row that stays zero.
    The item embedding is shared between the input and the output scores.
    """

    def __init__(self, n_items: int, config: SASRecConfig) -> None:
        super().__init__()
        d = config.hidden
        self.pad = n_items
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=n_items)
        self.pos_emb = nn.Embedding(config.max_len, d)
        self.drop = nn.Dropout(config.dropout)
        self.attn_norms = nn.ModuleList(
            nn.LayerNorm(d, eps=1e-8) for _ in range(config.blocks)
        )
        self.attns = nn.ModuleList(
            nn.MultiheadAttention(
                d, config.heads, dropout=config.dropout, batch_first=True
            )
            for _ in range(config.blocks)
        )
        self.ffn_norms = nn.ModuleList(
            nn.LayerNorm(d, eps=1e-8) for _ in range(config.blocks)
        )
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, d),
                nn.Dropout(config.dropout),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.Dropout(config.dropout),
            )
            for _ in range(config.blocks)
        )
        self.last_norm = nn.LayerNorm(d, eps=1e-8)
        # PyTorch starts embeddings at N(0, 1); scores of 50-dimensional dot products
        # then saturate the sigmoid and training crawls. Xavier, as in the reference
        # PyTorch port, starts every weight matrix small.
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.item_emb.weight[n_items].zero_()

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """Hidden state per position: (batch, length) item ids -> (batch, length, d)."""
        length = seqs.shape[1]
        x = self.item_emb(seqs) * math.sqrt(self.item_emb.embedding_dim)
        x = x + self.pos_emb(torch.arange(length, device=seqs.device))
        keep = (seqs != self.pad).unsqueeze(-1)
        x = self.drop(x) * keep
        # True above the diagonal: position t may not look at t+1, t+2, ...
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seqs.device), 1
        )
        for attn_norm, attn, ffn_norm, ffn in zip(
            self.attn_norms, self.attns, self.ffn_norms, self.ffns, strict=True
        ):
            q = attn_norm(x)
            out, _ = attn(q, x, x, attn_mask=future, need_weights=False)
            x = ffn_norm(q + out)
            x = (x + ffn(x)) * keep
        hidden: torch.Tensor = self.last_norm(x)
        return hidden

    def score(self, hidden: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Dot product of each hidden state with its item's embedding."""
        emb: torch.Tensor = self.item_emb(items)
        return (hidden * emb).sum(-1)


class SASRec(ScoringRecommender):
    """Trains `SASRecNet` with binary cross-entropy and one negative per position.

    Training stops early when validation NDCG@10 has not improved for
    `patience` checks; the best checkpoint is kept.

    Attributes:
        net: The network, built by `fit` once the number of items is known.
        history: One entry per validation check: epoch, mean loss, NDCG@10.
    """

    name = "sasrec"
    config_cls = SASRecConfig
    config: SASRecConfig

    def __init__(
        self, config: SASRecConfig | None = None, device: str | None = None
    ) -> None:
        self.config = config or SASRecConfig()
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.net: SASRecNet | None = None
        self.history: list[dict[str, float]] = []

    @timed
    def fit(self, split: Split) -> None:
        """Train on `split.train()`, choosing the epoch by validation NDCG@10."""
        with seeded(self.config.seed):
            self._train(split)

    def _train(self, split: Split) -> None:
        cfg = self.config
        self.net = SASRecNet(split.n_items, cfg).to(self.device)
        inputs, targets = self._training_pairs(split.train(), split.n_items)
        seen = [set(seq) for seq in split.train()]
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr, betas=(0.9, 0.98)
        )
        bce = nn.BCEWithLogitsLoss()
        best, best_state, waited = -1.0, copy.deepcopy(self.net.state_dict()), 0
        for epoch in range(1, cfg.max_epochs + 1):
            negatives = _sample_negatives(targets, seen, split.n_items)
            self.net.train()
            losses = []
            for rows in torch.randperm(len(inputs)).split(cfg.batch_size):
                seqs, pos, neg = (
                    t[rows].to(self.device) for t in (inputs, targets, negatives)
                )
                hidden = self.net(seqs)
                real = pos != self.net.pad
                pos_logit = self.net.score(hidden, pos)[real]
                neg_logit = self.net.score(hidden, neg)[real]
                loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(
                    neg_logit, torch.zeros_like(neg_logit)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % cfg.eval_every:
                continue
            ndcg = evaluate(self, split, part="valid", ks=(10,))["ndcg@10"]
            mean_loss = sum(losses) / len(losses)
            self.history.append({"epoch": epoch, "loss": mean_loss, "ndcg@10": ndcg})
            logger.info("epoch %d loss %.4f valid ndcg@10 %.4f", epoch, mean_loss, ndcg)
            if ndcg > best:
                best, best_state, waited = ndcg, copy.deepcopy(self.net.state_dict()), 0
            else:
                waited += 1
                if waited >= cfg.patience:
                    break
        self.net.load_state_dict(best_state)

    def _training_pairs(
        self, sequences: Sequence[History], n_items: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs and next-item targets, left-padded: (users, max_len) each."""
        length = self.config.max_len
        inputs = torch.full((len(sequences), length), n_items, dtype=torch.long)
        targets = torch.full((len(sequences), length), n_items, dtype=torch.long)
        for row, seq in enumerate(sequences):
            x, y = list(seq[:-1])[-length:], list(seq[1:])[-length:]
            inputs[row, length - len(x) :] = torch.tensor(x)
            targets[row, length - len(y) :] = torch.tensor(y)
        return inputs, targets

    def score_all(self, histories: Sequence[History]) -> torch.Tensor:
        """Score every item from the hidden state at each history's last position.

        Raises:
            RuntimeError: If `fit` has not been called.
        """
        if self.net is None:
            raise RuntimeError("call fit() before scoring")
        length, pad = self.config.max_len, self.net.pad
        seqs = torch.full((len(histories), length), pad, dtype=torch.long)
        for row, history in enumerate(histories):
            last = list(history)[-length:]
            if last:
                seqs[row, length - len(last) :] = torch.tensor(last)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            hidden: torch.Tensor = self.net(seqs.to(self.device))[:, -1, :]
            scores = hidden @ self.net.item_emb.weight[:pad].T
        self.net.train(was_training)
        return scores.cpu()


def _sample_negatives(
    targets: torch.Tensor, seen: list[set[int]], n_items: int
) -> torch.Tensor:
    """One random unbought item per real target (the paper's j not in S_u)."""
    negatives = torch.full_like(targets, n_items)
    for row, col in (targets != n_items).nonzero().tolist():
        item = random.randrange(n_items)
        while item in seen[row]:
            item = random.randrange(n_items)
        negatives[row, col] = item
    return negatives
