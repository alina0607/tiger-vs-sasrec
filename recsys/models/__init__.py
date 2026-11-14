"""Concrete models. Importing this package registers every model by name."""

from recsys.models.popularity import PopularityRecommender
from recsys.models.sasrec import SASRec, SASRecNet
from recsys.models.tiger import TIGER, TIGERNet

__all__ = ["TIGER", "PopularityRecommender", "SASRec", "SASRecNet", "TIGERNet"]
