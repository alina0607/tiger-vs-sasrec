"""Concrete models. Importing this package registers every model by name."""

from recsys.models.popularity import PopularityRecommender
from recsys.models.sasrec import SASRec, SASRecNet

__all__ = ["PopularityRecommender", "SASRec", "SASRecNet"]
