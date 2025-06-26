"""Data handling for classification tasks."""

from .datasets import BreedClassificationDataset, ClassificationDataset
from .predictions import ClassificationPredictions as ClassificationPredictions

__all__ = ["ClassificationDataset", "BreedClassificationDataset", "ClassificationPredictions"]
