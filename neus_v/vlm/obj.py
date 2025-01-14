import dataclasses
import enum
import logging
from typing import Any


class Status(enum.Enum):
    """Status Enum for the CV API."""

    UNKNOWN = 0
    SUCCESS = 1
    RUNNING = 2
    FAILURE = 3
    INVALID = 4


class DetectedObject:
    """Detected Object class."""

    name: str | None
    confidence: float = 0.0
    probability: float = 0.0
    confidence_of_all_obj: list[float] | None = dataclasses.field(default_factory=list)
    probability_of_all_obj: list[float] | None = dataclasses.field(default_factory=list)
    all_obj_detected: list[Any] | None = None
    number_of_detection: int = 0
    is_detected: bool | Status = Status.UNKNOWN
    model_name: str | None = None
    bounding_box_of_all_obj: list[Any] | None = None

    def __post_init__(self) -> None:
        """Post init."""
        if self.confidence_of_all_obj is not None and len(self.confidence_of_all_obj) > 0:
            self.confidence = max(self.confidence_of_all_obj)
        if self.probability_of_all_obj and len(self.probability_of_all_obj) > 0:
            self.probability = max(self.probability_of_all_obj)

    def get_probability(self) -> float:
        """Get probability."""
        if self.probability > 0:
            return self.probability
        if self.confidence > 0 and self.probability == 0:
            logging.info("Probability is not set, using confidence: %f", self.confidence)
            return self.confidence
        return self.probability
