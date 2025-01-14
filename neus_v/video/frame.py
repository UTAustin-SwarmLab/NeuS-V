import dataclasses
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass
class VideoFrame:
    """Frame class."""

    frame_idx: int
    timestamp: int | None = None
    frame_image: np.ndarray | None = None
    annotated_image: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    detected_object_set: dict | None = None
    object_of_interest: dict | None = None
    activity_of_interest: dict | None = None

    def save_frame_img(self, save_path: str) -> None:
        """Save frame image."""
        if self.frame_image is not None:
            cv2.imwrite(
                save_path,
                self.frame_image,
            )

    def is_any_object_detected(self) -> bool:
        """Check if object is detected."""
        return len(self.detected_object_set.objects) > 0

    @property
    def list_of_detected_object_of_interest(self) -> list:
        """Get detected object."""
        detected_obj = []
        for obj_name, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                detected_obj.append(obj_name)
        return detected_obj

    @property
    def detected_object_dict(self) -> dict:
        """Get detected object info as dict."""
        detected_obj = {}
        for obj_name, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                detected_obj[obj_name] = {}
                detected_obj[obj_name]["total_number_of_detection"] = obj_value.number_of_detection
                detected_obj[obj_name]["maximum_probability"] = max(obj_value.probability_of_all_obj)
                detected_obj[obj_name]["minimum_probability"] = min(obj_value.probability_of_all_obj)
                detected_obj[obj_name]["maximum_confidence"] = max(obj_value.confidence_of_all_obj)
                detected_obj[obj_name]["minimum_confidence"] = min(obj_value.confidence_of_all_obj)

        return detected_obj

    def detected_bboxes(self, probability_threshold: bool = False) -> list:
        """Get detected object.

        Args:
            probability_threshold (float | None): Probability threshold.
            Defaults to None.

        Returns:
            list: Bounding boxes.
        """
        bboxes = []

        for _, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                if probability_threshold:
                    for obj_prob in obj_value.probability_of_all_obj:
                        if obj_prob > 0:
                            bboxes += obj_value.bounding_box_of_all_obj
                else:
                    bboxes += obj_value.bounding_box_of_all_obj

        return bboxes


