import base64

import cv2
import numpy as np
from openai import OpenAI

from neus_v.calibration.vlm import smooth_mapping
from neus_v.vlm.obj import DetectedObject


class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model="OpenGVLab/InternVL2_5-8B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.system_message = (
            "You must only return a Yes or No, and not both, to any question asked.\n"
            "You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times.\n"
            "For example, if the question is 'Is there a cat present in the Image?', the answer must only be 'Yes' or 'No'."
        )

    # def _encode_frame(self, frame):
    #     return base64.b64encode(frame.tobytes()).decode("utf-8")
    def _encode_frame(self, frame):
        # Encode a uint8 numpy array (image) as a JPEG and then base64 encode it.
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def detect(self, seq_of_frames, scene_description, threshold=None, **kwargs):
        # Encode each frame.
        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]

        # Build the user message: a text prompt plus one image for each frame.
        user_content = [
            {
                "type": "text",
                "text": f"Does the sequence of these images depict '{scene_description}'",
            }
        ]
        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        # Create a chat completion request.
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )

        # Retrieve the list of TopLogprob objects.
        top_logprobs_list = chat_response.choices[0].logprobs.content[0].top_logprobs

        # Build a mapping from token text (stripped) to its probability.
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        # Extract probabilities for "Yes" and "No"
        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        # Compute the normalized probability for "Yes": p_yes / (p_yes + p_no)
        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            raise ValueError("No probabilities for 'Yes' or 'No' found in the response.")

        if threshold:
            confidence = smooth_mapping(confidence=confidence, false_threshold=threshold)
            if confidence < threshold:
                no_prob = 1.0

        return DetectedObject(
            name=scene_description,
            model_name=self.model,
            confidence=round(confidence, 3),
            probability=round(confidence, 3),
            number_of_detection=1,
            is_detected=yes_prob > no_prob,  # TODO: Check if this is correct
        )
