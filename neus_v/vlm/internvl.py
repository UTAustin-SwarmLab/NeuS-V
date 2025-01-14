import gc
import logging

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import softmax
from transformers import AutoModel, AutoTokenizer

from neus_v.vlm.internvl_utils import (
    assign_device_map,
    load_image,
    load_video_from_file,
    load_video_from_seq_of_frames,
    split_model,
)
from neus_v.vlm.obj import DetectedObject

MODEL_PATH = {
    "InternVL2-40B": "HuggingFace Model",
    "InternVL2-8B": "HuggingFace Model",
    "InternVL2-2B": "HuggingFace Model",
}


class InternVL:
    """InternVL's Vision Language Model."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        device: int = 0,
    ) -> None:
        """Initialization the InternVL."""
        logging.info(
            (
                "You are using the model based on HuggingFace API.",
                "The model will be downloaded to the HuggingFace cache dir.",
            )
        )
        self.model_name = model_name
        self._path = f"OpenGVLab/{model_name}"
        self._num_gpus = torch.cuda.device_count()
        self.device = device
        if multi_gpus:
            device_map = split_model(model_name)
        else:
            device_map = assign_device_map(model_name=model_name, manual_gpu_id=device)
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True, use_fast=False)

    def reset_model(self) -> None:
        """Reset the model to its initial state using pretrained weights."""
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)

    def clear_gpu_memory(self) -> None:
        """Clear CUDA cache and run garbage collection to free GPU memory."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()  # Run garbage collector

    def move_tensors_to_gpu(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Move all tensors in the module to GPU if they are on the CPU."""
        for name, tensor in module.named_buffers():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                module.register_buffer(
                    name,
                    tensor.cuda(self.device),
                    persistent=False,
                )
        for _, param in module.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.cuda(self.device)

    def infer_with_image(
        self,
        language: str,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            image is not None or image_path is not None
        ), "One of 'image' or 'image_path' must be defined."
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image)
        # set the max number of tiles in `max_num`
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda(self.device)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        image_prefix = "<image>\n"
        language = image_prefix + language
        return self.model.chat(self.tokenizer, pixel_values, language, generation_config)

    def infer_with_video(
        self,
        language: str,
        seq_of_frames: list[np.ndarray] | None = None,
        video_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> str:
        """Perform image inference with given video inputs."""
        assert (  # noqa: S101
            seq_of_frames is not None or video_path is not None
        ), "One of 'seq_of_frames' or 'video_path' must be defined."
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if video_path:
            pixel_values, num_patches_list = load_video_from_file(video_path)
        else:
            pixel_values, num_patches_list = load_video_from_seq_of_frames(seq_of_frames=seq_of_frames)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        language = video_prefix + language
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            language,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

    def detect(
        self,
        scene_description: str,
        frame_img: np.ndarray | None = None,
        seq_of_frames: list[np.ndarray] | None = None,
        video_path: str | None = None,
        threshold: float = 0.349,
        confidence_as_token_probability: bool = True,
    ) -> DetectedObject:
        """Detect objects in the given frame image.

        Args:
            frame_img (np.ndarray): The image frame to process.
            scene_description (str): Description of the scene.
            seq_of_frames (list[np.ndarray] | None):
                List of video frames to process.
            video_path (str | None): Path to video file to process.
            threshold (float): Detection threshold.
            confidence_as_token_probability (bool):
                Whether to use token probabilities for confidence.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        if confidence_as_token_probability:
            parsing_rule = [
                "You must only return a Yes or No, and not both, to any question asked. "  # noqa: E501
                "You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times.",  # noqa: E501
                "For example, if the question is 'Is there a cat present in the Image?', the answer must only be 'Yes' or 'No'.",  # noqa: E501
            ]
            parsing_rule = "\n".join(parsing_rule)
            prompt = rf"Is there a {scene_description} present in the image? " f"[PARSING RULE]\n:{parsing_rule}"

            if seq_of_frames or video_path:
                response, confidence = self.infer_with_video_confidence(
                    language=prompt,
                    seq_of_frames=seq_of_frames,
                    video_path=video_path,
                )
            else:
                response, confidence = self.infer_with_image_confidence(language=prompt, image=frame_img)
            # TODO: Add a check for the response to be Yes or NO or clean up response better  # noqa: E501
            if "yes" in response.lower():
                detected = True
                if confidence <= threshold:
                    confidence = 0.0
                    detected = False

            else:
                detected = False
                confidence = 0.0

        return DetectedObject(
            name=scene_description,
            model_name=self.model_name,
            confidence=round(confidence, 3),
            probability=round(confidence, 3),
            number_of_detection=1,
            is_detected=detected,
        )

    def infer_with_image_confidence(
        self,
        language: str,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Perform image inference and return response with confidence score.

        Args:
            language (str): The input prompt or question.
            image (np.ndarray | None): The input image as a numpy array.
            image_path (str | None): Path to the input image file.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling for generation.

        Returns:
            tuple[str, float]: Generated response and confidence score.
        """
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(image)
        # set the max number of tiles in `max_num`
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda(self.device)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        image_prefix = "<image>\n"
        language = image_prefix + language

        return self.chat_with_confidence(self.tokenizer, pixel_values, language, generation_config)

    def chat_with_confidence(  # noqa: PLR0913
        self,
        tokenizer: AutoTokenizer,
        pixel_values: torch.Tensor,
        question: str,
        generation_config: dict,
        num_patches_list: list[int] | None = None,
        IMG_START_TOKEN: str = "<img>",  # noqa: N803, S107
        IMG_END_TOKEN: str = "</img>",  # noqa: N803, S107
        IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",  # noqa: N803, S107
        verbose: bool = False,
    ) -> tuple[str, float]:
        """Generate a response with confidence score for the given input.

        Args:
            tokenizer: The tokenizer to use.
            pixel_values: Image tensor input.
            question: The input question or prompt.
            generation_config: Configuration for text generation.
            num_patches_list: List of number of patches for video frames.
            IMG_START_TOKEN: Token to mark the start of an image.
            IMG_END_TOKEN: Token to mark the end of an image.
            IMG_CONTEXT_TOKEN: Token for image context.
            verbose: Whether to print verbose output.

        Returns:
            A tuple containing the generated response and its confidence score.
        """
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)  # noqa: S101

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = self.model.conv_template
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")  # noqa: T201

        for num_patches in num_patches_list:
            context_tokens = IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
            image_tokens = IMG_START_TOKEN + context_tokens + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda(self.device)
        attention_mask = model_inputs["attention_mask"].cuda(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_config["return_dict_in_generate"] = True
        generation_config["output_scores"] = True
        generation_config["output_logits"] = True
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()

        logits_to_compute = np.where(generation_output.sequences[0].detach().cpu().numpy() != eos_token_id)[0]
        confidence = 1.0
        for logit in logits_to_compute:
            token = generation_output.sequences[0, logit].item()
            prob = softmax(generation_output.logits[logit])[0, token]
            confidence = prob.item() * confidence
        self.clear_gpu_memory()
        return response, confidence

    def infer_with_video_confidence(
        self,
        language: str,
        seq_of_frames: list[np.ndarray] | None = None,
        video_path: str | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Perform video inference and return response with confidence score.

        Args:
            language (str): The input prompt or question.
            seq_of_frames (list[np.ndarray] | None):
                List of video frames as numpy arrays.
            video_path (str | None): Path to the input video file.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling for generation.

        Returns:
            tuple[str, float]: Generated response and confidence score.
        """
        assert (  # noqa: S101
            seq_of_frames is not None or video_path is not None
        ), "One of 'seq_of_frames' or 'video_path' must be defined."

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if video_path:
            pixel_values, num_patches_list = load_video_from_file(video_path, device=self.device)
        else:
            pixel_values, num_patches_list = load_video_from_seq_of_frames(
                seq_of_frames=seq_of_frames, device=self.device
            )

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        language = video_prefix + language

        return self.chat_with_confidence(
            self.tokenizer,
            pixel_values,
            language,
            generation_config,
            num_patches_list=num_patches_list,
        )
