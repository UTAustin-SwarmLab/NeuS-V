import pickle
import warnings
from pathlib import Path

import gradio as gr

from neus_v.smooth_scoring import smooth_confidence_scores
from neus_v.utils import clear_gpu_memory
from neus_v.veval.eval import evaluate_video_with_sequence_of_images
from neus_v.veval.parse import parse_proposition_set, parse_tl_specification
from neus_v.vlm.vllm_client import VLLMClient

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="Conversion of an array with ndim > 0 to a scalar is deprecated"
)

# Paths and parameters
# WEIGHT_PATH = Path("/opt/mars/mnt/model_weights")
# WEIGHT_PATH = Path("/nas/mars/model_weights/")
WEIGHT_PATH = Path("./assets/")
pickle_path = WEIGHT_PATH / "distributions.pkl"
num_of_frame_in_sequence = 3
model = "InternVL2-8B"
device = 0
# Load the vision-language model
vision_language_model = VLLMClient(api_base="http://localhost:8000/v1", model="OpenGVLab/InternVL2_5-8B")
# Load distributions
print(f"Loading distributions from {pickle_path}")
with open(pickle_path, "rb") as f:
    distributions = pickle.load(f)
all_dimension_data = distributions.get(model).get("all_dimension")


# TODO: Make paths better for public release
def process_video(video_path, propositions, tl):
    """Process the video and compute the score_on_all."""
    proposition_set = parse_proposition_set(propositions.split(","))
    tl_spec = parse_tl_specification(tl)
    threshold = 0.349

    try:
        result = evaluate_video_with_sequence_of_images(
            vision_language_model=vision_language_model,
            confidence_as_token_probability=True,
            video_path=video_path,
            proposition_set=proposition_set,
            tl_spec=tl_spec,
            parallel_inference=False,
            num_of_frame_in_sequence=num_of_frame_in_sequence,
            threshold=threshold,
        )
        probability = result.get("probability")
        score_on_all = float(
            smooth_confidence_scores(
                target_data=[probability],
                prior_distribution=all_dimension_data,
            )
        )
        clear_gpu_memory()
        return score_on_all

    except Exception as e:
        clear_gpu_memory()
        return f"Error: {str(e)}"


# Gradio interface
def demo_interface(video, propositions, tl):
    """Wrapper for the Gradio interface."""
    return process_video(video, propositions, tl)


def main():
    # Example data from the original script
    example_video_path_1 = (
        "assets/A_storm_bursts_in_with_intermittent_lightning_and_causes_flooding_and_large_waves_crash_in.mp4"
    )
    example_video_path_2 = "assets/The ocean waves gently lapping at the shore, until a storm bursts in, and then lightning flashes across the sky.mp4"
    example_propositions = "waves lapping,ocean shore,storm bursts in,lightning on the sky"
    example_tl = '("waves_lapping" & "ocean_shore") U ("storm_bursts_in" U "lightning_on_the_sky")'

    demo = gr.Interface(
        fn=demo_interface,
        inputs=[
            gr.Video(label="Upload Video"),
            gr.Textbox(label="List of Propositions (comma-separated)"),
            gr.Textbox(label="Temporal Logic Specification"),
        ],
        outputs=gr.Textbox(label="Score on All"),
        title="Video Evaluation with Temporal Logic",
        description="Upload a video and provide propositions and temporal logic to evaluate the score_on_all.",
        examples=[
            [example_video_path_1, example_propositions, example_tl],
            [example_video_path_2, example_propositions, example_tl],
        ],
    )

    # demo.launch(allowed_paths=["/nas/mars/dataset/teaser"])
    demo.launch(allowed_paths=["assets/"], server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
