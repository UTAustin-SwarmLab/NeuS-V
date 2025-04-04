import argparse
import pickle
import warnings
from pathlib import Path

from neus_v.smooth_scoring import smooth_confidence_scores
from neus_v.utils import clear_gpu_memory
from neus_v.veval.eval import evaluate_video_with_sequence_of_images
from neus_v.veval.parse import parse_proposition_set, parse_tl_specification
from neus_v.vlm.internvl import InternVL

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="Conversion of an array with ndim > 0 to a scalar is deprecated"
)

# Paths and parameters
WEIGHT_PATH = Path("/nas/mars/model_weights/")
pickle_path = WEIGHT_PATH / "distributions.pkl"
num_of_frame_in_sequence = 3
model = "InternVL2-8B"
device = 7

# Load the vision-language model
vision_language_model = InternVL(model_name=model, device=device)

# Load distributions
with open(pickle_path, "rb") as f:
    distributions = pickle.load(f)
all_dimension_data = distributions.get(model).get("all_dimension")


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


def main():
    # parser = argparse.ArgumentParser(description="Process a video using temporal logic evaluation.")
    # parser.add_argument("video", type=str, help="Path to the video file.")
    # parser.add_argument("propositions", type=str, help="List of propositions (comma-separated).")
    # parser.add_argument("tl", type=str, help="Temporal logic specification.")

    # args = parser.parse_args()

    # score = process_video(args.video, args.propositions, args.tl)
    # print(f"Score on All: {score}")

    # Example usage
    example_video_path_1 = "/nas/mars/dataset/teaser/A_storm_bursts_in_with_intermittent_lightning_and_causes_flooding_and_large_waves_crash_in.mp4"
    example_video_path_2 = "/nas/mars/dataset/teaser/The ocean waves gently lapping at the shore, until a storm bursts in, and then lightning flashes across the sky.mp4"
    example_propositions = "waves lapping,ocean shore,storm bursts in,lightning on the sky"
    example_tl = '("waves_lapping" & "ocean_shore") U ("storm_bursts_in" U "lightning_on_the_sky")'

    print("Example 1:")
    print(f"Score: {process_video(example_video_path_1, example_propositions, example_tl)}")
    print("Example 2:")
    print(f"Score: {process_video(example_video_path_2, example_propositions, example_tl)}")


if __name__ == "__main__":
    main()
