import pickle
import warnings
from pathlib import Path

import gradio as gr

from neus_v.puls.prompts import Mode
from neus_v.puls.puls import PULS
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
WEIGHT_PATH = Path("./assets/")
pickle_path = WEIGHT_PATH / "distributions.pkl"
num_of_frame_in_sequence = 3
model = "InternVL2-8B"
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


def generate_from_puls(prompt, mode_choice):
    """Generate propositions and TL spec from a natural language prompt using PULS."""
    # Check if prompt is blank
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a text prompt first", duration=5)
        # return "Error: Please enter a text prompt first", "Error: Please enter a text prompt first"

    mode_map = {
        "Object-Action Alignment": Mode.OBJECT_ACTION_ALIGNMENT,
        "Overall Consistency (default)": Mode.OVERALL_CONSISTENCY,
        "Object Existence": Mode.OBJECT_EXISTENCE,
        "Spatial Relationship": Mode.SPATIAL_RELATIONSHIP,
    }

    selected_mode = mode_map[mode_choice]
    result = PULS(prompt, [selected_mode])

    # Extract the relevant propositions and spec based on the selected mode
    mode_key = selected_mode.name.lower().replace("_", " ")
    if mode_key == "object action alignment":
        mode_key = "object_action_alignment"
    elif mode_key == "overall consistency":
        mode_key = "overall_consistency"
    elif mode_key == "object existence":
        mode_key = "object_existence"
    elif mode_key == "spatial relationship":
        mode_key = "spatial_relationships"

    propositions = result.get(mode_key, [])
    spec = result.get(f"{mode_key}_spec", "")

    return ", ".join(propositions), spec


# Gradio interface
def demo_interface(video, propositions, tl):
    """Wrapper for the Gradio interface."""
    return process_video(video, propositions, tl)


# Example data from the original script
example_video_path_1 = (
    "assets/A_storm_bursts_in_with_intermittent_lightning_and_causes_flooding_and_large_waves_crash_in.mp4"
)
example_video_path_2 = "assets/The ocean waves gently lapping at the shore, until a storm bursts in, and then lightning flashes across the sky.mp4"
example_propositions = "waves lapping,ocean shore,storm bursts in,lightning on the sky"
example_tl = '("waves_lapping" & "ocean_shore") U ("storm_bursts_in" U "lightning_on_the_sky")'
example_prompt = (
    "The ocean waves gently lapping at the shore, until a storm bursts in, and then lightning flashes across the sky"
)

with gr.Blocks(title="NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models") as demo:
    gr.HTML(
        """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification 🤗</h1>
        <h2>Published at CVPR 2025</h2>
    </div>
    <div style="text-align: center; margin-bottom: 20px;">
        <p>
            <a href="https://arxiv.org/abs/2411.16718">📜 Paper</a> | 
            <a href="https://github.com/UTAustin-SwarmLab/NeuS-V">💻 GitHub</a> | 
            <a href="https://utaustin-swarmlab.github.io/NeuS-V">🌐 Project Page</a>
        </p>
    </div>
    <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
        ⚠️ This demo is for academic research and experiential use only. 
    </div>

    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            prompt_input = gr.Textbox(
                label="Text-to-Video Prompt",
                placeholder="The prompt used to generate the video...",
            )
            gr.Markdown(
                "You can either manually enter propositions and temporal logic specification below, or use the button below to automatically generate them from your text prompt using Prompt Understanding via Temporal Logic Specification (PULS). Please refer to our paper for more details."
            )

            with gr.Row():
                use_puls_checkbox = gr.Checkbox(label="Use PULS to auto-generate propositions and TL spec", value=False)
                mode_choice = gr.Dropdown(
                    choices=[
                        "Overall Consistency (default)",
                        "Object Existence",
                        "Spatial Relationship",
                        "Object-Action Alignment",
                    ],
                    label="PULS Mode",
                    visible=False,
                )
                generate_btn = gr.Button("Generate Propositions & TL Spec", visible=False)

            propositions_input = gr.Textbox(label="List of Propositions (comma-separated)", placeholder="A, B, C")
            tl_input = gr.Textbox(
                label="Temporal Logic Specification", placeholder="(A & B) U C - means A and B hold until C occurs"
            )

            process_btn = gr.Button("Process Video", variant="primary")

        with gr.Column():
            output_score = gr.Textbox(label="NeuS-V Score")
            gr.Markdown(
                """
                #### About the Score
                The NeuS-V score (0-1) measures how well your video matches the specified temporal logic conditions. A higher score indicates better alignment with the expected sequence of events.
                """
            )

    # Show/hide PULS controls based on checkbox
    use_puls_checkbox.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=[use_puls_checkbox],
        outputs=[mode_choice, generate_btn],
    )

    # Generate propositions and TL spec from natural language
    generate_btn.click(
        fn=generate_from_puls, inputs=[prompt_input, mode_choice], outputs=[propositions_input, tl_input]
    )

    # Process video with current propositions and TL spec
    process_btn.click(fn=demo_interface, inputs=[video_input, propositions_input, tl_input], outputs=[output_score])

    # Examples
    gr.Examples(
        examples=[
            [example_video_path_1, example_prompt, "Overall Consistency (default)", example_propositions, example_tl],
            [example_video_path_2, example_prompt, "Overall Consistency (default)", example_propositions, example_tl],
        ],
        inputs=[video_input, prompt_input, mode_choice, propositions_input, tl_input],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=["assets/"], server_name="0.0.0.0", server_port=7860)
