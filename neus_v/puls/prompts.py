from enum import Enum

class Mode(Enum):
    OBJECT_ACTION_ALIGNMENT = 1
    OBJECT_EXISTENCE = 2
    OVERALL_CONSISTENCY = 3
    SPATIAL_RELATIONSHIP = 4

mode_prompts = {
    Mode.OBJECT_ACTION_ALIGNMENT: (
        "\"object_action_alignment\":\n"
        "Extract actions and their participating objects. Each proposition must describe an action and its related objects.\n"
        "Example:\n"
        "\"object_action_alignment\": [\"person holds hotdog\", \"person walks\"]"
    ),
    Mode.OBJECT_EXISTENCE: (
        "\"object_existence\":\n"
        "Extract only the tangible objects mentioned in the prompt.\n"
        "Example:\n"
        "\"object_existence\": [\"person\", \"hotdog\", \"car\", \"truck\"]"
    ),
    Mode.OVERALL_CONSISTENCY: (
        "\"overall_consistency\":\n"
        "Extract all meaningful event propositions that describe the combined semantics of objects, actions, and spatial relationships — "
        "but avoid TL keywords such as 'and', 'or', 'not', 'until', 'eventually'.\n"
        "Example:\n"
        "\"overall_consistency\": [\"person holds hotdog\", \"person walks\", \"car next to truck\"]"
    ),
    Mode.SPATIAL_RELATIONSHIP: (
        "\"spatial_relationships\":\n"
        "Extract only spatial relationships between tangible objects (e.g., \"object A next to object B\"). Do not infer or hallucinate spatial relationships.\n"
        "Example:\n"
        "\"spatial_relationships\": [\"car next to truck\"]"
    )
}

mode_outputs = {
    Mode.OBJECT_ACTION_ALIGNMENT: (
        "  \"object_action_alignment\": [...],\n"
        "  \"object_action_alignment_spec\": \"...\","
    ),
    Mode.OBJECT_EXISTENCE: (
        "  \"object_existence\": [...],\n"
        "  \"object_existence_spec\": \"...\","
    ),
    Mode.OVERALL_CONSISTENCY: (
        "  \"overall_consistency\": [...],\n"
        "  \"overall_consistency_spec\": \"...\","
    ),
    Mode.SPATIAL_RELATIONSHIP: (
        "  \"spatial_relationships\": [...],\n"
        "  \"spatial_relationships_spec\": \"...\""
    )
}

header = (
    "You are an intelligent agent designed to extract structured representations from video description prompts. "
    "You will operate in two stages: (1) proposition extraction and (2) TL specification generation.\n\n"
)

stage1_intro = (
    "Stage 1: Proposition Extraction\n\n"
    "Given an input prompt summarizing a video, extract atomic propositions in the following four modes. "
    "Return all outputs in JSON format.\n\n"
)

stage2_intro = "Stage 2: TL Specification Generation\n\n"

spec_gen_intro = (
    "For each of the {n} list(s) of propositions extracted in Stage 1, generate a separate Temporal Logic (TL) specification "
    "describing the structure or sequence of events in that list.\n\n"
)

tl_instructions = (
    "Rules for TL specification:\n"
    "- The input is a single list of propositions from one of the extraction modes.\n"
    "- The output is a single TL formula using **only** the propositions from that list and the allowed TL symbols: "
    "['AND', 'OR', 'NOT', 'UNTIL', 'ALWAYS', 'EVENTUALLY']\n"
    "- Do not introduce any new propositions.\n"
    "- Each formula should reflect the temporal or logical relationships between the propositions in a way that makes semantic sense.\n\n"
)

input_template = "Input:\n{{\n  \"prompt\": \"{}\"\n}}\n\n"
expected_output_header = "Expected Output:\n{\n"
expected_output_footer = "\n}"
