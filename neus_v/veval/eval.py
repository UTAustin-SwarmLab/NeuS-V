import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from neus_v.automaton.video_automaton import VideoAutomaton
from neus_v.model_checking.stormpy import StormModelChecker
from neus_v.veval.parse import parse_tl_formula, parse_until_to_next_frame
from neus_v.video.frame import VideoFrame
from neus_v.video.read_video import read_video


def create_frame_windows(frames: list, window_size: int) -> list[list]:
    """Create non-overlapping windows of frames, with remainder in last window.

    Args:
        frames: List of frames
        window_size: Size of each window

    Returns:
        List of frame windows
    """
    windows = []
    for i in range(0, len(frames), window_size):
        windows.append(frames[i : i + window_size])
    return windows


def evaluate_video(
    vision_language_model,
    confidence_as_token_probability: bool,
    video_path: Path | str,
    proposition_set: list,
    tl_spec: str,
    parallel_inference: bool = False,
    threshold: float = 0.1,
    num_of_frame_in_sequence: int = 1,
) -> dict:
    """Evaluate a video using the given vision language model."""
    output_log = {
        "specification": None,
        "propositions": None,
        "probability": None,
        "min_probability": None,
        "max_probability": None,
        "propositions_avg_probability": {},
    }

    if isinstance(video_path, str):
        video_path = Path(video_path)
    video = read_video(video_path=video_path)

    # TODO: if there's F in the tl_spec
    ltl_formula = parse_tl_formula(tl_spec)
    ltl_formula = parse_until_to_next_frame(ltl_formula)

    video_automaton = VideoAutomaton(include_initial_state=True)

    video_automaton.set_up(proposition_set=proposition_set)
    model_checker = StormModelChecker(
        proposition_set=proposition_set,
        ltl_formula=ltl_formula,
    )

    proposition_probability_record = {}
    for proposition in proposition_set:
        proposition_probability_record[proposition] = []
    if model_checker.validate_tl_specification(ltl_formula):
        frame_count = 0
        all_frames: list[np.ndarray] = video.get_all_frames_of_video(
            return_format="ndarray",
            desired_interval_in_sec=1,
        )
        try:
            # for frame_img in all_frames:

            def process_frame(frame_img: np.ndarray, frame_count: int):
                sys.stdout.write(f"\rProcessing frame: {frame_count+1}/{len(all_frames)} ")
                sys.stdout.flush()
                object_of_interest = {}
                for proposition in proposition_set:
                    detected_object = vision_language_model.detect(
                        frame_img=frame_img,
                        scene_description=proposition,
                        confidence_as_token_probability=confidence_as_token_probability,
                        threshold=threshold,
                    )
                    object_of_interest[proposition] = detected_object
                    # proposition_probability_record.get(proposition).append(
                    #     detected_object.probability
                    # )
                video_frame = VideoFrame(
                    frame_idx=frame_count,
                    timestamp=None,
                    frame_image=frame_img,
                    object_of_interest=object_of_interest,
                )
                return video_frame, object_of_interest

            if parallel_inference:
                frame_windows = create_frame_windows(frames=all_frames, window_size=num_of_frame_in_sequence)
                results = Parallel(n_jobs=len(all_frames))(
                    delayed(process_frame)(frame_img, i) for i, frame_img in enumerate(all_frames)
                )
            else:
                frame_windows = create_frame_windows(frames=all_frames, window_size=num_of_frame_in_sequence)
                results = [process_frame(frame_img, i) for i, frame_img in enumerate(all_frames)]

            for video_frame, object_of_interest in results:
                video_automaton.add_frame(frame=video_frame)
                for proposition, detected_object in object_of_interest.items():
                    proposition_probability_record[proposition].append(detected_object.probability)

            video_automaton.add_terminal_state(add_with_terminal_label=True)
            sys.stdout.write("\n")  # Move to the next line after processing all frames
            result = model_checker.check_automaton(
                states=video_automaton.states,
                transitions=video_automaton.transitions,
                model_type="dtmc",
                use_filter=True,
            )
            output_log["specification"] = tl_spec
            output_log["propositions"] = proposition_set
            output_log["probability"] = round(float(str(result)), 6)
            output_log["min_probability"] = round(float(str(result.min)), 6)
            output_log["max_probability"] = round(float(str(result.max)), 6)
            for (
                proposition,
                probabilities,
            ) in proposition_probability_record.items():
                avg_probability = sum(probabilities) / len(probabilities)
                output_log["propositions_avg_probability"][proposition] = round(avg_probability, 3)
        except Exception as e:  # noqa: BLE001
            # print(f"\nError processing frame {frame_count}: {e}")
            import traceback

            print(f"\nError processing frame {frame_count}: {e}")
            traceback.print_exc()

    return output_log


def evaluate_video_with_sequence_of_images(
    vision_language_model,
    confidence_as_token_probability: bool,
    video_path: Path | str,
    proposition_set: list,
    tl_spec: str,
    parallel_inference: bool = False,
    num_of_frame_in_sequence: int = 3,
    threshold: float = 0.1,
) -> dict:
    """Evaluate a video using the given vision language model."""
    output_log = {
        "specification": None,
        "propositions": None,
        "probability": None,
        "min_probability": None,
        "max_probability": None,
        "propositions_avg_probability": {},
    }

    if isinstance(video_path, str):
        video_path = Path(video_path)
    video = read_video(video_path=video_path)

    # TODO: if there's F in the tl_spec
    ltl_formula = parse_tl_formula(tl_spec)
    ltl_formula = parse_until_to_next_frame(ltl_formula)

    video_automaton = VideoAutomaton(include_initial_state=True)

    video_automaton.set_up(proposition_set=proposition_set)
    model_checker = StormModelChecker(
        proposition_set=proposition_set,
        ltl_formula=ltl_formula,
    )

    proposition_probability_record = {}
    for proposition in proposition_set:
        proposition_probability_record[proposition] = []
    if model_checker.validate_tl_specification(ltl_formula):
        frame_count = 0
        all_frames: list[np.ndarray] = video.get_all_frames_of_video(
            return_format="ndarray",
            desired_interval_in_sec=0.5,
        )
        try:
            # for frame_img in all_frames:
            def process_frame(sequence_of_frames: list[np.ndarray], frame_count: int):
                sys.stdout.write(f"\rProcessing frame window: {frame_count+1}/{len(frame_windows)} ")
                sys.stdout.flush()
                object_of_interest = {}
                for proposition in proposition_set:
                    detected_object = vision_language_model.detect(
                        seq_of_frames=sequence_of_frames,
                        scene_description=proposition,
                        # confidence_as_token_probability=confidence_as_token_probability,
                        threshold=threshold,
                    )
                    object_of_interest[proposition] = detected_object
                    # proposition_probability_record.get(proposition).append(
                    #     detected_object.probability
                    # )
                    print(f"{proposition}: {detected_object.probability}")
                video_frame = VideoFrame(
                    frame_idx=frame_count,
                    timestamp=None,
                    frame_image=sequence_of_frames,
                    object_of_interest=object_of_interest,
                )
                return video_frame, object_of_interest

            if parallel_inference:
                frame_windows = create_frame_windows(frames=all_frames, window_size=num_of_frame_in_sequence)
                results = Parallel(n_jobs=len(frame_windows))(
                    delayed(process_frame)(frame_img, i) for i, frame_img in enumerate(frame_windows)
                )
            else:
                frame_windows = create_frame_windows(frames=all_frames, window_size=num_of_frame_in_sequence)
                results = [process_frame(sequence_of_frames, i) for i, sequence_of_frames in enumerate(frame_windows)]

            for video_frame, object_of_interest in results:
                video_automaton.add_frame(frame=video_frame)
                for proposition, detected_object in object_of_interest.items():
                    proposition_probability_record[proposition].append(detected_object.probability)

            video_automaton.add_terminal_state(add_with_terminal_label=False)
            sys.stdout.write("\n")  # Move to the next line after processing all frames
            result = model_checker.check_automaton(
                states=video_automaton.states,
                transitions=video_automaton.transitions,
                model_type="dtmc",
                use_filter=False,
            )
            output_log["specification"] = tl_spec
            output_log["propositions"] = proposition_set
            output_log["probability"] = round(float(str(result.at(0))), 6)
            output_log["min_probability"] = round(float(str(result.min)), 6)
            output_log["max_probability"] = round(float(str(result.max)), 6)
            for (
                proposition,
                probabilities,
            ) in proposition_probability_record.items():
                avg_probability = sum(probabilities) / len(probabilities)
                output_log["propositions_avg_probability"][proposition] = round(avg_probability, 3)
        except Exception as e:  # noqa: BLE001
            # print(f"\nError processing frame {frame_count}: {e}")
            import traceback

            print(f"\nError processing frame {frame_count}: {e}")
            traceback.print_exc()

    return output_log
