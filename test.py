"""Evaluate a FOCUS LoRA checkpoint using local TAO-format test data."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

if __package__:
    from . import train as training
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as training

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_step", type=training.positive_int, help="Saved optimizer step, e.g. 96.")
    parser.add_argument("--checkpoint_root", type=training.local_path, default=ROOT / "checkpoints")
    parser.add_argument("--output_root", type=training.local_path, default=ROOT / "results")
    training.add_model_arguments(parser)
    parser.add_argument("--eval_annotations", type=training.local_path, required=True)
    parser.add_argument("--eval_frames", type=training.local_path, required=True)
    parser.add_argument("--n_shot", type=training.positive_int, default=1)
    parser.add_argument("--target_ratio", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=training.positive_int, default=50)
    parser.add_argument("--max_examples", type=training.positive_int, help="Evaluate only this many test examples.")
    parser.add_argument("--dry_run", action="store_true", help="Check checkpoint and test images without loading weights.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.target_ratio) or args.target_ratio <= 0:
        parser.error("--target_ratio must be finite and positive")
    return args


def checkpoint_path(root, step):
    checkpoint = root / f"checkpoint-{step}"
    for name in (
        "adapter_config.json", "adapter_model.safetensors",
        "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "chat_template.jinja",
    ):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Checkpoint {step} is missing or incomplete: {path}. "
                "Wait for the checkpoint to finish saving, or select an existing step."
            )
    state_path = checkpoint / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("global_step") != step:
            raise ValueError(f"Checkpoint trainer_state.json does not match requested step {step}")
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Expected a LoRA adapter checkpoint: {checkpoint}")
    return checkpoint.resolve()


def load_test_examples(annotations, frames_root, n_shot):
    data = json.loads(Path(annotations).read_text())
    grouped = {key: defaultdict(list) for key in ("images", "annotations", "tracks")}
    for key, groups in grouped.items():
        for record in data[key]:
            groups[record["video_id"]].append(record)
    examples = []
    for video in data["videos"]:
        video_name = PurePosixPath(video["name"])
        if not video_name.parts or video_name.parts[0] != "test":
            raise ValueError(f"Video {video['id']} is not in the test split")
        video_data = {key: groups[video["id"]] for key, groups in grouped.items()}
        track = training.select_track(video_data, video["id"])
        selected = training.select_frames(video_data, track["id"], n_shot, video["id"])
        paths, sizes, boxes = [], [], []
        for annotation, frame in selected:
            relative = PurePosixPath(frame["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image path: {relative}")
            relative.relative_to(video_name)
            path = Path(frames_root) / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if annotation["category_id"] != track["category_id"]:
                raise ValueError(f"Annotation {annotation['id']} category differs from track")
            x, y, width, height = annotation["bbox"]
            if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError(f"Annotation {annotation['id']} has an invalid bounding box")
            paths.append(path)
            sizes.append((frame["width"], frame["height"]))
            boxes.append([x, y, x + width, y + height])
        examples.append({
            "video_id": video["id"], "video_name": video["name"], "track_id": track["id"],
            "paths": paths, "sizes": sizes, "boxes": boxes,
        })
    if not examples:
        raise ValueError(f"No test examples found in {annotations}")
    return examples


def prepare_example(example, target_ratio):
    images, boxes = [], []
    target_pixels = max(1, int(1280 * 720 * target_ratio))
    for path, box, size in zip(example["paths"], example["boxes"], example["sizes"]):
        image, normalized = training.transform_image(path, box, target_pixels, size)
        images.append(image)
        boxes.append(normalized)
    # The held-out query answer must never enter the generation prompt.
    messages = training.make_messages(boxes)[:-1]
    return images, messages, boxes[-1]


def parse_bbox(text):
    text = text.strip()
    if text.startswith("<answer>") and text.endswith("</answer>"):
        text = text[len("<answer>"):-len("</answer>")].strip()
    try:
        box = json.loads(text, parse_int=float)
    except json.JSONDecodeError as error:
        raise ValueError("Expected a JSON bounding box, optionally wrapped in <answer> tags") from error
    if isinstance(box, list) and len(box) == 2 and all(
        isinstance(point, list) and len(point) == 2 for point in box
    ):
        box = box[0] + box[1]
    if not isinstance(box, list) or len(box) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in box
    ):
        raise ValueError("Bounding box must contain four finite numbers")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Bounding box must have positive width and height")
    if not math.isfinite((box[2] - box[0]) * (box[3] - box[1])):
        raise ValueError("Bounding box area must be finite")
    return box


def box_iou(prediction, target):
    overlap_width = max(0, min(prediction[2], target[2]) - max(prediction[0], target[0]))
    overlap_height = max(0, min(prediction[3], target[3]) - max(prediction[1], target[1]))
    intersection = overlap_width * overlap_height
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    if target_area <= 0:
        raise ValueError("Ground-truth bounding box has zero area after coordinate normalization")
    return intersection / (prediction_area + target_area - intersection)


def generate_bbox(model, processor, images, messages, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=False,
        )
    completion = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        completion, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def main(argv=None):
    args = parse_args(argv)
    checkpoint = checkpoint_path(args.checkpoint_root, args.checkpoint_step)
    output = args.output_root / checkpoint.name
    if not args.dry_run:
        for name in ("predictions.jsonl", "metrics.json"):
            if (output / name).exists():
                raise FileExistsError(f"Refusing to overwrite {output / name}; select another --output_root")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for evaluation. Use --dry_run for a CPU data check.")
    examples = load_test_examples(args.eval_annotations, args.eval_frames, args.n_shot)
    total_examples = len(examples)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"TAO test examples: {len(examples)} / {total_examples}", flush=True)
    if args.dry_run:
        for example in examples:
            prepare_example(example, args.target_ratio)
        print(f"Checked all {len(examples)} selected test examples; no model loaded or evaluation run.")
        return

    processor = Qwen2VLProcessor.from_pretrained(
        str(checkpoint), use_fast=False, local_files_only=True,
    )
    model_path, model_options = training.model_source(args)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **model_options,
    )
    model = PeftModel.from_pretrained(
        model, str(checkpoint), is_trainable=False, local_files_only=True,
    ).to("cuda").eval()
    model.config.use_cache = True

    output.mkdir(parents=True, exist_ok=True)
    iou_sum, invalid_predictions = 0.0, 0
    with (output / "predictions.jsonl").open("x") as predictions:
        for index, example in enumerate(tqdm(examples, desc=f"checkpoint-{args.checkpoint_step}"), 1):
            images, messages, target = prepare_example(example, args.target_ratio)
            response = generate_bbox(model, processor, images, messages, args.max_new_tokens)
            error_message = None
            try:
                prediction = parse_bbox(response)
            except ValueError as error:
                prediction, iou = None, 0.0
                error_message = str(error)
                invalid_predictions += 1
                print(f"Invalid prediction for video {example['video_id']}: {error}", file=sys.stderr)
            else:
                iou = box_iou(prediction, target)
            iou_sum += iou
            record = {
                "video_id": example["video_id"], "video_name": example["video_name"],
                "track_id": example["track_id"], "query_image": str(example["paths"][-1]),
                "prediction": prediction, "ground_truth": target, "iou": iou,
                "response": response, "error": error_message,
            }
            predictions.write(json.dumps(record, allow_nan=False) + "\n")
            predictions.flush()
            tqdm.write(
                f"[{index}/{len(examples)}] video={example['video_id']} response={response}\n"
                f"prediction={prediction} ground_truth={target} "
                f"iou={iou:.6f} running_mean_iou={iou_sum / index:.6f}",
                file=sys.stdout,
            )
            sys.stdout.flush()
    metrics = {
        "checkpoint_step": args.checkpoint_step, "checkpoint": str(checkpoint),
        "num_examples": len(examples), "total_test_examples": total_examples,
        "n_shot": args.n_shot, "target_ratio": args.target_ratio,
        "invalid_predictions": invalid_predictions, "mean_iou": iou_sum / len(examples),
    }
    with (output / "metrics.json").open("x") as destination:
        json.dump(metrics, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
