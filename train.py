"""FOCUS training with local TAO-format data and a Hugging Face or local base model."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import torch
from peft import LoraConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"


def local_path(value):
    if "://" in value:
        raise argparse.ArgumentTypeError("must be a local filesystem path")
    return Path(value).expanduser()


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_model_arguments(parser):
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model_revision",
        help="Hub revision. The default Qwen model uses the pinned release revision.",
    )
    parser.add_argument("--cache_dir", type=local_path, help="Optional Hugging Face download cache.")
    parser.add_argument(
        "--local_files_only", action="store_true",
        help="Use only a local model directory or already-cached Hub files.",
    )


def model_source(args):
    name = args.model_name_or_path
    if "://" in name:
        raise ValueError("Use a Hugging Face model ID or an existing local model directory")
    path = Path(name).expanduser()
    if path.is_dir():
        return str(path.resolve()), {"local_files_only": True}
    if path.exists() or name.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Local model directory not found: {path}")
    return name, {
        "revision": args.model_revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else None),
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_annotations", type=local_path, required=True)
    parser.add_argument("--train_frames", type=local_path, required=True)
    add_model_arguments(parser)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--target_ratio", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.025)
    parser.add_argument("--n_shot", type=positive_int, default=4)
    parser.add_argument("--output_root", type=local_path, default=ROOT / "outputs" / "train")
    parser.add_argument("--deepspeed", type=local_path)
    parser.add_argument("--num_train_epochs", type=float, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Read all selected images and build examples without loading model weights.",
    )
    args = parser.parse_args(argv)
    for name in ("alpha", "beta", "gamma", "margin", "target_ratio", "num_train_epochs"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    if args.target_ratio <= 0 or args.num_train_epochs <= 0:
        parser.error("--target_ratio and --num_train_epochs must be positive")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("--max_steps must be positive or -1")
    return args


def select_track(data, video_id):
    track_ids = [
        annotation["track_id"]
        for annotation in data["annotations"]
        if annotation["video_id"] == video_id
    ]
    if not track_ids:
        raise ValueError(f"No annotated tracks found for video {video_id}")
    track_id = Counter(track_ids).most_common(1)[0][0]
    for track in data["tracks"]:
        if track["id"] == track_id and track["video_id"] == video_id:
            return track
    raise ValueError(f"Selected track {track_id} is missing metadata for video {video_id}")


def select_frames(data, track_id, k, video_id):
    if k < 1:
        raise ValueError("k must be at least 1")
    frames = {frame["id"]: frame for frame in data["images"]}
    candidates = []
    for annotation in data["annotations"]:
        if (annotation["video_id"], annotation["track_id"]) != (video_id, track_id):
            continue
        frame = frames[annotation["image_id"]]
        if frame["video_id"] != video_id:
            raise ValueError("Annotation and image refer to different videos")
        candidates.append((annotation, frame))

    candidates.sort(key=lambda pair: pair[1]["frame_index"])
    if len(candidates) < k + 1:
        raise ValueError(
            f"Need {k + 1} annotated frames for {k}-shot track {track_id}, "
            f"found {len(candidates)}"
        )
    selected = [candidates[round(i * (len(candidates) - 1) / k)] for i in range(k + 1)]
    if len({frame["id"] for _, frame in selected}) != k + 1:
        raise ValueError("Selected frame IDs are not distinct")
    return selected


def transform_image(path, box, target_pixels, expected_size):
    with Image.open(path) as source:
        source.load()
        if source.size != expected_size:
            raise ValueError(f"Image dimensions differ from annotation metadata: {path}")
        image = source.convert("RGB")
    width, height = image.size
    if width * height > target_pixels:
        scale = (target_pixels / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    x_min, y_min, x_max, y_max = box
    normalized = [
        int(1000 * x_min / width), int(1000 * y_min / height),
        int(1000 * x_max / width), int(1000 * y_max / height),
    ]
    return image, normalized


def make_messages(boxes):
    content = [{
        "type": "text",
        "text": (
            "Task: Track the same object across the sequence of frames below. "
            "Your goal is to follow the target object consistently throughout the sequence.\n\n"
            f"For the first {len(boxes)-1} frames, the bounding box of the object is already provided. "
            "Use this information and the visual context to predict where the same object appears "
            "in the final frame.\n\n"
            "Output the predicted bounding box for the last frame in the following format:\n"
            "<answer>[x_min, y_min, x_max, y_max]</answer>\n\n"
            "Make sure your prediction accurately matches the same object described in the scene."
        ),
    }]
    for index, box in enumerate(boxes):
        content.append({"type": "image"})
        if index < len(boxes) - 1:
            text = f"Frame {index+1}: The object is located at bounding box {box}"
        else:
            text = (
                f"Frame {index+1}: Predict the bounding box for the object. "
                "Respond only with <answer>[x_min, y_min, x_max, y_max]</answer>."
            )
        content.append({"type": "text", "text": text})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [
            {"type": "text", "text": f"<answer>{boxes[-1]}</answer>"},
        ]},
    ]


class TaoDataset(Dataset):
    def __init__(self, annotations, frames_root, n_shot=1, target_ratio=0.3):
        if n_shot < 1 or not math.isfinite(target_ratio) or target_ratio <= 0:
            raise ValueError("n_shot and target_ratio must be positive")
        self.target_pixels = max(1, int(1280 * 720 * target_ratio))
        data = json.loads(Path(annotations).read_text())
        grouped = {}
        for key in ("images", "annotations", "tracks"):
            grouped[key] = defaultdict(list)
            for record in data[key]:
                grouped[key][record["video_id"]].append(record)
        self.examples = []
        for video in data["videos"]:
            if PurePosixPath(video["name"]).parts[0] != "train":
                raise ValueError(f"Video {video['id']} is not in the training split")
            video_data = {key: values[video["id"]] for key, values in grouped.items()}
            track = select_track(video_data, video["id"])
            selected = select_frames(video_data, track["id"], n_shot, video["id"])
            paths, sizes, boxes = [], [], []
            for annotation, frame in selected:
                relative = PurePosixPath(frame["file_name"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe image path: {relative}")
                relative.relative_to(video["name"])
                path = Path(frames_root) / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                if annotation["category_id"] != track["category_id"]:
                    raise ValueError(f"Annotation {annotation['id']} category differs from track")
                x, y, width, height = annotation["bbox"]
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    raise ValueError(f"Annotation {annotation['id']} has non-finite coordinates")
                paths.append(path)
                sizes.append((frame["width"], frame["height"]))
                boxes.append([x, y, x + width, y + height])
            self.examples.append({
                "video_id": video["id"], "video_name": video["name"], "track_id": track["id"],
                "paths": paths, "sizes": sizes, "boxes": boxes,
            })
        if not self.examples:
            raise ValueError(f"No examples found in {annotations}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        images, boxes = [], []
        for path, box, size in zip(example["paths"], example["boxes"], example["sizes"]):
            image, normalized = transform_image(path, box, self.target_pixels, size)
            images.append(image)
            boxes.append(normalized)
        return {"images": images, "messages": make_messages(boxes)}


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, alpha, beta, gamma, margin=0.05, **kwargs):
        self.alpha, self.beta, self.gamma, self.margin = alpha, beta, gamma, margin
        self.needs_attention = any(weight != 0 for weight in (alpha, beta, gamma))
        super().__init__(*args, **kwargs)
        # This custom loss does not consume num_items_in_batch.
        self.model_accepts_loss_kwargs = False
        if self.needs_attention:
            tokenizer = self.processing_class.tokenizer
            delimiters = {
                "vision_start": "<|vision_start|>", "vision_end": "<|vision_end|>",
                "support_start": " [", "support_end": "]",
                "query_start": ">[", "query_end": "]</", "comma": ",",
            }
            encoded = {
                name: tokenizer.encode(text, add_special_tokens=False)
                for name, text in delimiters.items()
            }
            if any(len(tokens) != 1 for tokens in encoded.values()):
                raise ValueError("Attention losses require the Qwen2-VL bounding-box tokenization")
            self.token_ids = {name: tokens[0] for name, tokens in encoded.items()}

    def attention_masks(self, input_ids):
        positions = {
            name: (input_ids == token_id).nonzero(as_tuple=True)[0]
            for name, token_id in self.token_ids.items()
        }
        starts, ends = positions["vision_start"], positions["vision_end"]
        num_support = len(starts) - 1
        if num_support < 1 or len(starts) != len(ends):
            raise ValueError("Expected support images followed by one query image")
        if (
            len(positions["support_start"]) != num_support
            or len(positions["support_end"]) != num_support
            or len(positions["query_start"]) == 0
            or len(positions["query_end"]) == 0
            or positions["query_start"][-1] <= ends[-1]
            or positions["query_end"][-1] <= positions["query_start"][-1]
        ):
            raise ValueError("Tokenized bounding boxes do not match the support/query images")
        masks = {
            name: torch.zeros_like(input_ids, dtype=torch.bool)
            for name in ("support_image", "query_image", "support_bbox", "query_bbox")
        }
        for index in range(num_support):
            masks["support_image"][starts[index] + 1:ends[index]] = True
            masks["support_bbox"][
                positions["support_start"][index] + 1:positions["support_end"][index]
            ] = True
        masks["query_image"][starts[-1] + 1:ends[-1]] = True
        # Qwen merges the query delimiters into ">[" and "]</", unlike support boxes.
        masks["query_bbox"][positions["query_start"][-1] + 1:positions["query_end"][-1]] = True
        for name in ("support_bbox", "query_bbox"):
            masks[name][input_ids == self.token_ids["comma"]] = False
        if any(not mask.any() for mask in masks.values()):
            raise ValueError("An attention-loss region contains no tokens")
        return masks

    def rowwise_preference_loss(self, attention, mask, eps=1e-12):
        probabilities = attention / (attention.sum(dim=-1, keepdim=True) + eps)
        probabilities = probabilities + eps
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        positive = mask.float().unsqueeze(0).expand_as(probabilities)
        positive_probability = (probabilities * positive).sum(dim=-1) / (
            positive.sum(dim=-1) + eps
        )
        negative_probability = (probabilities * (1 - positive)).sum(dim=-1) / (
            (1 - positive).sum(dim=-1) + eps
        )
        return torch.relu(self.margin - (positive_probability - negative_probability)).square().mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, output_attentions=self.needs_attention)
        loss = outputs.loss
        if self.needs_attention:
            if not outputs.attentions or any(value is None for value in outputs.attentions):
                raise ValueError("Attention losses require the model's eager attention backend")
            attention = torch.stack(outputs.attentions, dim=0).mean(dim=2).mean(dim=0)
            auxiliary = []
            for batch_index in range(attention.size(0)):
                masks = self.attention_masks(inputs["input_ids"][batch_index])
                sample_loss = attention[batch_index].sum() * 0
                for weight, source, target in (
                    (self.alpha, "query_image", "support_image"),
                    (self.beta, "query_image", "support_bbox"),
                    (self.gamma, "query_bbox", "support_bbox"),
                ):
                    if weight != 0:
                        sample_loss = sample_loss + weight * self.rowwise_preference_loss(
                            attention[batch_index][masks[source]], masks[target],
                        )
                auxiliary.append(sample_loss)
            loss = loss + torch.stack(auxiliary).mean()
        return (loss, outputs) if return_outputs else loss


def main(argv=None):
    args = parse_args(argv)
    if (
        not args.dry_run
        and int(os.environ.get("WORLD_SIZE", "1")) <= 1
        and torch.cuda.device_count() > 1
    ):
        raise RuntimeError(
            "Single-process multi-GPU DataParallel cannot split Qwen2-VL image patches "
            "and image grids correctly. Use torchrun for distributed training, or set "
            "CUDA_VISIBLE_DEVICES to a single GPU."
        )
    if not args.dry_run:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if local_world_size > torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {local_world_size} local workers but only "
                f"{torch.cuda.device_count()} CUDA devices are visible."
            )
        if args.output_root.exists() and (
            not args.output_root.is_dir() or any(args.output_root.iterdir())
        ):
            raise FileExistsError(
                f"Output directory is not empty: {args.output_root}. "
                "Choose a new --output_root; existing checkpoints are never overwritten."
            )
    train_dataset = TaoDataset(
        args.train_annotations, args.train_frames, args.n_shot, args.target_ratio,
    )
    print(f"TAO training examples: {len(train_dataset)}", flush=True)
    if args.dry_run:
        for index in range(len(train_dataset)):
            train_dataset[index]
        print(f"Loaded all {len(train_dataset)} train examples ({args.n_shot + 1} images each)")
        print(f"First train target: {train_dataset[0]['messages'][-1]['content'][0]['text']}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. Use --dry_run to check the data on CPU.")

    # ZeRO-3 must be configured before from_pretrained to shard weight initialization.
    config = SFTConfig(
        output_dir=str(args.output_root),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=None,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        save_strategy="epoch",
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        deepspeed=str(args.deepspeed) if args.deepspeed is not None else None,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    needs_attention = any(weight != 0 for weight in (args.alpha, args.beta, args.gamma))
    model_path, model_options = model_source(args)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if needs_attention else "sdpa",
        **model_options,
    )
    model.config.use_cache = False
    processor = Qwen2VLProcessor.from_pretrained(
        model_path, use_fast=False, **model_options,
    )
    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.0, r=8, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = CustomSFTTrainer(
        model=model, args=config, train_dataset=train_dataset,
        peft_config=peft_config, processing_class=processor,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, margin=args.margin,
    )
    trainer.train()
    # Every rank participates in ZeRO-3 gathering; only rank zero writes the processor.
    trainer.save_model(str(args.output_root))
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_root)


if __name__ == "__main__":
    main()
