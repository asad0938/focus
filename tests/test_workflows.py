import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftConfig
from PIL import Image
from safetensors import safe_open

import prepare_data
import test as evaluator
import train as training


CHECKPOINT = ROOT / "checkpoints" / "checkpoint-96"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.frames = self.root / "frames"
        self.data = {
            "videos": [{"id": 16, "name": "train/source/video"}],
            "tracks": [{"id": 101, "video_id": 16, "category_id": 1}],
            "categories": [{"id": 1, "name": "object"}],
            "images": [], "annotations": [],
        }
        for image_id in range(5):
            name = f"train/source/video/frame{image_id:04d}.jpg"
            path = self.frames / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 200), "blue").save(path)
            self.data["images"].append({
                "id": image_id, "video_id": 16, "frame_index": image_id * 10,
                "file_name": name, "width": 100, "height": 200,
            })
            self.data["annotations"].insert(0, {
                "id": image_id, "video_id": 16, "track_id": 101,
                "image_id": image_id, "category_id": 1,
                "bbox": [10 + image_id * 5, 20, 30, 40],
            })
        self.annotations = self.root / "annotations.json"
        self.annotations.write_text(json.dumps(self.data))
        self.train_args = [
            "--train_annotations", str(self.annotations), "--train_frames", str(self.frames),
            "--output_root", str(self.root / "training"),
        ]
        self.eval_args = [
            "96", "--eval_annotations", str(self.annotations),
            "--eval_frames", str(self.frames), "--output_root", str(self.root / "results"),
        ]

    def use_test_split(self):
        self.data["videos"][0]["name"] = "test/source/video"
        for image in self.data["images"]:
            old = self.frames / image["file_name"]
            image["file_name"] = image["file_name"].replace("train/", "test/", 1)
            new = self.frames / image["file_name"]
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        self.annotations.write_text(json.dumps(self.data))

    def test_selection_preparation_and_training_for_all_shots(self):
        for shots, expected in ((1, [0, 4]), (2, [0, 2, 4]), (4, [0, 1, 2, 3, 4])):
            with self.subTest(shots=shots):
                manifest = prepare_data.build_manifest(self.data, self.frames, "train", shots)
                self.assertEqual([image["id"] for image in manifest["images"]], expected)
                self.assertEqual(
                    [image["role"] for image in manifest["images"]],
                    ["support"] * shots + ["query"],
                )
                self.annotations.write_text(json.dumps(manifest))
                dataset = training.TaoDataset(self.annotations, self.frames, shots)
                self.assertEqual(len(dataset), 1)
                sample = dataset[0]
                self.assertEqual(len(sample["images"]), shots + 1)
                self.assertEqual(sample["messages"][-1]["content"][0]["text"],
                                 "<answer>[300, 100, 600, 300]</answer>")
                with (
                    patch.object(training.Qwen2VLForConditionalGeneration, "from_pretrained") as model,
                    redirect_stdout(io.StringIO()),
                ):
                    training.main([*self.train_args, "--n_shot", str(shots), "--dry_run"])
                model.assert_not_called()
        self.assertFalse((self.root / "training").exists())

    def test_manifest_exclusions_are_explicit_and_outputs_not_overwritten(self):
        short = copy.deepcopy(self.data)
        short["annotations"] = short["annotations"][:2]
        with redirect_stderr(io.StringIO()) as errors, self.assertRaisesRegex(ValueError, "No videos"):
            prepare_data.build_manifest(short, self.frames, "train", 4)
        self.assertIn("Only 2 distinct annotated frames; need 5", errors.getvalue())
        destination = self.root / "prepared.json"
        args = [
            "--annotations", str(self.annotations), "--frames", str(self.frames),
            "--split", "train", "--n_shot", "4", "--output", str(destination),
        ]
        with redirect_stdout(io.StringIO()):
            prepare_data.main(args)
        original = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_data.main(args)
        self.assertEqual(destination.read_bytes(), original)

    def test_track_selection_matches_frequency_and_encounter_order(self):
        self.data["tracks"].append({"id": 100, "video_id": 16, "category_id": 1})
        self.data["annotations"].insert(0, {**self.data["annotations"][0], "track_id": 100})
        self.assertEqual(training.select_track(self.data, 16)["id"], 101)
        self.data["annotations"] = [
            self.data["annotations"][0], self.data["annotations"][1],
        ]
        self.assertEqual(training.select_track(self.data, 16)["id"], 100)

    def test_resize_is_fixed_budget_and_coordinates_use_original_dimensions(self):
        dataset = training.TaoDataset(self.annotations, self.frames, 4, target_ratio=0.01)
        self.assertLessEqual(dataset[0]["images"][0].width * dataset[0]["images"][0].height, 9216)
        self.assertEqual(dataset[0]["messages"][-1]["content"][0]["text"],
                         "<answer>[300, 100, 600, 300]</answer>")
        path = self.root / "large.jpg"
        Image.new("RGB", (1920, 1200)).save(path)
        for ratio, expected in ((1, False), (2.5, True)):
            image, box = training.transform_image(
                path, [0, 0, 1920, 1200], int(1280 * 720 * ratio), (1920, 1200),
            )
            self.assertEqual(image.size == (1920, 1200), expected)
            self.assertEqual(box, [0, 0, 1000, 1000])

    def test_missing_images_dimensions_and_wrong_shot_counts_fail(self):
        path = self.frames / self.data["images"][0]["file_name"]
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            training.TaoDataset(self.annotations, self.frames)
        with self.assertRaises(FileNotFoundError):
            prepare_data.build_manifest(self.data, self.frames, "train", 4)
        Image.new("RGB", (10, 20)).save(path)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            training.TaoDataset(self.annotations, self.frames)[0]
        with self.assertRaisesRegex(ValueError, "Need 6"):
            training.TaoDataset(self.annotations, self.frames, 5)

    def test_eval_prompts_and_cpu_dry_runs_for_all_shots(self):
        self.use_test_split()
        for shots in (1, 2, 4):
            with self.subTest(shots=shots):
                examples = evaluator.load_test_examples(self.annotations, self.frames, shots)
                images, messages, target = evaluator.prepare_example(examples[0], 0.3)
                self.assertEqual(len(images), shots + 1)
                self.assertEqual(target, [300, 100, 600, 300])
                self.assertEqual([message["role"] for message in messages], ["user"])
                self.assertNotIn(str(target), json.dumps(messages))
                with (
                    patch.object(evaluator.Qwen2VLForConditionalGeneration, "from_pretrained") as model,
                    redirect_stdout(io.StringIO()),
                ):
                    evaluator.main([*self.eval_args, "--n_shot", str(shots), "--dry_run"])
                model.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def test_real_processor_and_attention_spans_for_all_shots(self):
        processor = training.Qwen2VLProcessor.from_pretrained(
            str(CHECKPOINT), local_files_only=True, use_fast=False,
        )

        def initialize(instance, *args, **kwargs):
            instance.processing_class = processor

        with patch.object(training.SFTTrainer, "__init__", initialize):
            trainer = training.CustomSFTTrainer(alpha=0, beta=0.25, gamma=0, margin=0.025)
        self.assertFalse(trainer.model_accepts_loss_kwargs)
        for shots in (1, 2, 4):
            sample = training.TaoDataset(self.annotations, self.frames, shots)[0]
            text = processor.apply_chat_template(sample["messages"], tokenize=False)
            inputs = processor(text=[text], images=sample["images"], return_tensors="pt")
            ids = inputs["input_ids"][0]
            masks = trainer.attention_masks(ids)
            boxes = [
                [100 + 50 * index, 100, 400 + 50 * index, 300]
                for index in (round(i * 4 / shots) for i in range(shots + 1))
            ]
            # Coordinates can span multiple tokens; spaces remain, commas do not.
            box_tokens = [
                [
                    token for token in processor.tokenizer.encode(
                        json.dumps(box)[1:-1], add_special_tokens=False,
                    ) if token != trainer.token_ids["comma"]
                ]
                for box in boxes
            ]
            self.assertEqual(
                ids[masks["support_bbox"]].tolist(),
                [token for box in box_tokens[:-1] for token in box],
            )
            self.assertEqual(ids[masks["query_bbox"]].tolist(), box_tokens[-1])
            self.assertEqual(masks["support_image"].sum().item(),
                             shots * masks["query_image"].sum().item())

    def test_training_configuration_load_order_and_collective_saving(self):
        for rank in range(4):
            with (
                self.subTest(rank=rank),
                patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}),
                patch.object(training.torch.cuda, "is_available", return_value=True),
                patch.object(training.torch.cuda, "device_count", return_value=4),
                patch.object(training.Qwen2VLForConditionalGeneration, "from_pretrained") as model,
                patch.object(training.Qwen2VLProcessor, "from_pretrained") as processor,
                patch.object(training, "SFTConfig") as config,
                patch.object(training, "CustomSFTTrainer") as trainer,
                redirect_stdout(io.StringIO()),
            ):
                order = Mock()
                order.attach_mock(config, "config")
                order.attach_mock(model, "model")
                trainer.return_value.is_world_process_zero.return_value = rank == 0
                training.main([*self.train_args, "--deepspeed", str(ROOT / "zero3.json")])
                self.assertEqual([entry[0] for entry in order.mock_calls[:2]], ["config", "model"])
                self.assertEqual(config.call_args.kwargs, {
                    "output_dir": str(self.root / "training"), "num_train_epochs": 20,
                    "max_steps": -1, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 16, "max_length": None,
                    "optim": "adamw_torch_fused", "learning_rate": 2e-4,
                    "save_strategy": "epoch", "bf16": True, "max_grad_norm": 0.3,
                    "warmup_ratio": 0.03, "deepspeed": str(ROOT / "zero3.json"),
                    "eval_strategy": "no", "report_to": "none", "remove_unused_columns": False,
                })
                self.assertFalse(model.call_args.kwargs["local_files_only"])
                self.assertEqual(model.call_args.kwargs["revision"], training.DEFAULT_REVISION)
                self.assertEqual(model.call_args.kwargs["attn_implementation"], "eager")
                self.assertFalse(model.return_value.config.use_cache)
                self.assertEqual(
                    {key: trainer.call_args.kwargs[key] for key in ("alpha", "beta", "gamma", "margin")},
                    {"alpha": 0, "beta": 0.25, "gamma": 0, "margin": 0.025},
                )
                trainer.return_value.train.assert_called_once_with()
                trainer.return_value.save_model.assert_called_once_with(str(self.root / "training"))
                self.assertEqual(processor.return_value.save_pretrained.call_count, int(rank == 0))

    def test_training_safety_guards_precede_model_loading(self):
        output = self.root / "training"
        output.mkdir()
        marker = output / "existing"
        marker.write_text("preserve")
        for visible, world, error in (
            (4, "1", RuntimeError), (1, "4", RuntimeError), (4, "4", FileExistsError),
        ):
            with (
                patch.dict(os.environ, {"WORLD_SIZE": world, "LOCAL_WORLD_SIZE": world}),
                patch.object(training.torch.cuda, "device_count", return_value=visible),
                patch.object(training.Qwen2VLForConditionalGeneration, "from_pretrained") as model,
                self.assertRaises(error),
            ):
                training.main(self.train_args)
            model.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")

    def test_evaluation_writes_incrementally_and_invalid_predictions_count_as_zero(self):
        self.use_test_split()
        examples = evaluator.load_test_examples(self.annotations, self.frames, 1)
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        terminal = io.StringIO()
        output = self.root / "results" / "checkpoint-96"

        def responses():
            yield "<answer>[300, 100, 600, 300]</answer>"
            self.assertIn("[1/2]", terminal.getvalue())
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 1)
            self.assertFalse((output / "metrics.json").exists())
            yield "invalid"

        with (
            patch.object(evaluator.torch.cuda, "is_available", return_value=True),
            patch.object(evaluator, "load_test_examples", return_value=examples * 2),
            patch.object(evaluator, "generate_bbox", side_effect=responses()),
            patch.object(evaluator.Qwen2VLForConditionalGeneration, "from_pretrained") as base,
            patch.object(evaluator.Qwen2VLProcessor, "from_pretrained") as processor,
            patch.object(evaluator.PeftModel, "from_pretrained", return_value=model) as adapter,
            redirect_stdout(terminal), redirect_stderr(io.StringIO()) as errors,
        ):
            evaluator.main(self.eval_args)
        self.assertFalse(base.call_args.kwargs["local_files_only"])
        self.assertEqual(base.call_args.kwargs["revision"], training.DEFAULT_REVISION)
        processor.assert_called_once_with(str(CHECKPOINT), use_fast=False, local_files_only=True)
        adapter.assert_called_once_with(
            base.return_value, str(CHECKPOINT), is_trainable=False, local_files_only=True,
        )
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual((metrics["num_examples"], metrics["invalid_predictions"], metrics["mean_iou"]),
                         (2, 1, 0.5))
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertIsNone(predictions[-1]["prediction"])
        self.assertEqual(predictions[-1]["iou"], 0)
        self.assertIn("Invalid prediction for video 16", errors.getvalue())
        with self.assertRaises(FileExistsError):
            evaluator.main(self.eval_args)


class ModelAndCheckpointTests(unittest.TestCase):
    def test_public_model_local_directory_cache_and_argument_validation(self):
        argv = ["--train_annotations", "data/train.json", "--train_frames", "data/frames"]
        args = training.parse_args(argv)
        self.assertEqual((args.n_shot, args.beta, args.margin), (4, 0.25, 0.025))
        name, options = training.model_source(args)
        self.assertEqual(name, training.DEFAULT_MODEL)
        self.assertEqual(options["revision"], training.DEFAULT_REVISION)
        self.assertFalse(options["local_files_only"])
        with tempfile.TemporaryDirectory() as directory:
            args.model_name_or_path = directory
            self.assertEqual(training.model_source(args), (directory, {"local_files_only": True}))
            args.model_name_or_path = str(Path(directory) / "missing")
            with self.assertRaises(FileNotFoundError):
                training.model_source(args)
        args = training.parse_args([*argv, "--local_files_only", "--cache_dir", "cache"])
        self.assertTrue(training.model_source(args)[1]["local_files_only"])
        self.assertEqual(training.model_source(args)[1]["cache_dir"], Path("cache"))
        for invalid in (
            ["--n_shot", "0"], ["--target_ratio", "nan"], ["--max_steps", "0"],
            ["--train_frames", "https://example.invalid/frames"],
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args([*argv, *invalid])

    def test_bundled_adapter_is_inference_only_and_has_original_weights(self):
        self.assertEqual(evaluator.checkpoint_path(CHECKPOINT.parent, 96), CHECKPOINT)
        self.assertEqual({path.name for path in CHECKPOINT.iterdir()}, {
            "adapter_config.json", "adapter_model.safetensors", "added_tokens.json",
            "chat_template.jinja", "merges.txt", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer_config.json",
            "video_preprocessor_config.json", "vocab.json",
        })
        self.assertEqual(
            hashlib.sha256((CHECKPOINT / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "726af53199fd0b95a50d081a888d78a220fb7c851f3cbcd9e9976c4b20970555",
        )
        config = PeftConfig.from_pretrained(str(CHECKPOINT), local_files_only=True)
        self.assertEqual(config.base_model_name_or_path, training.DEFAULT_MODEL)
        self.assertEqual(config.revision, training.DEFAULT_REVISION)
        self.assertEqual((config.r, config.lora_alpha), (8, 16))
        with safe_open(CHECKPOINT / "adapter_model.safetensors", framework="pt") as weights:
            self.assertGreater(len(weights.keys()), 0)
            for key in weights.keys():
                self.assertIn(8, weights.get_slice(key).get_shape())

    def test_optional_state_and_incomplete_or_wrong_adapter_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-96"
            checkpoint.mkdir()
            for name in (
                "adapter_model.safetensors", "preprocessor_config.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja",
            ):
                (checkpoint / name).write_text("fixture")
            config = checkpoint / "adapter_config.json"
            config.write_text('{"peft_type":"LORA"}')
            self.assertEqual(evaluator.checkpoint_path(root, 96), checkpoint)
            state = checkpoint / "trainer_state.json"
            state.write_text('{"global_step":96}')
            self.assertEqual(evaluator.checkpoint_path(root, 96), checkpoint)
            state.write_text('{"global_step":8}')
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluator.checkpoint_path(root, 96)
            state.unlink()
            config.write_text('{"peft_type":"IA3"}')
            with self.assertRaisesRegex(ValueError, "LoRA"):
                evaluator.checkpoint_path(root, 96)
            (checkpoint / "chat_template.jinja").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing or incomplete"):
                evaluator.checkpoint_path(root, 96)

    def test_generation_decodes_only_new_tokens(self):
        class Inputs(dict):
            def to(self, device):
                return self

        model = MagicMock(device="cpu")
        model.generate.return_value = torch.tensor([[10, 11, 12, 99, 100]])
        processor = MagicMock()
        processor.return_value = Inputs(input_ids=torch.tensor([[10, 11, 12]]))
        processor.batch_decode.return_value = ["[1, 2, 3, 4]"]
        self.assertEqual(evaluator.generate_bbox(model, processor, [], [], 50), "[1, 2, 3, 4]")
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(processor.batch_decode.call_args.args[0].tolist(), [[99, 100]])

    def test_safe_bbox_parser_and_iou(self):
        for value in ("[0, 0, 100, 100]", "<answer>[0, 0, 100, 100]</answer>", "[[0, 0], [100, 100]]"):
            self.assertEqual(evaluator.parse_bbox(value), [0, 0, 100, 100])
        self.assertAlmostEqual(evaluator.box_iou([0, 0, 100, 100], [50, 0, 150, 100]), 1 / 3)
        for value in (
            "not json", "[true, 0, 1, 2]", "[0, 0, NaN, 1]", "[0, 0, Infinity, 1]",
            "[0, 0, 0, 1]", "[10, 0, 1, 1]", "[0, 1]", "[0, 0, 1e308, 1e308]",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.parse_bbox(value)

    def test_attention_losses_are_finite_and_differentiable_for_all_shots(self):
        for shots in (1, 2, 4):
            tokens = [30768, 87, 30961]
            for _ in range(shots):
                tokens += [151652, 151655, 151655, 151653, 508, 16, 11, 17, 11, 18, 11, 19, 60]
            tokens += [151652, 151655, 151655, 151653, 30768, 20, 11, 21, 11, 22, 11, 23, 30961]
            ids = torch.tensor([tokens])
            for weights in ((0, 0, 0), (1, 0, 0), (0, 0.25, 0), (0, 0, 1), (1, 1, 1)):
                trainer = object.__new__(training.CustomSFTTrainer)
                trainer.alpha, trainer.beta, trainer.gamma = weights
                trainer.margin, trainer.needs_attention = 0.025, any(weights)
                trainer.token_ids = {
                    "vision_start": 151652, "vision_end": 151653, "support_start": 508,
                    "support_end": 60, "query_start": 30768, "query_end": 30961, "comma": 11,
                }
                parameter = torch.randn(1, 2, len(tokens), len(tokens), requires_grad=True)
                base_loss = parameter.square().mean()
                outputs = SimpleNamespace(loss=base_loss, attentions=(parameter.softmax(dim=-1),))
                model = Mock(return_value=outputs)
                loss = trainer.compute_loss(model, {"input_ids": ids})
                self.assertEqual(model.call_args.kwargs["output_attentions"], bool(any(weights)))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
