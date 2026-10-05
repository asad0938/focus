# FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization

Code accompanying the **ICML 2026** paper.

**Authors:** Mohammed Asad Karim and Vinay Kumar Verma

**Paper:** [ICML 2026 proceedings](https://proceedings.mlr.press/v306/karim26a.html)

## Introduction

FOCUS localizes a target object in a query image using a few support images and
their bounding-box annotations, without requiring category names. It encourages
visual matching to the supported object instance rather than relying on semantic
priors.

Training has two stages. First, **supervised fine-tuning with attention loss**
encourages support-aware visual attention, grounding localization in the support
annotations. We then apply **Group Relative Policy Optimization (GRPO)** on top
of the attention-loss-trained model, using IoU and output-format rewards to
improve bounding-box accuracy and encourage valid predictions.

We share the code for the **attention-loss training stage** using
**Qwen2-VL-7B-Instruct** and LoRA. For the subsequent GRPO stage, we used
[Curr_REFT](https://github.com/ding523/Curr_REFT) and refer readers to that repository.

**Train with four shots, then evaluate the same checkpoint at one, two, and four shots.**

## Results

Paper-reported TAO mIoU from Table 3, on a 0-100 scale:

| Model | 1-shot | 2-shot | 4-shot |
| --- | ---: | ---: | ---: |
| Qwen2-VL-7B vanilla | 26.0 | 31.6 | 36.1 |
| Qwen2-VL-7B + SFT + attention loss | 51.7 | 54.0 | 57.1 |
| Qwen2-VL-7B + SFT + attention loss + GRPO | **55.8** | **63.0** | **68.5** |

The evaluator reports `mean_iou` on a 0-1 scale.

## Run Experiments

We used **four NVIDIA A100 GPUs** for training. Use Linux and Python 3.12, with
one GPU for inference. Run the commands from the directory containing `train.py`.

### 1. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0
DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r requirement.txt
```

The base model is downloaded from Hugging Face automatically when first used.

### 2. Download data

Obtain access to [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal)
and follow its source-data terms. The downloader fetches only the selected
support/query images and creates the training and evaluation manifests.

```bash
hf auth login
python download_data.py --split train --n_shot 4 --output data

for shots in 1 2 4; do
  python download_data.py --split test --n_shot "$shots" --output data
done
```

Images are shared under `data/frames/`; matching existing images are reused.

### 3. Train with four shots

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  train.py --deepspeed zero3.json \
  --train_annotations data/train_4shot.json --train_frames data/frames \
  --n_shot 4 --beta 0.25 --margin 0.025 --target_ratio 0.3 \
  --output_root outputs/train-4shot
```

### 4. Run inference at one, two, and four shots

Choose an existing saved checkpoint step:

```bash
CHECKPOINT_ROOT=outputs/train-4shot
CHECKPOINT_STEP=96
for shots in 1 2 4; do
  CUDA_VISIBLE_DEVICES=0 python test.py "$CHECKPOINT_STEP" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --eval_annotations "data/test_${shots}shot.json" --eval_frames data/frames \
    --n_shot "$shots" --target_ratio 2.5 \
    --output_root "results/eval-${shots}shot"
done
```

To use the bundled checkpoint without training, set `CHECKPOINT_ROOT=checkpoints`
and keep `CHECKPOINT_STEP=96`. Predictions and IoU metrics are saved under
`results/`. Existing training and inference outputs are not overwritten.

## Citation

```bibtex
@InProceedings{pmlr-v306-karim26a,
  title     = {{FOCUS}: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization},
  author    = {Karim, Mohammed Asad and Verma, Vinay Kumar},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {56062--56074},
  year      = {2026},
  volume    = {306},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/karim26a.html}
}
```

## License

This project is licensed under the Apache-2.0 License.
