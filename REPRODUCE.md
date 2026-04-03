# Reproducing Results

This document provides step-by-step instructions to reproduce all experiments from the paper. All commands are standalone — no cluster scheduler (e.g. SLURM) is required.

## Requirements

- **GPU:** All experiments were run on a single NVIDIA A100 80GB, except PV-tuning of the 8B model which used a B200.
- **RAM:** 64GB system RAM (128GB for 8B models)
- **Disk:** >= 250GB free. Model weights, quantized checkpoints, and PV-tuning datasets require significant space, especially when running multiple configurations.
- **Software:** Python >= 3.10, CUDA >= 11.8, PyTorch >= 2.3.0
- **Docker image (recommended):** [`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`](https://hub.docker.com/r/pytorch/pytorch/) — the `-devel` tag is required for CUDA kernel JIT compilation. After launching the container, `pip install -r requirements.txt` will upgrade PyTorch and install all dependencies at the correct versions.

**Single vs multi-GPU:** All commands below assume a single GPU. On a multi-GPU machine, either restrict to one GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py ...
```
or use multiple GPUs by ensuring `--finetune_batch_size` is divisible by the number of GPUs (e.g. `--finetune_batch_size 2` for 2 GPUs).

## Setup

```bash
# Clone and install
git clone <repo-url> && cd <repo-name>
pip install -r requirements.txt

# Install the AQLM inference library (needed for lm-eval with quantized models)
cd inference_lib && pip install -e . && cd ..

# Install lm-eval harness and ninja (for downstream task evaluation and CUDA JIT)
pip install lm-eval==0.4.4 ninja

# Set environment variables — adjust paths to your system
export OUTROOT="./outputs"
export HF_HOME="$OUTROOT/hf_cache"
export HF_DATASETS_CACHE="$OUTROOT/hf_cache"
export HF_TOKEN="your_hf_token_here"         # needed for gated models (Llama)
export AQ_USE_JIT=0                          # disable JIT (more stable)

mkdir -p "$OUTROOT"
```

---

## Experiment Map

Below is a mapping from each paper table/figure to the commands needed to reproduce it.

| Paper Table/Figure | Section | What to run |
|---|---|---|
| Table 1 (pre-PV, 2bpp, Llama 3B) | §1a, §1e, §1f | Quantize with b∈{4,8,16}, e∈{5,100}, baseline + OA-EM |
| Table 2 (post-PV, 2bpp, Llama 3B) | §1a + §2 + §3 | Quantize → PV-tune → evaluate all 8 configs |
| Table 3 (downstream summary) | §3c | lm-eval on all PV-tuned models |
| Table 4 (cross-rate/architecture) | §1a–§1d + §2 + §3 | 3bpp + 8B + Qwen quantize → PV-tune → evaluate |
| Table 5 (domain degradation) | §1a | Compare C4/LAMBADA/WikiText-2 from pre-PV runs |
| Tables 6–7 (3bpp detailed) | §1b + §2 + §3 | 3bpp quantize → PV-tune → lm-eval |
| Tables 8–11 (2bpp downstream per beam) | §1a, §1e, §1f + §2 + §3c | All beam configs → PV-tune → lm-eval |
| Tables 12–13 (8B detailed) | §1c + §2 + §3 | 8B quantize → PV-tune → lm-eval |
| Tables 14–15 (Qwen detailed) | §1d + §2 + §3 | Qwen quantize → PV-tune → lm-eval |
| Table 16 (Pareto analysis) | all of §1 + §2 | Collect Q-time + post-PV perplexity from all runs |

---

## 1. Quantization (AQLM Baseline vs OA-EM)

All quantization commands follow the same template. The key variables are:
- **Model:** `unsloth/Llama-3.2-3B`, `meta-llama/Llama-3.1-8B`, or `Qwen/Qwen2.5-3B`
- **Bitrate:** `--num_codebooks 2` (2bpp) or `--num_codebooks 3` (3bpp)
- **Beam width:** `--beam_size {4, 8, 16}`
- **Epoch budget:** `--max_epochs {5, 100}`
- **OA-EM:** add `--oa_em_rounds 3 --oa_em_steps 100 --oa_em_lr 1e-4` (omit for baseline)

### 1a. Llama 3.2 3B at 2bpp — baseline vs OA-EM (Tables 1, 2)

**Baseline (greedy init), b=8, e=100:**
```bash
python main.py \
    unsloth/Llama-3.2-3B \
    c4 \
    --save "$OUTROOT/llama3b_2bpp_baseline_b8_e100" \
    --num_codebooks 2 \
    --nbits_per_codebook 8 \
    --in_group_size 8 \
    --out_group_size 1 \
    --seed 42 \
    --nsamples 128 \
    --model_seqlen 4096 \
    --true-sequential \
    --beam_size 8 \
    --max_epochs 100 \
    --relative_mse_tolerance 0.01 \
    --steps_per_epoch 100 \
    --lr 1e-4 \
    --finetune_max_epochs 5 \
    --finetune_lr 1e-5 \
    --finetune_batch_size 1 \
    --use_fast_tokenizer \
    2>&1 | tee "$OUTROOT/llama3b_2bpp_baseline_b8_e100/quantize.log"
```

**OA-EM init, b=8, e=100:**
```bash
python main.py \
    unsloth/Llama-3.2-3B \
    c4 \
    --save "$OUTROOT/llama3b_2bpp_oaem_b8_e100" \
    --num_codebooks 2 \
    --nbits_per_codebook 8 \
    --in_group_size 8 \
    --out_group_size 1 \
    --seed 42 \
    --nsamples 128 \
    --model_seqlen 4096 \
    --true-sequential \
    --beam_size 8 \
    --max_epochs 100 \
    --relative_mse_tolerance 0.01 \
    --steps_per_epoch 100 \
    --lr 1e-4 \
    --finetune_max_epochs 5 \
    --finetune_lr 1e-5 \
    --finetune_batch_size 1 \
    --use_fast_tokenizer \
    --oa_em_rounds 3 --oa_em_steps 100 --oa_em_lr 1e-4 \
    2>&1 | tee "$OUTROOT/llama3b_2bpp_oaem_b8_e100/quantize.log"
```

> **Expected runtime:** ~9-10 hours per run on a single A100. WikiText-2 and C4 perplexity are printed at the end.

To reproduce all rows of Tables 1 and 2, run both baseline and OA-EM for each beam/epoch configuration:

| Config | `--beam_size` | `--max_epochs` | Table rows |
|---|---|---|---|
| b=4, e=100 | 4 | 100 | Table 1 row 1, Table 2 "Narrow beam" |
| b=8, e=100 | 8 | 100 | Table 1 row 2, Table 2 "Standard beam" |
| b=16, e=100 | 16 | 100 | Table 1 row 3, Table 2 "Wide beam" |
| b=8, e=5 | 8 | 5 | Table 1 row 4, Table 2 "Early stopping" |

This gives **8 quantization runs** (4 configs × baseline + OA-EM).

### 1b. Llama 3.2 3B at 3bpp (Tables 4, 6, 7)

Same as 1a (b=8, e=100 only) but with `--num_codebooks 3`. Run baseline and OA-EM:

```bash
# Baseline
python main.py unsloth/Llama-3.2-3B c4 \
    --save "$OUTROOT/llama3b_3bpp_baseline_b8" \
    --num_codebooks 3 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer --new_eval

# OA-EM
python main.py unsloth/Llama-3.2-3B c4 \
    --save "$OUTROOT/llama3b_3bpp_oaem_b8" \
    --num_codebooks 3 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer --new_eval \
    --oa_em_rounds 3 --oa_em_steps 100 --oa_em_lr 1e-4
```

> **Expected runtime:** ~12-13 hours per run.

### 1c. Llama 3.1 8B at 2bpp (Tables 4, 12, 13)

Same as 1a (b=8, e=100) but with `meta-llama/Llama-3.1-8B`.

```bash
# Baseline
python main.py meta-llama/Llama-3.1-8B c4 \
    --save "$OUTROOT/llama8b_2bpp_baseline_b8" \
    --num_codebooks 2 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer

# OA-EM
python main.py meta-llama/Llama-3.1-8B c4 \
    --save "$OUTROOT/llama8b_2bpp_oaem_b8" \
    --num_codebooks 2 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer \
    --oa_em_rounds 3 --oa_em_steps 100 --oa_em_lr 1e-4
```

> **Expected runtime:** ~25-30 hours per run.

### 1d. Qwen 2.5 3B at 2bpp (Tables 4, 14, 15)

Same as 1a (b=8, e=100) but with `Qwen/Qwen2.5-3B` and `--trust_remote_code`.

```bash
# Baseline
python main.py Qwen/Qwen2.5-3B c4 \
    --save "$OUTROOT/qwen3b_2bpp_baseline_b8" \
    --num_codebooks 2 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer --trust_remote_code

# OA-EM
python main.py Qwen/Qwen2.5-3B c4 \
    --save "$OUTROOT/qwen3b_2bpp_oaem_b8" \
    --num_codebooks 2 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 \
    --seed 42 --nsamples 128 --model_seqlen 4096 --true-sequential \
    --beam_size 8 --max_epochs 100 --relative_mse_tolerance 0.01 --steps_per_epoch 100 \
    --lr 1e-4 --finetune_max_epochs 5 --finetune_lr 1e-5 --finetune_batch_size 1 \
    --use_fast_tokenizer --trust_remote_code \
    --oa_em_rounds 3 --oa_em_steps 100 --oa_em_lr 1e-4
```

> **Note:** For Qwen, use `--block_type Qwen2DecoderLayer` in PV-tuning (Step 2).

---

## 2. PV-Tuning (Post-Training Fine-Tuning)

PV-tuning is required to reproduce Tables 2–4, 7–16. Every quantized checkpoint from Step 1 should be PV-tuned.

### 2a. Prepare tokenized dataset (once per model family)

```bash
# Llama 3.2 3B
torchrun --nproc-per-node=1 finetune.py \
    --base_model unsloth/Llama-3.2-3B \
    --quantized_model "$OUTROOT/llama3b_2bpp_baseline_b8_e100" \
    --model_seqlen 4096 \
    --block_type LlamaDecoderLayer \
    --load_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name allenai/c4 \
    --dataset_config_name en \
    --data_files "en/c4-train.00000-of-01024.json.gz" \
    --split "train[:10000]" \
    --seed 42 \
    --use_fast_tokenizer \
    --trust_remote_code \
    --save_dataset_and_exit "$OUTROOT/pvtune_llama3b_c4_10k"

# Llama 3.1 8B (same command, different model)
torchrun --nproc-per-node=1 finetune.py \
    --base_model meta-llama/Llama-3.1-8B \
    --quantized_model "$OUTROOT/llama8b_2bpp_baseline_b8" \
    --model_seqlen 4096 \
    --block_type LlamaDecoderLayer \
    --load_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name allenai/c4 \
    --dataset_config_name en \
    --data_files "en/c4-train.00000-of-01024.json.gz" \
    --split "train[:10000]" \
    --seed 42 \
    --use_fast_tokenizer \
    --trust_remote_code \
    --save_dataset_and_exit "$OUTROOT/pvtune_llama8b_c4_10k"

# Qwen 2.5 3B
torchrun --nproc-per-node=1 finetune.py \
    --base_model Qwen/Qwen2.5-3B \
    --quantized_model "$OUTROOT/qwen3b_2bpp_baseline_b8" \
    --model_seqlen 4096 \
    --block_type Qwen2DecoderLayer \
    --load_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name allenai/c4 \
    --dataset_config_name en \
    --data_files "en/c4-train.00000-of-01024.json.gz" \
    --split "train[:10000]" \
    --seed 42 \
    --use_fast_tokenizer \
    --trust_remote_code \
    --save_dataset_and_exit "$OUTROOT/pvtune_qwen3b_c4_10k"
```

### 2b. Run PV-tuning

Run this for **each** quantized checkpoint from Step 1. Replace `$BASE_MODEL`, `$BLOCK_TYPE`, `$QUANT_DIR`, `$DATASET_DIR`, and `$SAVE_DIR` accordingly. **Important:** `$QUANT_DIR` must point to the raw AQLM checkpoint from Step 1 (containing `.pth` files), **not** an HF-converted model.

```bash
torchrun --nproc-per-node=1 finetune.py \
    --base_model "$BASE_MODEL" \
    --quantized_model "$QUANT_DIR" \
    --model_seqlen 4096 \
    --block_type "$BLOCK_TYPE" \
    --load_dtype bfloat16 \
    --amp_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name "$DATASET_DIR" \
    --seed 42 \
    --update_codes \
    --update_codebooks_and_scales \
    --update_non_quantized_parameters \
    --lamb \
    --debias \
    --lr 3e-4 \
    --adam_beta1 0.90 \
    --adam_beta2 0.95 \
    --max_code_change_per_step 1e-2 \
    --code_lr 1e-2 \
    --code_beta1 0.0 \
    --code_beta2 0.95 \
    --beam_size 1 \
    --delta_decay 0 \
    --batch_size 32 \
    --microbatch_size 1 \
    --max_epochs 5 \
    --gradient_checkpointing \
    --print_every_steps 10 \
    --keep_best_model \
    --save "$SAVE_DIR" \
    --save_every_steps 100 \
    --use_fast_tokenizer \
    --trust_remote_code \
    --offload_optimizer \
    2>&1 | tee "$SAVE_DIR/pvtune.log"
```

**Variable reference:**

| Model | `$BASE_MODEL` | `$BLOCK_TYPE` | `$DATASET_DIR` |
|---|---|---|---|
| Llama 3.2 3B | `unsloth/Llama-3.2-3B` | `LlamaDecoderLayer` | `$OUTROOT/pvtune_llama3b_c4_10k` |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | `LlamaDecoderLayer` | `$OUTROOT/pvtune_llama8b_c4_10k` |
| Qwen 2.5 3B | `Qwen/Qwen2.5-3B` | `Qwen2DecoderLayer` | `$OUTROOT/pvtune_qwen3b_c4_10k` |

> **Expected runtime:** ~1 hour 40 minutes for 3B parameter models on A100, and under 1 hour for the 8B parameter model on B200.

---

## 3. Evaluation

### 3a. Convert checkpoint to HuggingFace format

Before running lm-eval, convert the AQLM checkpoint to HF format. Create the output directory before running conversion (`mkdir -p "$HF_OUT_DIR"`).

**For a standard quantized model (no PV-tuning):**
```bash
mkdir -p "$HF_OUT_DIR"
python convert_to_hf.py \
    "$BASE_MODEL" \
    "$QUANT_DIR" \
    "$HF_OUT_DIR" \
    --save_tokenizer
```

**For a PV-tuned model** (two-step conversion). `$QUANT_DIR` is the raw AQLM checkpoint from Step 1, `$SAVE_DIR` is the PV-tuning output from Step 2:
```bash
# Step 1: Convert FSDP checkpoint to legacy AQLM format
python convert_legacy_model_format.py \
    --base_model "$BASE_MODEL" \
    --quantized_model "$QUANT_DIR" \
    --pv_fsdp_dir "$SAVE_DIR/best_model" \
    --save "$SAVE_DIR/converted" \
    --load_dtype bfloat16 \
    --code_dtype uint16

# Step 2: Convert to HF format
mkdir -p "$HF_OUT_DIR"
python convert_to_hf.py \
    "$BASE_MODEL" \
    "$SAVE_DIR/converted" \
    "$HF_OUT_DIR" \
    --save_tokenizer
```

### 3b. Perplexity (WikiText-2 and C4)

Perplexity on WikiText-2 and C4 is evaluated automatically at the end of each quantization run (Step 1). These values correspond to Tables 1, 4, 5, 6, 12, and 14.

To evaluate a PV-tuned model's perplexity, use the converted AQLM checkpoint (Step 3a, Step 1 output):
```bash
python main.py \
    "$BASE_MODEL" \
    c4 \
    --load "$SAVE_DIR/converted" \
    --val_size 128 \
    --use_fast_tokenizer \
    --new_eval
```

### 3c. Downstream tasks (Tables 3, 7, 8–11, 13, 15)

Run lm-eval on each HF-converted model:

```bash
# ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande (zero-shot)
lm_eval --model hf \
    --model_args pretrained=$HF_OUT_DIR,trust_remote_code=True \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa \
    --batch_size 16 \
    --output_path "$OUTROOT/lmeval_results"

# LAMBADA (zero-shot)
lm_eval --model hf \
    --model_args pretrained=$HF_OUT_DIR,trust_remote_code=True \
    --tasks lambada_openai \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_path "$OUTROOT/lmeval_lambada_results"
```

### 3d. FP16 baselines

For the FP16 reference numbers in Tables 1 and 6:

```bash
lm_eval --model hf \
    --model_args pretrained=unsloth/Llama-3.2-3B \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,lambada_openai \
    --batch_size 16 \
    --output_path "$OUTROOT/lmeval_fp16_llama3b"
```

---

## Quick Smoke Test

To verify the pipeline works without running a full experiment (~5-10 minutes):

```bash
python main.py \
    unsloth/Llama-3.2-3B \
    c4 \
    --save "$OUTROOT/smoke_test" \
    --num_codebooks 2 \
    --nbits_per_codebook 8 \
    --in_group_size 8 \
    --out_group_size 1 \
    --seed 42 \
    --nsamples 4 \
    --model_seqlen 4096 \
    --true-sequential \
    --beam_size 1 \
    --max_epochs 2 \
    --steps_per_epoch 10 \
    --lr 1e-4 \
    --finetune_max_epochs 1 \
    --finetune_lr 1e-5 \
    --finetune_batch_size 1 \
    --use_fast_tokenizer \
    --oa_em_rounds 1 --oa_em_steps 10 --oa_em_lr 1e-4
```

This runs a minimal quantization to verify dependencies and pipeline correctness. The resulting model quality will be poor (too few samples/epochs), but the pipeline should complete without errors.

---

## Total Compute Budget

To reproduce all paper results from scratch:

| Experiment set | Runs | Hours per run | Total GPU-hours |
|---|---|---|---|
| Llama 3B, 2bpp (4 configs × 2 inits) | 8 | ~9h | ~72h |
| Llama 3B, 3bpp (1 config × 2 inits) | 2 | ~13h | ~26h |
| Llama 8B, 2bpp (1 config × 2 inits) | 2 | ~28h | ~56h |
| Qwen 3B, 2bpp (1 config × 2 inits) | 2 | ~9h | ~18h |
| PV-tuning (all 14 models) | 14 | ~4h | ~56h |
| lm-eval (all models + FP16) | ~15 | ~0.5h | ~8h |
| **Total** | | | **~236 GPU-hours** |

## Notes

- Perplexity is evaluated on WikiText-2 and C4 validation sets automatically at the end of quantization.
- The `--new_eval` flag uses the updated C4 validation split (`c4_new`) for perplexity evaluation.
- PV-tuning selects the best WikiText-2 checkpoint via `--keep_best_model`.
