# Checkpoint Validation

This document describes the reusable Student-only checkpoint validation
workflow under `examples/evaluation`. It is independent from every training
launcher: it does not create an Actor update, load a Teacher, read privileged
reasoning, or run a training reward loop.

The workflow is:

```text
checkpoint weights
  -> merge FSDP/Megatron shards when required
  -> generate K full rollouts for every benchmark question
  -> preserve raw output and canonical gold in JSONL
  -> extract answers and judge equivalence with Ye Wenxuan's parser
  -> write per-benchmark metrics and one summary per checkpoint
```

## Required inputs

Every invocation must identify three things explicitly:

| Input | Where it is supplied | Meaning |
|---|---|---|
| Weight root | `validation-config` → `checkpoint.root` | Directory containing `global_step_<N>` checkpoint folders |
| Checkpoint steps | `--steps` or `checkpoint.steps` | Exact saved steps to evaluate |
| Benchmarks | `--benchmarks` | Exact enabled IDs from `benchmark_catalog.json` |

The command also receives an environment config and a validation config:

```text
--environment-config  cluster paths, Python environments, and hardware
--validation-config   model/checkpoint identity and validation protocol
```

### Weight root

For a Verl FSDP training run:

```text
$FENG_J/checkpoints/<training_run_id>/
├── global_step_20/
│   └── actor/
├── global_step_40/
│   └── actor/
└── global_step_60/
    └── actor/
```

Set:

```json
"checkpoint": {
  "root": "checkpoints/<training_run_id>",
  "backend": "fsdp",
  "actor_subdir": "actor",
  "steps": []
}
```

A relative `checkpoint.root` is resolved against the environment config's
absolute `workspace_root`. The supported backends are `fsdp`, `megatron`, and
`hf`. FSDP/Megatron shards are merged once into `actor_huggingface` beside the
source `actor` directory. The checkpoint's saved shard world size is
independent from the six GPUs used for later generation.

### Checkpoint steps

One checkpoint:

```text
--steps 40
```

Several checkpoints:

```text
--steps 20,40,60,80,100
```

Alternatively, store defaults in the validation JSON:

```json
"steps": [20, 40, 60, 80, 100]
```

When `--steps` is present, it completely replaces `checkpoint.steps`. Steps
are deduplicated, sorted, and executed sequentially.

### Benchmarks

Select only enabled IDs registered in `benchmark_catalog.json`:

```text
--benchmarks aime24,aime25,math500,gsm8k
```

The catalog supplies each dataset's parquet path, expected prompt count,
canonical ground-truth field, maximum prompt length, and Ye `data_name`.
No dataset is inferred from the checkpoint or training run name.

## Configuration separation

The environment and validation protocol are intentionally separate:

```text
examples/evaluation/configs/
├── environment/
│   └── feng_j_1node_6x80gb.json
└── validation/
    ├── qwen3_4b_openthought_official_solution.json
    └── qwen3_4b_openthought_deepseek_r1.json
```

The environment config owns:

```text
workspace_root
Verl Python
Ye-rescore Python
Ye parser path
result/work roots
node count, GPU count, and physical VRAM
```

The validation config owns:

```text
training_run_id and checkpoint root/backend
thinking mode
sampling mode and K
response length
temperature/top-p/top-k
vLLM TP, loader batch size, concurrency, and memory utilization
Ye-rescoring enablement
```

Environment fields are rejected if placed in the validation JSON.

## Thinking and sampling controls

Qwen's chat template may default to thinking mode. This validation workflow
overrides it explicitly through:

```json
"generation": {
  "enable_thinking": false,
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0.0,
  "n_samples": 16
}
```

The current two Qwen3-4B validation configs are therefore:

```text
mode     = no-thinking
sampling = enabled
K        = 16 rollouts per question
```

`enable_thinking=false` is passed to:

```text
data.apply_chat_template_kwargs.enable_thinking=False
```

The standalone generation server sends an OpenAI-compatible request, which
does not contain a `do_sample` field. Actual decoding is sampled when
`temperature > 0` and greedy when `temperature = 0`. The matrix therefore
treats `do_sample` as a validated protocol switch:

```text
do_sample=true  -> require temperature > 0; preserve temperature/top-p/top-k/min-p
do_sample=false -> require n_samples=1; force temperature=0
```

Both mode fields must be JSON booleans (`true`/`false`), not strings. The
preflight rejects inconsistent combinations. The checked-in
`do_sample=true`, `temperature=0.7` configuration is sampled decoding.

With `n_samples=16`, the reported empirical metric is Pass@16. For Pass@4,
set `n_samples=4` and resize `generation_batch_size` for the desired queued
request count.

The matrix prints the resolved mode before accessing GPUs:

```text
model mode     : no-thinking
sampling       : enabled, n=16, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0
```

## Current Qwen3-4B validation preflight

The checked-in Feng_J environment and Qwen3-4B validation configs resolve to:

| Parameter | Value |
|---|---:|
| Nodes | 1 |
| GPUs per node | 6 |
| Tensor parallel size | 1 |
| vLLM replicas | 6 |
| Maximum prompt length | 2,048 |
| Maximum/expected response length | 4,096 / 4,096 |
| Loader batch | 9 prompts |
| Samples per prompt | 16 |
| Generated requests per loader batch | 144 |
| `max_num_seqs` | 24 per replica |
| Configured cluster active cap | 144 sequences |
| `max_num_batched_tokens` | 8,192 per replica |
| GPU memory utilization | 0.70 |
| Physical VRAM | 80 GiB/GPU |
| vLLM memory budget | 56 GiB/GPU |

For Qwen3-4B BF16 (36 layers, 8 KV heads, head dimension 128):

```text
KV bytes/token
= 2 * 36 * 8 * 128 * 2
= 147,456 bytes
= 144 KiB/token

maximum sequence
= 2,048 + 4,096
= 6,144 tokens

KV per maximum sequence
= 6,144 * 144 KiB
= 864 MiB
= 0.84375 GiB

24 maximum sequences
= 20.25 GiB/GPU
```

A coarse capacity estimate subtracts approximately 7.45 GiB of BF16 model
weights and the configured 12 GiB runtime reserve from the 56 GiB vLLM
budget:

```text
estimated usable KV = 56 - 7.45 - 12 = 36.55 GiB
estimated capacity  = floor(36.55 / 0.84375) = 43 sequences/GPU
configured limit    = 24 sequences/GPU
```

CUDA graphs, activations, NCCL, colocated runtime state, fragmentation, and
other overhead are additional. The vLLM startup KV-capacity report and
observed peak memory are authoritative.

`generation_batch_size=9` counts prompts. With `n_samples=16`, one loader
batch queues `9 * 16 = 144` generated requests. Requests may be queued; they
are not guaranteed to be resident simultaneously. Residency is bounded by
the six replicas, `max_num_seqs=24`, and actual KV capacity.

## One-time Ye-rescore setup

The original Ye parser uses `latex2sympy2`, so it runs in a separate Python
3.10 environment:

```bash
export FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J

"$FENG_J/conda/miniconda3/bin/conda" create \
  -p "$FENG_J/conda/envs/ye_rescore" \
  python=3.10 \
  -y

"$FENG_J/conda/envs/ye_rescore/bin/python" -m pip install \
  -r examples/evaluation/requirements_ye_rescore.txt
```

The environment config currently resolves:

```text
$FENG_J/conda/envs/verl_v080_official_script/bin/python
$FENG_J/conda/envs/ye_rescore/bin/python
$FENG_J/Lulu_OPSD-main/parser.py
```

The matrix imports the Ye parser before checkpoint generation and fails
before allocating GPUs when this setup is incomplete.

## Dry-run

Always inspect the resolved matrix first:

```bash
export FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J
cd "$FENG_J/verl-v0.8.0-opsd-test"

"$FENG_J/conda/envs/verl_v080_official_script/bin/python" \
  examples/evaluation/run_generation_matrix.py \
  --environment-config examples/evaluation/configs/environment/feng_j_1node_6x80gb.json \
  --validation-config examples/evaluation/configs/validation/qwen3_4b_openthought_official_solution.json \
  --steps 20,40,60,80,100 \
  --benchmarks aime24,aime25,math500,gsm8k \
  --dry-run
```

Verify the printed checkpoint paths, steps, benchmarks, `data_name`, model
mode, sampling parameters, and output root.

## Formal validation

Run the same command without `--dry-run`:

```bash
"$FENG_J/conda/envs/verl_v080_official_script/bin/python" \
  examples/evaluation/run_generation_matrix.py \
  --environment-config examples/evaluation/configs/environment/feng_j_1node_6x80gb.json \
  --validation-config examples/evaluation/configs/validation/qwen3_4b_openthought_official_solution.json \
  --steps 20,40,60,80,100 \
  --benchmarks aime24,aime25,math500,gsm8k
```

The execution order is checkpoint-first:

```text
step20  -> all selected benchmarks -> step20 summary
step40  -> all selected benchmarks -> step40 summary
...
step100 -> all selected benchmarks -> step100 summary
```

Use `--continue-on-error` to retain failed tasks in an incomplete checkpoint
summary while continuing later tasks. Use `--force` only when complete raw
rollouts must be regenerated. Without `--force`, structurally complete raw
JSONL and matching Ye sidecars are reused.

## Outputs

Final results are independent from training logs and checkpoints:

```text
$FENG_J/checkpoint_validation_results/<training_run_id>/
├── manifest.json
├── step_20/
│   ├── ye_benchmark_summary.json
│   ├── aime24.jsonl
│   ├── aime24.ye_rescored.jsonl
│   ├── aime24.ye_metrics.json
│   ├── aime25.jsonl
│   ├── math500.jsonl
│   └── gsm8k.jsonl
└── step_40/
    └── ...
```

For each benchmark:

```text
<benchmark>.jsonl
    Full raw rollout, prompt, canonical gold, sample_uid, and rollout_id.

<benchmark>.ye_rescored.jsonl
    Ye-extracted answer, correctness, and any judge error for every rollout.

<benchmark>.ye_metrics.json
    Pass@1 estimator, correct rollout count, first-draw accuracy,
    empirical Pass@K, and questions with at least one correct rollout.
```

For each checkpoint:

```text
ye_benchmark_summary.json
    One table containing every selected benchmark's metrics.
```

At the matrix root:

```text
manifest.json
    Resolved environment/validation identity, task state, hashes, paths,
    failures, and checkpoint-summary locations.
```

Empty, failed, truncated, or unextractable generations stay in the
denominator as incorrect. Stored legacy `score`, `reward`, or `correct`
fields are never presented as Ye-rescored judgments.
