# Reusable checkpoint benchmark runner

For the standalone checkpoint-generation matrix, required inputs,
thinking/sampling controls, preflight, commands, and outputs, read
[`VALIDATION.md`](VALIDATION.md).

`run_checkpoint_benchmark.sh` evaluates one Hugging Face model plus any number
of verl FSDP/Megatron checkpoints with one validation protocol. Raw generations
remain under a timestamped evaluation directory. A single registry is updated
for every benchmark of the same algorithm/model pair:

```text
$RESULTS_ROOT/<algorithm_id>/<model_id>/benchmark_registry.json
```

## Model manifest

The runner accepts a tab-separated manifest with no header. The first model is
the comparison baseline.

```text
# label<TAB>backend<TAB>path
base_initial_posttrained<TAB>hf<TAB>Qwen/Qwen3-1.7B
step25<TAB>fsdp<TAB>/path/to/global_step_25/actor
```

Supported backends are `hf`, `fsdp`, and `megatron`. Sharded checkpoints are
merged once into `actor_huggingface` beside the original `actor` directory and
reused by later benchmarks.

## Result layout

```text
benchmark_results/<algorithm>/<model>/
├── benchmark_registry.json
└── runs/<training_run>/<benchmark>/<eval_id>/
    ├── config_snapshot.json
    ├── model_manifest.tsv
    ├── resolved_models.tsv
    ├── summary.json
    └── models/<checkpoint>/
        ├── eval.log
        ├── exit_status.txt
        └── validation/0.jsonl
```

The registry stores normalized accuracy, exact pass-at-N, majority-vote
accuracy, invalid-answer rate, approximate response-limit rate, and paired
bootstrap comparisons against the first model in the manifest.

Run `run_opsd_answer_qwen3_1_7b_aime24.sh` for the current OPSD-answer AIME24
experiment. For another algorithm or training run, keep the generic runner
unchanged and add a new manifest plus a small launcher that exports the IDs,
benchmark file, and protocol.

## Checkpoint generation matrix with Ye rescoring

`run_generation_matrix.py` runs the existing scorer-free
`run_checkpoint_generation.sh` for every selected checkpoint and benchmark.
Generation still does not call a Verl reward function. After each raw JSONL
passes its completeness checks, the matrix invokes
`rescore_generation_jsonl.py`, which loads Ye Wenxuan's
`Lulu_OPSD-main/parser.py` and applies:

```python
gold = strip_string(str(row["gts"]))
pred = extract_answer(str(row["output"]), data_name=ye_data_name)
correct = bool(math_equal(pred, gold))
```

The raw JSONL remains unchanged. Empty or unextractable outputs stay in the
denominator as incorrect.

Benchmark parquet files are registered in:

```text
examples/evaluation/benchmark_catalog.json
```

Each entry supplies:

```text
enabled
data_path
expected_prompts
prompt_key
responses_key
ground_truth_field
ye_data_name
max_prompt_length
```

Relative `data_path` values are resolved against the evaluation environment's
`workspace_root`, so the benchmark catalog is not tied to one absolute cluster
path.

`ye_data_name` is mandatory when rescoring is enabled. It prevents the matrix
from guessing which extraction branch to use. The common mappings are:

| Benchmark | `ye_data_name` |
|---|---|
| MATH, MATH-500 | `math` |
| GSM8K | `gsm8k` |
| AIME 2024/2025 | `aime24`, `aime25` |
| AMC23 | `amc23` |
| OlympiadBench | `olympiadbench` |
| MMLU STEM / MMLU-Pro | `mmlu_stem`, `mmlu_pro` |
| ARC-Challenge / TruthfulQA | `arc_challenge`, `truthfulqa` |
| SAT Math / AQuA | `sat_math`, `aqua` |
| SVAMP / ASDiv / MAWPS / TabMWP | `svamp`, `asdiv`, `mawps`, `tabmwp` |

The JSONL `gts` value must already be the canonical final gold answer. In
particular, GSM8K's official `####` reference parsing is not applied to a
canonical `gts`, and the rollout itself never needs to contain `####`.

Ye's original parser imports the legacy `latex2sympy2`, while the Verl
environment uses `latex2sympy2_extended`. Keep the exact Ye dependency in a
small separate Python 3.10 environment instead of changing the training
environment:

```bash
"$FENG_J/conda/miniconda3/bin/conda" create \
  -p "$FENG_J/conda/envs/ye_rescore" python=3.10 -y

"$FENG_J/conda/envs/ye_rescore/bin/python" -m pip install \
  -r examples/evaluation/requirements_ye_rescore.txt
```

The evaluation environment config uses:

```text
$FENG_J/conda/envs/ye_rescore/bin/python
$FENG_J/Lulu_OPSD-main/parser.py
```

Before any checkpoint generation starts, the matrix imports this parser in the
rescore environment and fails immediately if the path or a dependency is
missing.

To add another parquet benchmark, add a new identifier with the catalog
fields above and use the exact Ye `data_name`. Disabled entries cannot be
selected until `enabled=true` and `data_path` is set.

## Separate environment and validation configurations

Evaluation is configured independently from every training launcher:

```text
examples/evaluation/configs/
├── environment/
│   └── feng_j_1node_6x80gb.json
└── validation/
    ├── qwen3_4b_openthought_official_solution.json
    └── qwen3_4b_openthought_deepseek_r1.json
```

The environment config owns only:

```text
workspace_root
Verl and Ye Python paths
Ye parser path
result and work roots
node/GPU count and physical VRAM
```

The validation config owns only:

```text
training-run and checkpoint identity
checkpoint backend and selected default steps
thinking and sampling protocol
prompt-independent response length and K
vLLM batching, parallelism, concurrency, and memory utilization
whether Ye rescoring is enabled
```

Benchmark paths, expected question counts, prompt limits, and Ye `data_name`
remain in the third independent file, `benchmark_catalog.json`.

The checkpoint step list is deliberately empty. Supply the exact saved steps
at invocation time so an incomplete or unrelated checkpoint is not selected
silently.

Run only MATH-500 for five checkpoints:

```bash
export FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J
cd "$FENG_J/verl-v0.8.0-opsd-test"

"$FENG_J/conda/envs/verl_v080_official_script/bin/python" \
  examples/evaluation/run_generation_matrix.py \
  --environment-config examples/evaluation/configs/environment/feng_j_1node_6x80gb.json \
  --validation-config examples/evaluation/configs/validation/qwen3_4b_openthought_official_solution.json \
  --steps 20,40,60,80,100 \
  --benchmarks math500
```

Run the four currently registered benchmarks:

```bash
"$FENG_J/conda/envs/verl_v080_official_script/bin/python" \
  examples/evaluation/run_generation_matrix.py \
  --environment-config examples/evaluation/configs/environment/feng_j_1node_6x80gb.json \
  --validation-config examples/evaluation/configs/validation/qwen3_4b_openthought_official_solution.json \
  --steps 20,40,60,80,100 \
  --benchmarks aime24,aime25,math500,gsm8k
```

For the DeepSeek-R1 training run, change only `--validation-config` to:

```text
examples/evaluation/configs/validation/qwen3_4b_openthought_deepseek_r1.json
```

The final result layout is checkpoint-first:

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

Runtime parquet files and logs are kept separately under:

```text
$FENG_J/evaluation_work/checkpoint_generation_matrix
```

Each final rollout row includes:

```text
training_run_id
checkpoint_step
checkpoint_id
benchmark_id
sample_uid
rollout_id
input
prompt
output
gts
extra_info (when present)
```

The matrix is sequential and resumable. A complete existing output is
validated without regeneration. Its Ye sidecars are reused only when the raw
JSONL hash, parser hash, `data_name`, prompt count, K, and details-file hash
all still match. Each task's Ye metrics and sidecar paths are also recorded in
`manifest.json`. Use `--force` to regenerate the raw output, `--dry-run` to
inspect the resolved matrix without accessing cluster files, and
`--continue-on-error` to continue after a failed task.

After the last selected benchmark for each checkpoint, the matrix prints one
checkpoint-level table and writes `step_<N>/ye_benchmark_summary.json`. It
contains every selected benchmark's rollout accuracy / Pass@1 estimator,
correct-rollout count, first-draw accuracy, empirical Pass@K, and
questions-with-any-correct count. With `--continue-on-error`, failed or
missing benchmarks remain visible and the checkpoint summary is marked
`incomplete`. Its path and status are also stored under
`checkpoint_summaries` in `manifest.json`.

Each per-benchmark `*.ye_metrics.json` stores the same benchmark metrics with
parser and input provenance. The detailed `*.ye_rescored.jsonl` sidecar
contains only the rollout keys, `ye_pred_answer`, `ye_correct`, and any judge
error; it does not duplicate the full rollout text.

The scorer can also be run independently on an existing complete JSONL:

```bash
"$FENG_J/conda/envs/ye_rescore/bin/python" \
  examples/evaluation/rescore_generation_jsonl.py \
  --input-jsonl /path/to/math500.jsonl \
  --benchmark-id math500 \
  --data-name math \
  --expected-prompts 500 \
  --samples-per-prompt 16 \
  --ye-parser-path "$FENG_J/Lulu_OPSD-main/parser.py"
```
