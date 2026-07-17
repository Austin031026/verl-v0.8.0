# Reusable checkpoint benchmark runner

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
