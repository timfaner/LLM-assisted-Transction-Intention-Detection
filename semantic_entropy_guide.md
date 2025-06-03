# Semantic Entropy Guide

This tool clusters multiple answers for each question, computes question-level semantic entropy, aggregates section-level entropy, and reports contract-level summary metrics.

## Run

```bash
python -m semantic_entropy_analyzer.runner --results_path intent_results/results.pkl --output_dir entropy_results --mode step3
```

Run without API-based entailment checks:

```bash
python -m semantic_entropy_analyzer.runner --results_path intent_results/results.pkl --output_dir entropy_results --no_api --mode step3
```

## Outputs

- `entropy_results.pkl`: serialized entropy results.
- `entropy_summary.json`: summary metrics.
- `analysis/`: derived analysis and plots.
