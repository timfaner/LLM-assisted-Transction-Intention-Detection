# Answer Generation Guide

## Environment

Set an API key through the environment or through command-line arguments:

```bash
export OPENAI_API_KEY="your-api-key"
```

Optional proxy settings:

```bash
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
```

## Run Analysis

Basic command:

```bash
python -m sc_analyzer.main --input_dir contracts_to_analyze --model_type api --model_name gpt-4
```

Step-by-step execution:

```bash
python -m sc_analyzer.main --step step1 --input_dir contracts_to_analyze
python -m sc_analyzer.main --step step2 --input_results intent_results/results.pkl
python -m sc_analyzer.main --step step3 --input_results intent_results/results.pkl
```

## Inspect Results

```bash
python view_results.py
python view_results.py --file intent_results/results.pkl --full-text
python view_results.py --file intent_results/results.pkl --output report.txt
```
