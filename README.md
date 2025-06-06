# LLM-Assisted Smart Contract Intention Detection

This project analyzes smart contract code with large language models and measures the stability of extracted intentions with semantic entropy.

## Components

- `sc_analyzer`: smart contract analysis, model access, configuration, and result persistence.
- `semantic_entropy_analyzer`: semantic clustering, entropy calculation, and result analysis.
- `contracts_to_analyze`: sample contracts and transaction data.
- `intent_results`: local analysis output directory.

## Setup

Install dependencies from `requirements.txt`.

Copy `api_keys.example.json` to `api_keys.json` and fill in local API keys. `api_keys.json` is ignored and must not be committed.

## Usage

Run a full smart contract analysis:

```bash
python -m sc_analyzer.main --input_dir /path/to/contracts --model_name gpt-4 --debug
```

Run semantic entropy analysis:

```bash
python -m semantic_entropy_analyzer.runner --results_path intent_results/results.pkl --output_dir entropy_results --mode step3
```

See `answer_generation_guide.md` and `semantic_entropy_guide.md` for command examples.
