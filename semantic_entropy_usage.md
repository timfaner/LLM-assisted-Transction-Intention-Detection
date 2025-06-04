# texttexttexttexttexttexttexttexttexttexttexttexttexttexttext

## 1. texttexttexttext

### 1.1 texttextOpenAI APItexttext texttexttexttexttexttexttexttext 

**texttexttext texttexttexttexttexttext texttext **

```bash
export OPENAI_API_KEY="texttextAPItexttext"
```

**texttexttext texttexttextAPItexttexttexttexttexttexttexttext**

```bash
# texttexttexttexttexttexttexttext--no_apitexttext
--no_api
```

### 1.2 texttexttexttexttexttext

texttexttexttexttexttexttextOpenAI APItexttexttexttexttexttexttexttext 

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
```

## 2. texttexttexttexttexttexttext

### 2.1 texttexttexttexttexttext

```bash
python -m semantic_entropy_analyzer.runner --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl
```

### 2.2 texttexttexttexttexttext texttexttexttexttext 

```bash
python -m semantic_entropy_analyzer.runner \
  --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl \
  --output_dir entropy_results \
  --device cuda \
  --log_level DEBUG \
  --debug \
  --save_detailed
```

### 2.3 texttexttexttext

- `--results_path` texttexttexttexttexttexttexttexttexttext texttext 
- `--output_dir` texttexttexttexttexttexttexttexttexttext
- `--device` texttexttexttext cudatextcpu 
- `--no_api` texttexttextAPItexttexttexttexttexttexttexttext texttexttexttextAPI 
- `--skip_analysis` texttexttexttexttexttext
- `--save_detailed` texttexttexttexttexttexttexttexttexttext
- `--debug` texttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext
- `--log_level` texttexttexttext DEBUG, INFO, WARNING, ERROR, CRITICAL 

## 3. texttexttexttexttexttext

texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttextresults_pathtexttexttexttexttexttextentropy_resultstexttexttext  

### 3.1 texttexttexttexttexttext

- `entropy_results.pkl` texttexttexttexttexttexttexttext Python pickletexttext 
- `entropy_summary.json` texttexttexttexttexttexttext JSONtexttext 
- `analysis/` texttexttexttexttexttext
  - `analysis_results.json` texttexttexttexttexttext
  - `plots/` texttexttexttexttexttexttext
    - `contract_entropy_distribution.png` texttexttexttexttexttexttexttext
    - `section_entropy_comparison.png` texttexttexttexttexttexttexttexttext

### 3.2 texttexttexttexttexttext

texttexttexttexttexttexttexttexttexttexttexttexttexttexttext 
```bash
# texttexttexttexttexttexttexttext
open entropy_results/analysis/plots/contract_entropy_distribution.png

# texttexttexttexttexttexttexttexttext
open entropy_results/analysis/plots/section_entropy_comparison.png
```

## 4. texttexttexttexttexttexttexttext

texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext 

```bash
# 1. texttextAPItexttexttexttexttext
export OPENAI_API_KEY="texttextAPItexttext"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 2. texttexttexttexttexttexttexttexttexttext texttexttexttexttexttext 
python -m sc_analyzer.main --input_dir contracts_to_analyze --model_name gpt-3.5-turbo --num_tests 3

# 3. texttexttexttexttexttexttext
python -m semantic_entropy_analyzer.runner --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl --debug

# 4. texttexttexttexttexttexttexttexttexttexttexttext
open entropy_results/analysis/plots/
```

## 5. texttexttexttexttexttext

### 5.1 APItexttexttexttext

- texttextAPItexttexttexttexttexttext
- texttexttexttexttexttexttexttexttexttext
- texttexttexttexttext`--no_api`texttexttexttextAPItexttexttexttexttexttext

### 5.2 texttexttexttexttexttexttexttexttexttext

- texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext
- texttexttexttexttexttexttexttexttexttexttexttexttexttext
- texttext`--debug`texttexttexttexttexttexttexttexttexttext

### 5.3 texttexttexttexttexttext

- texttextmatplotlibtexttexttexttext
- texttexttexttexttexttexttexttexttexttext texttexttext texttexttexttexttexttexttexttexttexttext

## 6. texttexttexttexttexttexttext

### 6.1 texttexttexttexttexttext

texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext 
- texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext
- texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext
- texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext

### 6.2 texttexttexttext

texttexttexttexttexttexttexttexttexttexttext 

1. **texttexttexttext** texttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttext
2. **texttexttexttext** texttexttexttexttexttexttexttexttexttext
3. **texttexttexttext** texttexttexttexttexttexttexttexttext
4. **texttexttexttext** texttexttexttext texttexttexttext texttexttexttext texttext texttext texttexttext
5. **texttexttexttext** texttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttext 

### 6.3 texttexttexttext

- **texttexttexttexttexttexttexttext** texttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext
- **texttexttexttexttexttexttexttexttext** texttexttexttexttexttext texttexttexttext texttexttexttext texttext texttext texttexttexttexttext

### 6.4 texttexttexttext

- **0.0 - 0.5** texttexttexttext texttexttexttexttexttext
- **0.5 - 1.5** texttexttexttexttext texttexttexttexttexttext
- **1.5texttext** texttexttexttext texttexttexttexttexttexttext

texttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttext  