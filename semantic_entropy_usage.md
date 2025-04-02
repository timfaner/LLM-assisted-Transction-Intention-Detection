# 智能合约语义熵分析系统操作说明

## 1. 环境准备

### 1.1 设置OpenAI API密钥（用于语义等价判断）

**方式一：使用环境变量（推荐）**

```bash
export OPENAI_API_KEY="您的API密钥"
```

**方式二：不使用API进行语义等价判断**

```bash
# 在运行命令时添加--no_api参数
--no_api
```

### 1.2 配置代理设置

在中国大陆访问OpenAI API通常需要使用代理。

```bash
export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
```

## 2. 运行语义熵分析

### 2.1 基本分析命令

```bash
python -m semantic_entropy_analyzer.runner --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl
```

### 2.2 完整分析命令（带所有参数）

```bash
python -m semantic_entropy_analyzer.runner \
  --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl \
  --output_dir entropy_results \
  --device cuda \
  --log_level DEBUG \
  --debug \
  --save_detailed
```

### 2.3 参数说明

- `--results_path`：意图分析结果文件路径（必需）
- `--output_dir`：保存语义熵结果的目录
- `--device`：运行设备（cuda或cpu）
- `--no_api`：不使用API进行语义等价判断（默认使用API）
- `--skip_analysis`：跳过分析阶段
- `--save_detailed`：保存详细的熵分析结果
- `--debug`：开启详细调试日志，显示语义熵计算过程
- `--log_level`：日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

## 3. 查看分析结果

语义熵分析结果会保存在指定的输出目录中（默认为results_path的父目录下的entropy_results文件夹）。

### 3.1 主要结果文件

- `entropy_results.pkl`：完整的语义熵结果（Python pickle格式）
- `entropy_summary.json`：语义熵结果摘要（JSON格式）
- `analysis/`：分析结果目录
  - `analysis_results.json`：详细分析结果
  - `plots/`：可视化图表目录
    - `contract_entropy_distribution.png`：合约语义熵分布图
    - `section_entropy_comparison.png`：各部分语义熵对比图

### 3.2 查看分析图表

直接打开生成的图表文件进行查看：
```bash
# 查看合约熵分布图
open entropy_results/analysis/plots/contract_entropy_distribution.png

# 查看各部分熵对比图
open entropy_results/analysis/plots/section_entropy_comparison.png
```

## 4. 完整操作流程示例

以下是一个完整的语义熵分析操作流程示例：

```bash
# 1. 设置API密钥和代理
export OPENAI_API_KEY="您的API密钥"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 2. 运行智能合约意图分析（如果尚未运行）
python -m sc_analyzer.main --input_dir contracts_to_analyze --model_name gpt-3.5-turbo --num_tests 3

# 3. 运行语义熵分析
python -m semantic_entropy_analyzer.runner --results_path intent_results/run-YYYYMMDD_HHMMSS/files/results.pkl --debug

# 4. 查看生成的图表和分析结果
open entropy_results/analysis/plots/
```

## 5. 常见问题排查

### 5.1 API调用失败

- 检查API密钥是否正确
- 检查代理设置是否正确
- 可尝试使用`--no_api`参数跳过API语义等价判断

### 5.2 计算结果中出现零熵值

- 检查原始意图分析是否生成了足够多样的答案
- 确保每个问题有多个不同的答案
- 增加`--debug`参数查看详细计算过程

### 5.3 图表显示问题

- 确保matplotlib正确安装
- 如果显示中文字体警告，可忽略，图表标签已经使用英文

## 6. 结果结构和解读

### 6.1 语义熵的含义

语义熵是衡量模型生成内容多样性和不确定性的指标：
- 较高的熵值表示模型生成了更多样化的答案
- 较低的熵值表示模型生成的答案相似度高
- 零熵值通常表示所有答案在语义上完全等价

### 6.2 结果结构

语义熵结果采用分层结构：

1. **整体摘要**：包含分析合约数量、整体平均熵和计算时间
2. **合约级别**：每个合约的平均语义熵
3. **意图级别**：每个意图的整体熵值
4. **部分级别**：每个部分（合约交互、状态变化、事件、影响）的熵值
5. **问题级别**：每个问题的熵值（仅在启用详细调试时显示）

### 6.3 图表解读

- **合约语义熵分布图**：显示不同合约熵值的分布情况，包括平均值和中位数
- **各部分语义熵对比图**：对比不同部分（合约交互、状态变化、事件、影响）的平均熵值

### 6.4 熵值分析

- **0.0 - 0.5**：低多样性，答案高度相似
- **0.5 - 1.5**：中等多样性，存在一些变化
- **1.5以上**：高多样性，答案有显著差异

通过分析不同部分的熵值，可以识别模型在哪些方面的分析更为稳定或不确定，从而评估智能合约分析的可靠性。 