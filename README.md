# LLM辅助智能合约意图检测系统

本项目是一个使用大语言模型(LLM)分析智能合约代码并提取合约意图的系统。它包含两个主要组件：智能合约分析器和语义熵分析器。

## 项目架构

```
.
├── sc_analyzer/                  # 智能合约分析核心模块
│   ├── __init__.py               # 初始化文件
│   ├── main.py                   # 主程序入口
│   ├── models.py                 # LLM模型接口
│   ├── utils.py                  # 工具函数
│   └── data_types.py             # 数据结构定义
│
├── semantic_entropy_analyzer/    # 语义熵分析器
│   ├── __init__.py               # 初始化文件
│   ├── semantic_entropy.py       # 语义熵计算核心
│   ├── runner.py                 # 运行器
│   └── results_analyzer.py       # 结果分析器
│
├── contracts_to_analyze/         # 待分析的智能合约
│   ├── ERC20-USDC/               # USDC合约示例
│   └── ERC20-USDT/               # USDT合约示例
│
├── intent_results/               # 分析结果输出目录
│
├── setup.py                      # 项目安装配置（已弃用）
├── requirements.txt              # 项目依赖
└── view_results.py               # 查看分析结果的脚本（已弃用）
```

## 核心组件说明

### 1. 智能合约分析器 (sc_analyzer)

智能合约分析器是系统的核心组件，负责调用大语言模型分析智能合约代码，提取合约的意图。

- **main.py**: 主程序入口，处理命令行参数，协调分析流程
- **models.py**: 封装了与各种LLM接口的交互，支持OpenAI、Anthropic等API以及本地大模型
- **utils.py**: 提供日志、文件处理、结果保存等工具函数
- **data_types.py**: 定义了系统中使用的所有数据结构，为序列化到pickle文件的数据提供类型提示和结构化访问

### 2. 语义熵分析器 (semantic_entropy_analyzer)

语义熵分析器用于评估智能合约意图提取的一致性和稳定性。

- **semantic_entropy.py**: 实现了语义熵的计算方法，用于量化不同运行结果之间的差异
- **runner.py**: 语义熵分析的运行器，管理分析流程
- **results_analyzer.py**: 分析和可视化语义熵结果

### 3. 项目工作流程

1. 从`contracts_to_analyze`目录读取智能合约代码
2. 使用`sc_analyzer`调用LLM分析合约代码，提取意图
3. 将分析结果保存到`intent_results`目录
4. 使用`semantic_entropy_analyzer`计算意图提取的语义熵


## 环境要求

详细的依赖项请参阅`requirements.txt`文件，主要包括：

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- OpenAI/Anthropic API客户端
- 数据分析和可视化库

## 使用方法

请参考项目根目录下的操作说明文档：
- `答案生成操作说明.md`：如何生成智能合约分析结果
- `语义熵生成操作说明.md`：如何计算和分析语义熵

## 运行方式

### 使用命令行参数

```bash
python -m sc_analyzer.main --input_dir /path/to/contracts --model_name gpt-4 --debug
```

### 使用配置文件

可以使用JSON或YAML格式的配置文件指定所有参数：

```bash
# 使用JSON配置文件
python -m sc_analyzer.main --config config_example.json

# 使用YAML配置文件
python -m sc_analyzer.main --config config_example.yaml
```

命令行参数优先级高于配置文件参数。例如，即使配置文件中指定了`model_name`，也可以在命令行中覆盖它：

```bash
python -m sc_analyzer.main --config config_example.json --model_name gpt-3.5-turbo
```

## 配置系统

本项目使用模块化的配置系统，配置处理逻辑位于`sc_analyzer/config.py`文件中。配置系统提供以下功能：

1. 支持命令行参数和配置文件（JSON/YAML）
2. 命令行参数优先级高于配置文件
3. 自动参数验证
4. 类型安全的配置访问

## 配置示例

### JSON配置文件示例（config_example.json）

```json
{
    "input_dir": "/path/to/contracts",
    "model_type": "api",
    "model_name": "gpt-4",
    "max_contracts": 20,
    "save_interval": 5,
    "num_tests": 3,
    "step": "step1",
    "log_level": "INFO",
    "http_proxy": "http://127.0.0.1:7890"
}
```

### YAML配置文件示例（config_example.yaml）

```yaml
# 智能合约分析器配置示例

# 输入/输出选项
input_dir: /path/to/contracts
max_contracts: 20
save_interval: 5
num_tests: 3

# 步骤执行选项
step: step1

# 模型选项
model_type: api
model_name: gpt-4
embedding_model: text-embedding-ada-002

# 代理设置
http_proxy: http://127.0.0.1:7890

# 日志选项
log_level: INFO
```

## 参数说明

所有命令行可用的参数都可以在配置文件中指定：

- `input_dir`: 包含智能合约文件夹的目录
- `model_type`: 使用的模型类型（"api"或"local"）
- `model_name`: 模型名称，如"gpt-4"、"gpt-3.5-turbo"等
- `max_contracts`: 要处理的最大合约数量
- `save_interval`: 每处理这么多合约后保存结果
- `num_tests`: 每个合约运行的测试次数
- `step`: 执行步骤（"step1"、"step2"、"step3"或"all"）
- `log_level`: 日志级别（"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"）

使用YAML配置文件需要安装PyYAML：
```bash
pip install pyyaml
```

## 数据结构定义

系统使用了`TypedDict`类型系统来定义所有序列化到pickle文件中的数据结构。主要数据结构包括：

### 分析结果数据结构

- **AnalysisResults**: 完整的分析结果
- **ContractData**: 单个智能合约的分析数据
- **IntentData**: 单次意图分析的结果
- **Section**: 意图的一个部分（如合约交互、状态变化等）
- **Question**: 针对意图特定部分的问题
- **AnswerWithArgLogprob**: 问题的一次回答

### 语义熵数据结构

- **EntropyResults**: 语义熵计算的完整结果
- **ContractEntropy**: 合约的语义熵数据
- **IntentEntropy**: 意图的语义熵数据
- **QuestionEntropy**: 问题的语义熵数据
- **EntropySummary**: 语义熵计算的摘要信息

这些数据结构定义清晰地映射了系统中的各个组件和处理流程，使代码更易于理解和维护。所有与pickle文件相关的函数都已使用这些类型进行了类型标注。