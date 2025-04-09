# LLM辅助智能合约意图检测系统

本项目是一个使用大语言模型(LLM)分析智能合约代码并提取合约意图的系统。它包含两个主要组件：智能合约分析器和语义熵分析器。

## 项目架构

```
.
├── sc_analyzer/                  # 智能合约分析核心模块
│   ├── __init__.py               # 初始化文件
│   ├── main.py                   # 主程序入口
│   ├── models.py                 # LLM模型接口
│   └── utils.py                  # 工具函数
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
├── setup.py                      # 项目安装配置
├── requirements.txt              # 项目依赖
└── view_results.py               # 查看分析结果的脚本
```

## 核心组件说明

### 1. 智能合约分析器 (sc_analyzer)

智能合约分析器是系统的核心组件，负责调用大语言模型分析智能合约代码，提取合约的意图。

- **main.py**: 主程序入口，处理命令行参数，协调分析流程
- **models.py**: 封装了与各种LLM接口的交互，支持OpenAI、Anthropic等API以及本地大模型
- **utils.py**: 提供日志、文件处理、结果保存等工具函数

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
 
