"""智能合约分析系统中使用的数据结构定义。

本文件定义了系统中所有序列化到pickle文件的数据结构，
以提供类型提示和结构化的数据访问方式。
"""
from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime
from pathlib import Path


class AnswerWithArgLogprob(TypedDict):
    """问题的单次回答。"""
    answer_id: str
    answer: str
    avg_logprob: float
    token_log_likelihoods: List[float]
    embedding: Optional[List[float]]


AnswerWithArgLogprobList = List[AnswerWithArgLogprob]


class Question(TypedDict):
    """针对意图特定部分的问题。"""
    question_id: str
    question: str
    answers: AnswerWithArgLogprobList
    entropy: float  # 问题的语义熵
    relative_entropy: float  # 相对于平均熵的值


class Section(TypedDict):
    """意图的一个部分（如合约交互、状态变化等）。"""
    section_id: str
    section_name: str
    section_content: str
    questions: List[Question]


class IntentData(TypedDict):
    """单次意图分析的结果。"""
    intent_id: str
    test_id: int
    contract_path: str
    contract_file: str
    timestamp: str
    intent: str
    intent_length: int
    token_log_likelihoods: List[float]
    embedding: Optional[List[float]]
    sections: List[Section]  # 在步骤2后添加


class ContractData(TypedDict):
    """单个智能合约的分析数据。"""
    folder_path: str
    relative_path: str
    contract_file: str
    transaction_file: Optional[str]
    content_length: int
    content_hash: str
    test_results: List[IntentData]


class IndexEntry(TypedDict):
    """索引条目。"""
    contract_path: str
    test_idx: int


class Indexes(TypedDict):
    """各种元素的索引。"""
    intent_index: Dict[str, IndexEntry]
    section_index: Dict[str, IndexEntry]
    question_index: Dict[str, IndexEntry]
    answer_index: Dict[str, IndexEntry]


class TestConfig(TypedDict):
    """测试配置信息。"""
    num_tests: int
    test_start_time: str

## wrong
class AnalysisResults(TypedDict):
    """完整的分析结果。"""
    contract_intents: Dict[str, ContractData]
    prompts: Dict[str, str]
    model_info: Dict[str, Any]
    test_config: TestConfig
    indexes: Indexes


# 语义熵计算结果的数据结构
class QuestionEntropy(TypedDict):
    """问题的语义熵数据。"""
    question_id: str
    question: str
    section_name: str
    entropy: float
    num_answers: int
    num_clusters: int


class IntentEntropy(TypedDict):
    """意图的语义熵数据。"""
    intent_id: str
    overall_entropy: float
    section_entropies: Dict[str, float]


class ContractEntropy(TypedDict):
    """合约的语义熵数据。"""
    contract_path: str
    avg_overall_entropy: float
    intent_entropies: List[IntentEntropy]


class EntropySummary(TypedDict):
    """语义熵计算的摘要信息。"""
    time_taken: float
    mode: str
    num_contracts: int
    avg_overall_entropy: float


class EntropyResults(TypedDict):
    """语义熵计算的完整结果。"""
    contracts: List[ContractEntropy]
    questions: List[QuestionEntropy]
    overall_entropy: float
    summary: EntropySummary 