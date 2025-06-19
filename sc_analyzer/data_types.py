"""texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext 

texttexttexttexttexttexttexttexttexttexttexttexttexttexttextpickletexttexttexttexttexttexttext 
texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext 
"""
from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime
from pathlib import Path


class AnswerWithArgLogprob(TypedDict):
    """texttexttexttexttexttexttext """
    answer_id: str
    answer: str
    avg_logprob: float
    token_log_likelihoods: List[float]
    embedding: Optional[List[float]]


AnswerWithArgLogprobList = List[AnswerWithArgLogprob]


class Question(TypedDict):
    """texttexttexttexttexttexttexttexttexttexttext """
    question_id: str
    question: str
    answers: AnswerWithArgLogprobList
    entropy: float  # texttexttexttexttexttext
    relative_entropy: float  # texttexttexttexttexttexttexttext


class Section(TypedDict):
    """texttexttexttexttexttexttext texttexttexttexttext texttexttexttexttext  """
    section_id: str
    section_name: str
    section_content: str
    questions: List[Question]


class IntentData(TypedDict):
    """texttexttexttexttexttexttexttexttext """
    intent_id: str
    test_id: int
    contract_path: str
    contract_file: str
    timestamp: str
    intent: str
    intent_length: int
    token_log_likelihoods: List[float]
    embedding: Optional[List[float]]
    sections: List[Section]  # texttexttext2texttexttext


class ContractData(TypedDict):
    """texttexttexttexttexttexttexttexttexttexttext """
    folder_path: str
    relative_path: str
    contract_file: str
    transaction_file: Optional[str]
    content_length: int
    content_hash: str
    test_results: List[IntentData]


class IndexEntry(TypedDict):
    """texttexttexttext """
    contract_path: str
    test_idx: int


class Indexes(TypedDict):
    """texttexttexttexttexttexttext """
    intent_index: Dict[str, IndexEntry]
    section_index: Dict[str, IndexEntry]
    question_index: Dict[str, IndexEntry]
    answer_index: Dict[str, IndexEntry]


class TestConfig(TypedDict):
    """texttexttexttexttexttext """
    num_tests: int
    test_start_time: str

## wrong
class AnalysisResults(TypedDict):
    """texttexttexttexttexttexttext """
    contract_intents: Dict[str, ContractData]
    prompts: Dict[str, str]
    model_info: Dict[str, Any]
    test_config: TestConfig
    indexes: Indexes


# texttexttexttexttexttexttexttexttexttexttexttext
class QuestionEntropy(TypedDict):
    """texttexttexttexttexttexttexttext """
    question_id: str
    question: str
    section_name: str
    entropy: float
    num_answers: int
    num_clusters: int


class IntentEntropy(TypedDict):
    """texttexttexttexttexttexttexttext """
    intent_id: str
    overall_entropy: float
    section_entropies: Dict[str, float]


class ContractEntropy(TypedDict):
    """texttexttexttexttexttexttexttext """
    contract_path: str
    avg_overall_entropy: float
    intent_entropies: List[IntentEntropy]


class EntropySummary(TypedDict):
    """texttexttexttexttexttexttexttexttexttext """
    time_taken: float
    mode: str
    num_contracts: int
    avg_overall_entropy: float


class EntropyResults(TypedDict):
    """texttexttexttexttexttexttexttexttexttext """
    contracts: List[ContractEntropy]
    questions: List[QuestionEntropy]
    overall_entropy: float
    summary: EntropySummary 