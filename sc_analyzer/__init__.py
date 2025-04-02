"""智能合约意图分析器包。"""
__version__ = '0.1.0'

from .models import get_model, BaseModel, LocalLLMModel, APIModel
from .utils import (
    setup_logger, 
    log_w_indent, 
    md5hash, 
    wandb_restore, 
    save_results,
    read_smart_contract,
    chunk_contract,
    extract_contract_metadata,
    load_results
)

# 注意：避免循环导入
# semantic_entropy_analyzer包需要从sc_analyzer导入，所以我们不能在这里导入
# from semantic_entropy_analyzer import SemanticEntropyCalculator, ResultsAnalyzer

__all__ = [
    'get_model', 
    'BaseModel', 
    'LocalLLMModel', 
    'APIModel',
    'setup_logger',
    'log_w_indent',
    'md5hash',
    'wandb_restore',
    'save_results',
    'read_smart_contract',
    'chunk_contract',
    'extract_contract_metadata',
    'load_results'
]
