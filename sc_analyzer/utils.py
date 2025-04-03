"""智能合约意图分析的工具函数。"""
import os
import logging
import pickle
import hashlib
from pathlib import Path
import wandb
import numpy as np
import math
from typing import List, Dict, Union, Any, Optional
import traceback


def setup_logger(debug=False):
    """Setup logger to always print time and level."""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(level)
    
    if debug:
        logging.debug("调试模式已启用 - 将显示详细日志")


def log_w_indent(text, indent=0, symbol='>>'):
    """Log message with indentation for better readability."""
    # ANSI color codes for different indentation levels
    color_codes = {
        0: "\033[1m",       # Bold
        1: "\033[31m",      # Red
        2: "\033[33m",      # Yellow
        3: "\033[34m",      # Blue
        4: "\033[35m",      # Magenta
    }
    reset = "\033[0m"
    
    indent_level = max(0, min(indent, len(color_codes) - 1))
    color_code = color_codes[indent_level]
    
    if indent > 0:
        logging.info(color_code + (indent * 2) * symbol + ' ' + text + reset)
    else:
        logging.info(color_code + text + reset)


def md5hash(string):
    """Generate MD5 hash for a string."""
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def calculate_semantic_entropy_from_logprobs(log_probs: List[float]) -> float:
    """
    从一组对数似然概率值计算语义熵。
    
    语义熵是一个衡量概率分布不确定性的指标。对于相似的答案，
    语义熵值较低，表示模型对答案的确定性较高；对于差异较大的答案，
    语义熵值较高，表示模型对答案的不确定性较高。
    
    Args:
        log_probs: 对数似然概率值列表
        
    Returns:
        计算得到的语义熵
    """
    if not log_probs or len(log_probs) == 0:
        return 0.0
        
    # 将对数概率转换为概率
    probs = [math.exp(lp) for lp in log_probs]
    
    # 标准化概率使其和为1
    total = sum(probs)
    if total == 0:
        return 0.0
        
    normalized_probs = [p / total for p in probs]
    
    # 计算香农熵: -sum(p * log(p))
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in normalized_probs)
    return entropy


def calculate_question_entropy(question_data: Dict[str, Any]) -> float:
    """
    计算单个问题的语义熵，基于其多个答案的对数似然概率。
    
    Args:
        question_data: 包含问题和答案信息的字典
        
    Returns:
        问题的语义熵
    """
    log_likelihoods = question_data.get('log_likelihoods', [])
    
    if not log_likelihoods:
        return 0.0
    
    return calculate_semantic_entropy_from_logprobs(log_likelihoods)


def calculate_intent_semantic_entropy(questions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算整个意图的语义熵，基于其所有问题的熵。
    
    Args:
        questions_data: 包含所有问题和答案信息的列表
        
    Returns:
        包含各级语义熵的字典
    """
    if not questions_data:
        return {
            'intent_entropy': 0.0,
            'question_entropies': [],
            'avg_question_entropy': 0.0
        }
    
    # 计算每个问题的熵
    question_entropies = []
    for question_data in questions_data:
        entropy = calculate_question_entropy(question_data)
        question_data['entropy'] = entropy  # 更新问题数据，添加熵
        question_entropies.append(entropy)
    
    # 计算平均问题熵
    avg_question_entropy = sum(question_entropies) / len(question_entropies) if question_entropies else 0
    
    # 计算整体意图熵（问题熵的平均值）
    intent_entropy = avg_question_entropy
    
    return {
        'intent_entropy': intent_entropy,
        'question_entropies': question_entropies,
        'avg_question_entropy': avg_question_entropy
    }


def update_question_data_with_entropy(questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    更新问题数据，添加语义熵信息。
    
    Args:
        questions_data: 要更新的问题数据列表
        
    Returns:
        更新后的问题数据列表
    """
    entropy_results = calculate_intent_semantic_entropy(questions_data)
    
    # 为每个问题添加其相对熵（相对于平均熵）
    avg_entropy = entropy_results['avg_question_entropy']
    
    if avg_entropy > 0:
        for question_data, entropy in zip(questions_data, entropy_results['question_entropies']):
            question_data['entropy'] = entropy
            question_data['relative_entropy'] = entropy / avg_entropy
    else:
        for question_data in questions_data:
            question_data['entropy'] = 0.0
            question_data['relative_entropy'] = 1.0
    
    return questions_data, entropy_results['intent_entropy']


def wandb_restore(wandb_run, filename):
    """从wandb运行中恢复文件。"""
    try:
        api = wandb.Api()
        run = api.run(wandb_run)
        
        temp_dir = Path('tmp_wandb')
        temp_dir.mkdir(exist_ok=True)
        
        run.file(filename).download(
            root=str(temp_dir), replace=True, exist_ok=True)
        file_path = temp_dir / filename
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data, run.config
    
    except Exception as e:
        logging.error(f"从wandb恢复时出错：{e}")
        return None, None


def save_results(results, output_dir, filename='results.pkl'):
    """将结果保存到pickle文件并同步到wandb。
    
    Args:
        results: 要保存的结果数据
        output_dir: 输出目录
        filename: 输出文件名
        
    Returns:
        保存的文件路径
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建完整的输出路径
    output_path = output_dir / filename
    
    # 保存到本地文件
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"结果已保存到本地: {output_path}")
    except Exception as e:
        logging.error(f"保存结果到本地文件时出错: {e}")
        if debug:
            logging.error(traceback.format_exc())
        return None
    
    # 同步到wandb
    try:
        # 使用base_path参数来保持文件夹结构
        wandb.save(str(output_path), base_path=str(output_dir))
        logging.info(f"结果已同步到wandb")
    except Exception as e:
        logging.error(f"同步到wandb时出错: {e}")
        if debug:
            logging.error(traceback.format_exc())
    
    # 同时保存到files/文件夹
    try:
        files_dir = output_dir.parent / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        files_path = files_dir / filename
        with open(files_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"结果已保存到files文件夹: {files_path}")
    except Exception as e:
        logging.error(f"保存结果到files文件夹时出错: {e}")
        if debug:
            logging.error(traceback.format_exc())
    
    return str(output_path)


def load_results(results_path):
    """从pickle文件加载结果。
    
    Args:
        results_path: Pickle文件的路径
        
    Returns:
        加载的结果数据，如果加载失败则返回None
    """
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"成功从 {results_path} 加载结果")
        return results
    except Exception as e:
        logging.error(f"加载结果文件 {results_path} 时出错: {e}")
        if debug:
            logging.error(traceback.format_exc())
        return None


def read_smart_contract(file_path):
    """Read smart contract file with proper encoding handling."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"智能合约文件不存在: {file_path}")
    
    # 尝试判断文件类型
    if file_path.suffix.lower() != '.sol':
        logging.warning(f"文件 {file_path} 不是标准的.sol后缀，但仍尝试读取")
    
    # 记录文件大小
    file_size = file_path.stat().st_size
    logging.info(f"尝试读取智能合约文件: {file_path}，大小: {file_size} 字节")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logging.debug(f"使用UTF-8编码成功读取文件 {file_path}")
            return content
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logging.warning(f"文件 {file_path} 使用 {encoding} 编码解码，而非utf-8")
                return content
            except UnicodeDecodeError:
                continue
        
        # If all fail, read as binary and decode with errors='replace'
        logging.warning(f"尝试所有常见编码都失败，将使用二进制读取并替换无法解码的字符")
        with open(file_path, 'rb') as f:
            binary_content = f.read()
        content = binary_content.decode('utf-8', errors='replace')
        logging.warning(f"文件 {file_path} 无法正确解码，使用了'replace'错误处理")
        return content


def chunk_contract(contract_content, max_tokens=3500, overlap=500):
    """
    Split large contracts into overlapping chunks to fit token limits.
    This is a simple line-based chunking approach.
    """
    lines = contract_content.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        # Crude estimation: 1 token ≈ 4 characters
        line_tokens = len(line) // 4 + 1
        
        if current_length + line_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append('\n'.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_lines = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_lines.copy()
            current_length = sum(len(l) // 4 + 1 for l in current_chunk)
        
        current_chunk.append(line)
        current_length += line_tokens
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def extract_contract_metadata(contract_content):
    """Extract basic metadata from a smart contract."""
    metadata = {
        'contract_names': [],
        'imports': [],
        'pragma': None,
        'total_functions': 0,
        'total_lines': len(contract_content.split('\n')),
        'total_characters': len(contract_content)
    }
    
    lines = contract_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Extract pragma directive
        if line.startswith('pragma ') and metadata['pragma'] is None:
            metadata['pragma'] = line
        
        # Extract imports
        elif line.startswith('import '):
            metadata['imports'].append(line)
        
        # Extract contract names
        elif line.startswith('contract ') or line.startswith('library ') or line.startswith('interface '):
            parts = line.split(' ')
            if len(parts) > 1:
                contract_name = parts[1].split('{')[0].strip()
                metadata['contract_names'].append(contract_name)
        
        # Count function definitions (crude approximation)
        elif 'function ' in line and ('{' in line or ';' in line):
            metadata['total_functions'] += 1
    
    return metadata
