"""智能合约意图分析的工具函数。"""
import os
import logging
import pickle
import hashlib
from pathlib import Path
import wandb


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


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
    """将结果保存到pickle文件。"""
    output_path = Path(output_dir) / filename
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"结果已保存到{output_path}")
    return str(output_path)


def load_results(results_path):
    """从pickle文件加载结果。
    
    Args:
        results_path: Pickle文件的路径
        
    Returns:
        加载的结果数据
    """
    results_path = Path(results_path)
    if not results_path.exists():
        logging.error(f"结果文件不存在: {results_path}")
        return None
    
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"从{results_path}加载结果成功")
        return results
    except Exception as e:
        logging.error(f"加载结果时出错: {e}")
        return None


def read_smart_contract(file_path):
    """Read smart contract file with proper encoding handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logging.warning(f"File {file_path} decoded with {encoding} instead of utf-8")
                return content
            except UnicodeDecodeError:
                continue
        
        # If all fail, read as binary and decode with errors='replace'
        with open(file_path, 'rb') as f:
            binary_content = f.read()
        logging.warning(f"File {file_path} could not be decoded properly, using 'replace' error handling")
        return binary_content.decode('utf-8', errors='replace')


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
