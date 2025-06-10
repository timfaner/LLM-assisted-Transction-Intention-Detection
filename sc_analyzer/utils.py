"""texttexttexttexttexttexttexttexttexttexttexttexttext """
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
        logging.debug("texttexttexttexttexttexttext - texttexttexttexttexttexttext")


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
    texttexttexttexttexttexttexttexttexttextCalculate semantic entropy 
    
    texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttext 
    texttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext 
    texttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttext 
    
    Args:
        log_probs: texttexttexttexttexttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttexttext
    """
    if not log_probs or len(log_probs) == 0:
        return 0.0
        
    # texttexttexttexttexttexttexttexttexttext
    probs = [math.exp(lp) for lp in log_probs]
    
    # texttexttexttexttexttexttexttexttext1
    total = sum(probs)
    if total == 0:
        return 0.0
        
    normalized_probs = [p / total for p in probs]
    
    # texttexttexttexttext: -sum(p * log(p))
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in normalized_probs)
    return entropy


def calculate_question_entropy(question_data: Dict[str, Any]) -> float:
    """
    texttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttext 
    
    Args:
        question_data: texttexttexttexttexttexttexttexttexttexttexttext
        
    Returns:
        texttexttexttexttexttext
    """
    log_likelihoods = question_data.get('log_likelihoods', [])
    
    if not log_likelihoods:
        return 0.0
    
    return calculate_semantic_entropy_from_logprobs(log_likelihoods)


def calculate_intent_semantic_entropy(questions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    texttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext 
    
    Args:
        questions_data: texttexttexttexttexttexttexttexttexttexttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttexttexttexttext
    """
    if not questions_data:
        return {
            'intent_entropy': 0.0,
            'question_entropies': [],
            'avg_question_entropy': 0.0
        }
    
    # texttexttexttexttexttexttexttext
    question_entropies = []
    for question_data in questions_data:
        entropy = calculate_question_entropy(question_data)
        question_data['entropy'] = entropy  # texttexttexttexttexttext texttexttext
        question_entropies.append(entropy)
    
    # texttexttexttexttexttexttext
    avg_question_entropy = sum(question_entropies) / len(question_entropies) if question_entropies else 0
    
    # texttexttexttexttexttexttext texttexttexttexttexttexttext 
    intent_entropy = avg_question_entropy
    
    return {
        'intent_entropy': intent_entropy,
        'question_entropies': question_entropies,
        'avg_question_entropy': avg_question_entropy
    }


def update_question_data_with_entropy(questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    texttexttexttexttexttext texttexttexttexttexttexttext 
    
    Args:
        questions_data: texttexttexttexttexttexttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttexttexttexttext
    """
    entropy_results = calculate_intent_semantic_entropy(questions_data)
    
    # texttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttext 
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
    """textwandbtexttexttexttexttexttexttext """
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
        logging.error(f"textwandbtexttexttexttexttext {e}")
        return None, None


def save_results(results, output_dir, filename='results.pkl'):
    """texttexttexttexttexttextpickletexttexttexttexttexttextwandb 
    
    Args:
        results: texttexttexttexttexttexttexttext
        output_dir: texttexttexttext
        filename: texttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttext
    """
    # texttexttexttexttexttexttexttext
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # texttexttexttexttexttexttexttexttext
    output_path = output_dir / filename
    
    # texttexttexttexttexttexttext
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"texttexttexttexttexttexttexttext: {output_path}")
    except Exception as e:
        logging.error(f"texttexttexttexttexttexttexttexttexttexttexttext: {e}")
        if debug:
            logging.error(traceback.format_exc())
        return None
    
    # texttexttextwandb
    try:
        # texttextbase_pathtexttexttexttexttexttexttexttexttexttext
        wandb.save(str(output_path), base_path=str(output_dir))
        logging.info(f"texttexttexttexttexttextwandb")
    except Exception as e:
        logging.error(f"texttexttextwandbtexttexttext: {e}")
        if debug:
            logging.error(traceback.format_exc())
    
    # texttexttexttexttextfiles/texttexttext
    try:
        files_dir = output_dir.parent / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        files_path = files_dir / filename
        with open(files_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"texttexttexttexttexttextfilestexttexttext: {files_path}")
    except Exception as e:
        logging.error(f"texttexttexttexttextfilestexttexttexttexttexttext: {e}")
        if debug:
            logging.error(traceback.format_exc())
    
    return str(output_path)


def load_results(results_path):
    """textpickletexttexttexttexttexttext 
    
    Args:
        results_path: Pickletexttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttext texttexttexttexttexttexttexttexttextNone
    """
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"texttexttext {results_path} texttexttexttext")
        return results
    except Exception as e:
        logging.error(f"texttexttexttexttexttext {results_path} texttexttext: {e}")
        if debug:
            logging.error(traceback.format_exc())
        return None


def read_smart_contract(file_path):
    """Read smart contract file with proper encoding handling."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"texttexttexttexttexttexttexttexttext: {file_path}")
    
    # texttexttexttexttexttexttexttext
    if file_path.suffix.lower() != '.sol':
        logging.warning(f"texttext {file_path} texttexttexttexttext.soltexttext texttexttexttexttexttext")
    
    # texttexttexttexttexttext
    file_size = file_path.stat().st_size
    logging.info(f"texttexttexttexttexttexttexttexttexttext: {file_path} texttext: {file_size} texttext")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logging.debug(f"texttextUTF-8texttexttexttexttexttexttexttext {file_path}")
            return content
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logging.warning(f"texttext {file_path} texttext {encoding} texttexttexttext texttextutf-8")
                return content
            except UnicodeDecodeError:
                continue
        
        # If all fail, read as binary and decode with errors='replace'
        logging.warning(f"texttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext")
        with open(file_path, 'rb') as f:
            binary_content = f.read()
        content = binary_content.decode('utf-8', errors='replace')
        logging.warning(f"texttext {file_path} texttexttexttexttexttext texttexttext'replace'texttexttexttext")
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
