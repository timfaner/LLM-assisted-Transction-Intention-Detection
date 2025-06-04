"""Main script for analyzing smart contracts and extracting intentions with language models."""
import os
import argparse
import logging
import datetime
import traceback
import re
from pathlib import Path
import pickle
import json
import uuid
from collections import defaultdict

import wandb
import torch

from sc_analyzer.models import get_model
from sc_analyzer.utils import (
    setup_logger, log_w_indent, md5hash, save_results, read_smart_contract
)

# Global constants
PROJECT_NAME = 'smart_contract_intent'
RESULTS_FILENAME = 'results.pkl'

# texttexttexttexttexttexttexttexttext
INTENT_SECTIONS = [
    'contract_interaction', 
    'state_changes', 
    'events', 
    'implications'
]

def generate_unique_id(prefix="", suffix=""):
    """texttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
    unique_part = str(uuid.uuid4()).split("-")[0]  # textUUIDtexttexttexttexttexttexttexttexttexttexttexttext
    return f"{prefix}{unique_part}{suffix}"

def parse_intent_sections(intent):
    """texttexttexttexttexttexttexttexttexttext 
    
    Args:
        intent: texttexttexttexttexttexttext
        
    Returns:
        texttexttexttexttexttexttexttexttexttext
    """
    sections = {}
    
    # texttexttexttexttransaction_analysistexttext
    transaction_match = re.search(r'<transaction_analysis>(.*?)</transaction_analysis>', 
                                 intent, re.DOTALL)
    
    if transaction_match:
        transaction_text = transaction_match.group(1).strip()
        
        # texttexttexttexttexttext
        contract_match = re.search(r'<contract_interaction>(.*?)</contract_interaction>', 
                                  transaction_text, re.DOTALL)
        state_match = re.search(r'<state_changes>(.*?)</state_changes>', 
                               transaction_text, re.DOTALL)
        events_match = re.search(r'<events>(.*?)</events>', 
                                transaction_text, re.DOTALL)
        implications_match = re.search(r'<implications>(.*?)</implications>', 
                                     transaction_text, re.DOTALL)
        
        # texttexttexttexttexttexttext
        sections['full_transaction'] = transaction_text
        sections['contract_interaction'] = contract_match.group(1).strip() if contract_match else ""
        sections['state_changes'] = state_match.group(1).strip() if state_match else ""
        sections['events'] = events_match.group(1).strip() if events_match else ""
        sections['implications'] = implications_match.group(1).strip() if implications_match else ""
    else:
        # texttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttext
        sections['full_transaction'] = intent
        # texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext
        for section in INTENT_SECTIONS:
            section_match = re.search(rf'<{section}>(.*?)</{section}>', intent, re.DOTALL)
            sections[section] = section_match.group(1).strip() if section_match else ""
    
    return sections

def main(args):
    """Main entry point for processing smart contracts and extracting intentions."""
    # Setup logging and environment
    setup_logger(args.debug)
    
    # texttextOpenAI APItexttexttexttexttexttext
    if args.model_type == "api" and not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        logging.error("texttextAPItexttexttexttexttexttextOpenAI APItexttext texttexttext--api_keytexttexttexttexttextOPENAI_API_KEYtexttexttexttext ")
        return 1
    
    # Initialize wandb for experiment tracking
    user = os.getenv('USER', 'default_user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', 'local_run')
    
    # Use the project-level intent_results directory for outputs
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "intent_results"
    os.makedirs(results_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run-{run_id}"
    os.makedirs(run_dir / "files", exist_ok=True)
    
    # Configure wandb
    os.environ["WANDB_DIR"] = str(results_dir)
    os.environ["WANDB_RUN_ID"] = run_id
    
    wandb.init(
        project=PROJECT_NAME if not args.debug else f"{PROJECT_NAME}_debug",
        dir=str(results_dir),
        config=args,
        notes=f"SLURM_JOB_ID: {slurm_jobid}"
    )
    
    logging.info('New run started with configuration:')
    logging.info(args)
    logging.info('Wandb setup complete.')

    # Initialize model for contract analysis
    model = get_model(
        model_type=args.model_type,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        api_key=args.api_key,
        use_local=args.use_local,
        local_model_path=args.local_model_path,
        device=args.device,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy
    )
    
    # Scan the input directory for contract folders
    contracts_path = Path(args.input_dir)
    contract_folders = [f for f in contracts_path.iterdir() if f.is_dir()]
    logging.info(f"text{args.input_dir}texttexttext{len(contract_folders)}texttexttexttexttexttext")
    
    if len(contract_folders) == 0:
        logging.warning(f"texttext text{args.input_dir}texttexttexttexttexttexttexttexttexttexttext")
    
    # Process each contract folder
    results = {
        'contract_intents': {},
        'prompts': model.get_prompts_for_log(),
        'model_info': model.get_model_info(),
        'test_config': {
            'num_tests': args.num_tests,
            'test_start_time': datetime.datetime.now().isoformat()
        }
    }
    
    # texttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttext
    results['indexes'] = {
        'intent_index': {},      # intent_id -> texttexttexttexttexttexttexttexttext
        'section_index': {},     # section_id -> intent_id texttexttexttexttext
        'question_index': {},    # question_id -> section_id texttexttexttexttext
        'answer_index': {}       # answer_id -> question_id texttexttexttexttext
    }
    
    for idx, contract_folder in enumerate(contract_folders):
        if idx >= args.max_contracts and args.max_contracts > 0:
            logging.info(f"texttexttexttexttexttexttexttexttexttexttexttexttext({args.max_contracts})")
            break
            
        relative_path = contract_folder.relative_to(contracts_path)
        logging.info(f"texttexttexttexttexttexttexttexttext {idx+1}/{len(contract_folders)}: {relative_path}")
        
        # texttexttexttexttexttexttexttexttexttexttexttexttext
        contract_file = None
        transaction_file = None
        
        # texttexttexttexttexttexttexttexttexttexttext
        if args.debug:
            logging.debug(f"texttexttext {contract_folder} texttext:")
            for file in contract_folder.iterdir():
                logging.debug(f"  - {file.name} ({file.suffix})")
        
        for file in contract_folder.iterdir():
            if file.suffix.lower() == '.sol':
                contract_file = file
                logging.info(f"texttexttexttexttexttext: {file.name}")
            elif file.suffix.lower() in ['.json', '.xlsx', '.csv', '.txt']:
                transaction_file = file
                logging.info(f"texttexttexttexttexttexttexttext: {file.name}")
        
        if not contract_file:
            logging.error(f"texttexttexttext {contract_folder} texttexttexttext.soltexttexttexttext")
            continue
            
        if not transaction_file:
            logging.warning(f"texttexttexttext {contract_folder} texttexttexttexttexttexttexttexttexttext")
        
        # texttexttexttexttexttext
        try:
            contract_content = read_smart_contract(contract_file)
            logging.info(f"texttexttexttext {contract_file.name} texttexttexttext texttext {len(contract_content)}texttext")
            
            # texttexttexttexttexttexttexttexttexttexttexttexttexttexttext
            if args.debug:
                content_preview = contract_content[:100] + ("..." if len(contract_content) > 100 else "")
                logging.debug(f"texttexttexttexttexttext: {content_preview}")
        except Exception as e:
            logging.error(f"texttexttexttexttexttext {contract_file} texttexttext: {e}")
            if args.debug:
                logging.error(traceback.format_exc())
            continue
        
        # texttexttexttexttexttext
        transaction_data = ""
        if transaction_file:
            try:
                if transaction_file.suffix.lower() == '.json':
                    with open(transaction_file, 'r', encoding='utf-8') as f:
                        transaction_data = json.dumps(json.load(f), indent=2)
                else:
                    with open(transaction_file, 'r', encoding='utf-8') as f:
                        transaction_data = f.read()
                logging.info(f"texttexttexttexttexttext {transaction_file.name} texttexttexttext texttext {len(transaction_data)}texttext")
                
                # texttexttexttexttexttexttexttexttexttexttexttexttexttexttext
                if args.debug:
                    data_preview = transaction_data[:100] + ("..." if len(transaction_data) > 100 else "")
                    logging.debug(f"texttexttexttexttexttext: {data_preview}")
            except Exception as e:
                logging.error(f"texttexttexttexttexttext {transaction_file} texttexttext: {e}")
                if args.debug:
                    logging.error(traceback.format_exc())
        
        # Create a list to store results for each contract
        contract_results = []
        
        # Perform n tests for each contract
        for test_idx in range(args.num_tests):
            logging.info(f"texttexttexttexttexttext {test_idx+1}/{args.num_tests}")
            
            # Generate intent using LLM
            try:
                logging.info(f"texttexttexttexttexttexttexttexttexttext...")
                intent_result = model.generate_intent(contract_content, transaction_data)
                
                # texttexttexttexttexttext
                if isinstance(intent_result, tuple) and len(intent_result) >= 3:
                    intent, token_log_likelihoods, embedding = intent_result
                else:
                    # texttexttexttexttexttexttexttexttext
                    intent = intent_result
                    token_log_likelihoods = []
                    embedding = None
                
                logging.info(f"texttext {relative_path} texttexttext {test_idx+1} texttexttext texttext: {len(intent)}texttext")
                log_w_indent(f"texttext {test_idx+1} text100texttexttext: {intent[:100]}...", indent=1)
                
                # texttexttexttexttexttexttexttexttext
                parsed_sections = parse_intent_sections(intent)
                logging.info(f"texttexttexttexttexttext {len(parsed_sections)} texttexttext")
                
                # texttexttexttexttexttexttexttexttexttext
                intent_id = generate_unique_id(prefix=f"intent_{contract_file.stem}_{test_idx}_")
                
                # texttexttexttext
                results['indexes']['intent_index'][intent_id] = {
                    'contract_path': str(relative_path),
                    'test_idx': test_idx
                }
                
                # texttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttext
                intent_data = {
                    'intent_id': intent_id,
                    'test_id': test_idx,
                    'contract_path': str(relative_path),
                    'contract_file': str(contract_file),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'intent': intent,
                    'intent_length': len(intent),
                    'token_log_likelihoods': token_log_likelihoods,
                    'embedding': embedding,
                    'sections': []  # texttexttexttexttexttexttexttext
                }
                
                # texttexttexttexttexttext
                total_questions = 0
                total_answers = 0
                
                for section_name in INTENT_SECTIONS:
                    section_content = parsed_sections.get(section_name, "")
                    if not section_content:
                        logging.warning(f"texttexttext {section_name} texttexttexttext")
                        continue
                    
                    section_id = generate_unique_id(prefix=f"{intent_id}_{section_name}_")
                    
                    # texttexttexttext
                    results['indexes']['section_index'][section_id] = {
                        'intent_id': intent_id,
                        'section_name': section_name
                    }
                    
                    logging.info(f"texttexttext {section_name} texttexttexttexttexttext...")
                    
                    # texttexttexttexttexttexttexttext
                    section_questions = model.generate_questions_for_section(section_content, section_name)
                    logging.info(f"text {section_name} texttexttexttexttexttext {len(section_questions)} texttexttext")
                    
                    # texttexttexttext
                    for q_idx, question in enumerate(section_questions):
                        log_w_indent(f"{section_name} texttext {q_idx+1}: {question[:100]}...", indent=2)
                    
                    section_data = {
                        'section_id': section_id,
                        'intent_id': intent_id,
                        'section_name': section_name,
                        'content': section_content,
                        'questions': []
                    }
                    
                    # texttexttexttexttexttexttexttexttext
                    questions_list = []
                    
                    for q_idx, question in enumerate(section_questions):
                        question_id = generate_unique_id(prefix=f"{section_id}_q{q_idx}_")
                        
                        # texttexttexttext
                        results['indexes']['question_index'][question_id] = {
                            'section_id': section_id,
                            'question_idx': q_idx
                        }
                        
                        logging.info(f"texttexttext {section_name} texttexttexttexttext {q_idx+1} texttexttexttext...")
                        
                        # texttexttexttext
                        answers = model.generate_answers(question, section_content)
                        total_answers += len(answers)
                        
                        # texttexttexttexttextlogprobtexttexttexttexttexttext
                        log_likelihoods = [a['avg_logprob'] for a in answers if a['avg_logprob'] is not None]
                        
                        question_data = {
                            'question_id': question_id,
                            'section_id': section_id,
                            'intent_id': intent_id,
                            'question_idx': q_idx,
                            'question': question,
                            'answers': [],
                            'log_likelihoods': log_likelihoods
                        }
                        
                        # texttexttexttext
                        for a_idx, answer_data in enumerate(answers):
                            answer_id = generate_unique_id(prefix=f"{question_id}_a{a_idx}_")
                            
                            # texttexttexttext
                            results['indexes']['answer_index'][answer_id] = {
                                'question_id': question_id,
                                'answer_idx': a_idx
                            }
                            
                            answer_entry = {
                                'answer_id': answer_id,
                                'question_id': question_id,
                                'section_id': section_id,
                                'intent_id': intent_id,
                                'answer_idx': a_idx,
                                'answer': answer_data['answer'],
                                'avg_logprob': answer_data['avg_logprob'],
                                'logprobs': answer_data['logprobs']
                            }
                            
                            question_data['answers'].append(answer_entry)
                        
                        questions_list.append(question_data)
                    
                    section_data['questions'] = questions_list
                    total_questions += len(questions_list)
                    intent_data['sections'].append(section_data)
                
                # texttexttexttexttexttexttexttexttext
                contract_results.append(intent_data)
                
                # Log to wandb
                log_data = {
                    'contracts_processed': idx + 1,
                    'test_id': test_idx,
                    'latest_contract_length': len(contract_content),
                    'latest_intent_length': len(intent),
                    'token_log_likelihood_available': len(token_log_likelihoods) > 0,
                    'avg_token_log_likelihood': sum(token_log_likelihoods) / len(token_log_likelihoods) if token_log_likelihoods else None,
                    'embedding_available': embedding is not None,
                    'num_sections_processed': len(intent_data['sections']),
                    'num_questions_generated': total_questions,
                    'num_answers_generated': total_answers
                }
                
                if embedding is not None:
                    log_data['embedding_dimension'] = len(embedding)
                
                wandb.log(log_data)
                
            except Exception as e:
                logging.error(f"texttexttext {contract_folder} texttexttext {test_idx+1} texttexttexttexttexttext: {e}")
                if args.debug:
                    logging.error(traceback.format_exc())
                continue
        
        # Save all test results for the contract
        results['contract_intents'][str(relative_path)] = {
            'folder_path': str(contract_folder),
            'relative_path': str(relative_path),
            'contract_file': str(contract_file),
            'transaction_file': str(transaction_file) if transaction_file else None,
            'content_length': len(contract_content),
            'content_hash': md5hash(contract_content),
            'test_results': contract_results
        }
            
        # Save intermediate results
        if args.save_interval > 0 and (idx + 1) % args.save_interval == 0:
            results['test_config']['last_processed'] = str(contract_folder)
            save_results(results, wandb.run.dir, RESULTS_FILENAME)
            logging.info(f"texttexttexttexttexttexttext texttexttext {idx}/{len(contract_folders)} texttexttexttexttexttext")
    
    # Save final results
    results['test_config']['test_end_time'] = datetime.datetime.now().isoformat()
    # texttexttextwandb
    save_results(results, wandb.run.dir, RESULTS_FILENAME)
    # texttexttexttexttexttexttexttexttexttexttext
    save_direct_results(results, run_dir)
    
    logging.info(f"texttexttext {len(results['contract_intents'])} texttexttexttexttexttext")
    logging.info(f"texttexttexttexttexttext {wandb.run.dir}/{RESULTS_FILENAME} text {run_dir}/files/{RESULTS_FILENAME}")

def save_results(results, output_dir, filename):
    """texttexttexttexttexttextpickletexttext """
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"texttexttexttexttexttext {output_path}")
    wandb.save(filename)

def save_direct_results(results, run_dir):
    """texttexttexttexttexttexttexttexttexttexttexttext texttexttextwandb """
    output_path = run_dir / "files" / RESULTS_FILENAME
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"texttexttexttexttexttexttexttext {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="texttexttexttexttexttexttexttexttext")
    
    # Input/output options
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="texttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="wandbtexttexttexttexttext texttext: /tmp/<texttexttext>/sc_intent ")
    parser.add_argument("--max_contracts", type=int, default=0,
                        help="texttexttexttexttexttexttexttexttexttext 0texttexttexttexttext ")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="texttexttexttexttexttexttexttexttexttexttexttexttext 0texttexttexttext ")
    parser.add_argument("--num_tests", type=int, default=6,
                        help="texttexttexttexttexttexttexttexttexttexttext")
    
    # Model options
    parser.add_argument("--model_type", type=str, default="api",
                        choices=["api", "local"], help="texttexttexttexttexttexttext")
    parser.add_argument("--model_name", type=str, default="gpt-4",
                        help="OpenAItexttexttexttext texttext gpt-3.5-turbo, gpt-4 ")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                        help="OpenAItexttexttexttexttexttext")
    parser.add_argument("--api_key", type=str, default=None,
                        help="texttexttexttexttextAPItexttext")
    parser.add_argument("--use_local", action="store_true",
                        help="texttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="texttexttexttexttexttexttexttexttext")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="texttexttexttexttexttexttexttexttext")
    
    # texttexttexttext
    parser.add_argument("--http_proxy", type=str, default=None,
                        help="HTTPtexttexttexttext texttext http://127.0.0.1:7890 ")
    parser.add_argument("--https_proxy", type=str, default=None,
                        help="HTTPStexttexttexttext texttext http://127.0.0.1:7890 ")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="texttexttexttexttexttext")
    
    args = parser.parse_args()
    main(args)
