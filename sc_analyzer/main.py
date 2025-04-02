"""使用大语言模型分析智能合约并提取其意图的主脚本。"""
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

# 定义意图的各个部分
INTENT_SECTIONS = [
    'contract_interaction', 
    'state_changes', 
    'events', 
    'implications'
]

def generate_unique_id(prefix="", suffix=""):
    """生成带有前缀和后缀的唯一标识符。"""
    unique_part = str(uuid.uuid4()).split("-")[0]  # 取UUID的第一部分作为简短标识符
    return f"{prefix}{unique_part}{suffix}"

def parse_intent_sections(intent):
    """解析意图中的各个部分。
    
    Args:
        intent: 完整的意图文本
        
    Returns:
        包含各部分内容的字典
    """
    sections = {}
    
    # 提取整个transaction_analysis部分
    transaction_match = re.search(r'<transaction_analysis>(.*?)</transaction_analysis>', 
                                 intent, re.DOTALL)
    
    if transaction_match:
        transaction_text = transaction_match.group(1).strip()
        
        # 提取各子部分
        contract_match = re.search(r'<contract_interaction>(.*?)</contract_interaction>', 
                                  transaction_text, re.DOTALL)
        state_match = re.search(r'<state_changes>(.*?)</state_changes>', 
                               transaction_text, re.DOTALL)
        events_match = re.search(r'<events>(.*?)</events>', 
                                transaction_text, re.DOTALL)
        implications_match = re.search(r'<implications>(.*?)</implications>', 
                                     transaction_text, re.DOTALL)
        
        # 存储各部分内容
        sections['full_transaction'] = transaction_text
        sections['contract_interaction'] = contract_match.group(1).strip() if contract_match else ""
        sections['state_changes'] = state_match.group(1).strip() if state_match else ""
        sections['events'] = events_match.group(1).strip() if events_match else ""
        sections['implications'] = implications_match.group(1).strip() if implications_match else ""
    else:
        # 如果没有找到标准格式，保存整个内容作为完整交易
        sections['full_transaction'] = intent
        # 尝试其他可能的格式或返回空字符串
        for section in INTENT_SECTIONS:
            section_match = re.search(rf'<{section}>(.*?)</{section}>', intent, re.DOTALL)
            sections[section] = section_match.group(1).strip() if section_match else ""
    
    return sections

def main(args):
    """处理智能合约并提取意图的主函数。"""
    # Setup logging and environment
    setup_logger(args.debug)
    
    # 检查OpenAI API密钥是否设置
    if args.model_type == "api" and not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        logging.error("使用API模式但未设置OpenAI API密钥。请使用--api_key参数或设置OPENAI_API_KEY环境变量。")
        return 1
    
    # Initialize wandb for experiment tracking
    user = os.getenv('USER', 'default_user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', 'local_run')
    
    # 使用项目根目录下的intent_results文件夹作为结果保存路径
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "intent_results"
    os.makedirs(results_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run-{run_id}"
    os.makedirs(run_dir / "files", exist_ok=True)
    
    # 设置wandb配置
    os.environ["WANDB_DIR"] = str(results_dir)
    os.environ["WANDB_RUN_ID"] = run_id
    
    wandb.init(
        project=PROJECT_NAME if not args.debug else f"{PROJECT_NAME}_debug",
        dir=str(results_dir),
        config=args,
        notes=f"SLURM_JOB_ID: {slurm_jobid}"
    )
    
    logging.info('新运行开始，配置如下:')
    logging.info(args)
    logging.info('Wandb设置完成。')

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
    logging.info(f"在{args.input_dir}中找到{len(contract_folders)}个合约文件夹")
    
    if len(contract_folders) == 0:
        logging.warning(f"警告：在{args.input_dir}中未找到任何合约文件夹")
    
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
    
    # 添加辅助索引，用于快速查找意图和问题
    results['indexes'] = {
        'intent_index': {},      # intent_id -> 合约路径和测试索引
        'section_index': {},     # section_id -> intent_id 和部分名称
        'question_index': {},    # question_id -> section_id 和问题索引
        'answer_index': {}       # answer_id -> question_id 和答案索引
    }
    
    for idx, contract_folder in enumerate(contract_folders):
        if idx >= args.max_contracts and args.max_contracts > 0:
            logging.info(f"已达到要处理的最大合约数量({args.max_contracts})")
            break
            
        relative_path = contract_folder.relative_to(contracts_path)
        logging.info(f"正在处理合约文件夹 {idx+1}/{len(contract_folders)}: {relative_path}")
        
        # 查找合约文件和交易数据文件
        contract_file = None
        transaction_file = None
        
        # 列出文件夹内容用于调试
        if args.debug:
            logging.debug(f"文件夹 {contract_folder} 内容:")
            for file in contract_folder.iterdir():
                logging.debug(f"  - {file.name} ({file.suffix})")
        
        for file in contract_folder.iterdir():
            if file.suffix.lower() == '.sol':
                contract_file = file
                logging.info(f"找到合约文件: {file.name}")
            elif file.suffix.lower() in ['.json', '.xlsx', '.csv', '.txt']:
                transaction_file = file
                logging.info(f"找到交易数据文件: {file.name}")
        
        if not contract_file:
            logging.error(f"在文件夹 {contract_folder} 中未找到.sol合约文件")
            continue
            
        if not transaction_file:
            logging.warning(f"在文件夹 {contract_folder} 中未找到交易数据文件")
        
        # 读取合约内容
        try:
            contract_content = read_smart_contract(contract_file)
            logging.info(f"合约文件 {contract_file.name} 读取成功，大小：{len(contract_content)}字节")
            
            # 输出合约内容的简短摘要用于调试
            if args.debug:
                content_preview = contract_content[:100] + ("..." if len(contract_content) > 100 else "")
                logging.debug(f"合约内容预览: {content_preview}")
        except Exception as e:
            logging.error(f"读取合约文件 {contract_file} 时出错: {e}")
            if args.debug:
                logging.error(traceback.format_exc())
            continue
        
        # 读取交易数据
        transaction_data = ""
        if transaction_file:
            try:
                if transaction_file.suffix.lower() == '.json':
                    with open(transaction_file, 'r', encoding='utf-8') as f:
                        transaction_data = json.dumps(json.load(f), indent=2)
                else:
                    with open(transaction_file, 'r', encoding='utf-8') as f:
                        transaction_data = f.read()
                logging.info(f"交易数据文件 {transaction_file.name} 读取成功，大小：{len(transaction_data)}字节")
                
                # 输出交易数据的简短摘要用于调试
                if args.debug:
                    data_preview = transaction_data[:100] + ("..." if len(transaction_data) > 100 else "")
                    logging.debug(f"交易数据预览: {data_preview}")
            except Exception as e:
                logging.error(f"读取交易文件 {transaction_file} 时出错: {e}")
                if args.debug:
                    logging.error(traceback.format_exc())
        
        # Create a list to store results for each contract
        contract_results = []
        
        # Perform n tests for each contract
        for test_idx in range(args.num_tests):
            logging.info(f"正在进行测试 {test_idx+1}/{args.num_tests}")
            
            # Generate intent using LLM
            try:
                logging.info(f"正在调用模型生成意图...")
                intent_result = model.generate_intent(contract_content, transaction_data)
                
                # 解包返回结果
                if isinstance(intent_result, tuple) and len(intent_result) >= 3:
                    intent, token_log_likelihoods, embedding = intent_result
                else:
                    # 兼容旧版本返回格式
                    intent = intent_result
                    token_log_likelihoods = []
                    embedding = None
                
                logging.info(f"已为 {relative_path} 生成第 {test_idx+1} 次意图，长度: {len(intent)}字符")
                log_w_indent(f"意图 {test_idx+1} 前100个字符: {intent[:100]}...", indent=1)
                
                # 解析意图的各个部分
                parsed_sections = parse_intent_sections(intent)
                logging.info(f"意图已分解为 {len(parsed_sections)} 个部分")
                
                # 生成唯一的意图标识符
                intent_id = generate_unique_id(prefix=f"intent_{contract_file.stem}_{test_idx}_")
                
                # 更新索引
                results['indexes']['intent_index'][intent_id] = {
                    'contract_path': str(relative_path),
                    'test_idx': test_idx
                }
                
                # 重新组织数据结构，添加唯一标识符和明确的层次关系
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
                    'sections': []  # 将单独填充各部分
                }
                
                # 处理每个部分
                total_questions = 0
                total_answers = 0
                
                for section_name in INTENT_SECTIONS:
                    section_content = parsed_sections.get(section_name, "")
                    if not section_content:
                        logging.warning(f"未找到 {section_name} 部分内容")
                        continue
                    
                    section_id = generate_unique_id(prefix=f"{intent_id}_{section_name}_")
                    
                    # 更新索引
                    results['indexes']['section_index'][section_id] = {
                        'intent_id': intent_id,
                        'section_name': section_name
                    }
                    
                    logging.info(f"正在为 {section_name} 部分生成问题...")
                    
                    # 为该部分生成问题
                    section_questions = model.generate_questions_for_section(section_content, section_name)
                    logging.info(f"为 {section_name} 部分成功生成 {len(section_questions)} 个问题")
                    
                    # 记录问题
                    for q_idx, question in enumerate(section_questions):
                        log_w_indent(f"{section_name} 问题 {q_idx+1}: {question[:100]}...", indent=2)
                    
                    section_data = {
                        'section_id': section_id,
                        'intent_id': intent_id,
                        'section_name': section_name,
                        'content': section_content,
                        'questions': []
                    }
                    
                    # 为每个问题生成答案
                    questions_list = []
                    
                    for q_idx, question in enumerate(section_questions):
                        question_id = generate_unique_id(prefix=f"{section_id}_q{q_idx}_")
                        
                        # 更新索引
                        results['indexes']['question_index'][question_id] = {
                            'section_id': section_id,
                            'question_idx': q_idx
                        }
                        
                        logging.info(f"正在为 {section_name} 部分的问题 {q_idx+1} 生成答案...")
                        
                        # 生成答案
                        answers = model.generate_answers(question, section_content)
                        total_answers += len(answers)
                        
                        # 收集问题的logprob值但不计算熵
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
                        
                        # 添加答案
                        for a_idx, answer_data in enumerate(answers):
                            answer_id = generate_unique_id(prefix=f"{question_id}_a{a_idx}_")
                            
                            # 更新索引
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
                
                # 添加意图数据到结果
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
                logging.error(f"为合约 {contract_folder} 生成第 {test_idx+1} 次意图时出错: {e}")
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
            logging.info(f"已保存中间结果，完成了 {idx}/{len(contract_folders)} 个合约的处理")
    
    # Save final results
    results['test_config']['test_end_time'] = datetime.datetime.now().isoformat()
    # 保存到wandb
    save_results(results, wandb.run.dir, RESULTS_FILENAME)
    # 同时直接保存到项目目录
    save_direct_results(results, run_dir)
    
    logging.info(f"已完成 {len(results['contract_intents'])} 个合约的分析")
    logging.info(f"结果已保存到 {wandb.run.dir}/{RESULTS_FILENAME} 和 {run_dir}/files/{RESULTS_FILENAME}")

def save_results(results, output_dir, filename):
    """将结果保存到pickle文件。"""
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"结果已保存到 {output_path}")
    wandb.save(filename)

def save_direct_results(results, run_dir):
    """直接将结果保存到指定目录，不依赖wandb。"""
    output_path = run_dir / "files" / RESULTS_FILENAME
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"结果已直接保存到 {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能合约意图分析器")
    
    # Input/output options
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含智能合约文件夹的目录")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="wandb文件的目录（默认: /tmp/<用户名>/sc_intent）")
    parser.add_argument("--max_contracts", type=int, default=0,
                        help="要处理的最大合约数量（0表示无限制）")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="每处理这么多合约后保存结果（0表示禁用）")
    parser.add_argument("--num_tests", type=int, default=6,
                        help="每个合约运行的测试次数")
    
    # Model options
    parser.add_argument("--model_type", type=str, default="api",
                        choices=["api", "local"], help="使用的模型类型")
    parser.add_argument("--model_name", type=str, default="gpt-4",
                        help="OpenAI模型名称（例如，gpt-3.5-turbo, gpt-4）")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                        help="OpenAI嵌入模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="模型服务的API密钥")
    parser.add_argument("--use_local", action="store_true",
                        help="使用本地下载的大语言模型")
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="本地下载模型的路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行本地模型的设备")
    
    # 代理设置
    parser.add_argument("--http_proxy", type=str, default=None,
                        help="HTTP代理地址（例如：http://127.0.0.1:7890）")
    parser.add_argument("--https_proxy", type=str, default=None,
                        help="HTTPS代理地址（例如：http://127.0.0.1:7890）")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    
    args = parser.parse_args()
    main(args)
