"""使用大语言模型分析智能合约并提取其意图的主脚本。"""
import os
import argparse
import logging
import datetime
from pathlib import Path
import pickle
from collections import defaultdict

import wandb
import torch

from sc_analyzer.models import get_model
from sc_analyzer.utils import setup_logger, log_w_indent, md5hash, save_results

# Global constants
PROJECT_NAME = 'smart_contract_intent'
RESULTS_FILENAME = 'results.pkl'

def main(args):
    """处理智能合约并提取意图的主函数。"""
    # Setup logging and environment
    setup_logger()
    
    # Initialize wandb for experiment tracking
    user = os.getenv('USER', 'default_user')
    slurm_jobid = os.getenv('SLURM_JOB_ID', 'local_run')
    wandb_path = args.wandb_dir or f'/tmp/{user}/sc_intent'
    os.makedirs(wandb_path, exist_ok=True)
    
    wandb.init(
        project=PROJECT_NAME if not args.debug else f"{PROJECT_NAME}_debug",
        dir=wandb_path,
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
        api_key=args.api_key,
        use_local=args.use_local,
        local_model_path=args.local_model_path,
        device=args.device
    )
    
    # Scan the input directory for smart contract files
    contracts_path = Path(args.input_dir)
    contract_files = list(contracts_path.glob("**/*.sol"))
    logging.info(f"在{args.input_dir}中找到{len(contract_files)}个智能合约文件")
    
    # Process each contract file
    results = {
        'contract_intents': {},
        'prompts': model.get_prompts_for_log(),
        'model_info': model.get_model_info(),
        'test_config': {
            'num_tests': args.num_tests,
            'test_start_time': datetime.datetime.now().isoformat()
        }
    }
    
    for idx, contract_file in enumerate(contract_files):
        if idx >= args.max_contracts and args.max_contracts > 0:
            logging.info(f"已达到要处理的最大合约数量({args.max_contracts})")
            break
            
        relative_path = contract_file.relative_to(contracts_path)
        logging.info(f"正在处理合约 {idx+1}/{len(contract_files)}: {relative_path}")
        
        # Read contract content
        try:
            with open(contract_file, 'r', encoding='utf-8') as f:
                contract_content = f.read()
        except Exception as e:
            logging.error(f"读取合约文件 {contract_file} 时出错: {e}")
            continue
        
        # Create a list to store results for each contract
        contract_results = []
        
        # Perform n tests for each contract
        for test_idx in range(args.num_tests):
            logging.info(f"正在进行测试 {test_idx+1}/{args.num_tests}")
            
            # Generate intent using LLM
            try:
                intent = model.generate_intent(contract_content)
                logging.info(f"已为 {relative_path} 生成第 {test_idx+1} 次意图")
                log_w_indent(f"意图 {test_idx+1}: {intent}", indent=1)
                
                # Save test result
                test_result = {
                    'test_id': test_idx,
                    'intent': intent,
                    'intent_length': len(intent),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                contract_results.append(test_result)
                
                # Log to wandb
                wandb.log({
                    'contracts_processed': idx + 1,
                    'test_id': test_idx,
                    'latest_contract_length': len(contract_content),
                    'latest_intent_length': len(intent)
                })
                
            except Exception as e:
                logging.error(f"为合约 {contract_file} 生成第 {test_idx+1} 次意图时出错: {e}")
                continue
        
        # Save all test results for the contract
        results['contract_intents'][str(relative_path)] = {
            'file_path': str(contract_file),
            'relative_path': str(relative_path),
            'content_length': len(contract_content),
            'content_hash': md5hash(contract_content),
            'test_results': contract_results
        }
            
        # Save intermediate results
        if args.save_interval > 0 and (idx + 1) % args.save_interval == 0:
            save_results(results, wandb.run.dir)
    
    # Save final results
    results['test_config']['test_end_time'] = datetime.datetime.now().isoformat()
    save_results(results, wandb.run.dir)
    logging.info("运行成功完成")
    
def save_results(results, output_dir):
    """将结果保存到pickle文件。"""
    output_path = os.path.join(output_dir, RESULTS_FILENAME)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"结果已保存到 {output_path}")
    wandb.save(RESULTS_FILENAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能合约意图分析器")
    
    # Input/output options
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含智能合约文件的目录")
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
                        help="模型名称（例如，gpt-4, claude-3-opus）")
    parser.add_argument("--api_key", type=str, default=None,
                        help="模型服务的API密钥")
    parser.add_argument("--use_local", action="store_true",
                        help="使用本地下载的大语言模型")
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="本地下载模型的路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行本地模型的设备")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    
    args = parser.parse_args()
    main(args)
