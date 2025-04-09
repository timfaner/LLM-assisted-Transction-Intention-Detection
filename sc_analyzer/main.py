"""使用大语言模型分析智能合约并提取其意图的主脚本。"""
import os
import logging
import datetime
import traceback
import re
from pathlib import Path
import json


from sc_analyzer.data_types import Question

import wandb
  

from sc_analyzer.models import get_model, LegacyAPIModel
from sc_analyzer.utils import (
    setup_logger, log_w_indent, md5hash, save_results,
)
from sc_analyzer.config import get_config,PromptConfig


# Global constants
PROJECT_NAME = 'smart_contract_intent'
RESULTS_FILENAME = 'results.pkl'



def main():
    """处理智能合约并提取意图的主函数。"""
    # 获取配置
    args = get_config()
    prompt_config = PromptConfig()
    
    # 设置日志
    setup_logger(args.log_level)
    
    # Initialize wandb for experiment tracking
    slurm_jobid = os.getenv('SLURM_JOB_ID', 'local_run')
    
    # 使用项目根目录下的intent_results文件夹作为结果保存路径
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "intent_results"
    os.makedirs(results_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = results_dir / f"run-{run_id}"
    os.makedirs(run_dir / "files", exist_ok=True)
    
    # 设置wandb配置
    os.environ["WANDB_DIR"] = str(results_dir)
    os.environ["WANDB_RUN_ID"] = run_id
    
    wandb.init(
        project=PROJECT_NAME,
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
        model_provider=args.model_provider,
        embedding_model=args.embedding_model,
        device=args.device,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy
    )
    
    # 根据步骤参数执行相应的处理

    question = "Who is tim keith ferguson?"

    ## todo 研究system promot 对答案的影响

    system_prompt = prompt_config.get_system_prompt("generate_answers")
    answers = model.get_multiple_answers(
        answers_num = 7, 
        answers_temperature = 1, 
        system_prompt = system_prompt,
        question = question,
        )

    q = Question(
        question_id = "1",
        question = question,
        answers = answers,
        entropy = 100,
        relative_entropy = 100
    )

    results = [q]

    
    # 保存最终结果

    save_results(results, wandb.run.dir, RESULTS_FILENAME)


if __name__ == "__main__":
    main()
