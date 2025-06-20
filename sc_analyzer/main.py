"""Main script for analyzing smart contracts and extracting intentions with language models."""
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
    """Main entry point for processing smart contracts and extracting intentions."""
    # Load configuration
    args = get_config()
    prompt_config = PromptConfig()
    
    # Set up logging
    setup_logger(args.log_level)
    
    # Initialize wandb for experiment tracking
    slurm_jobid = os.getenv('SLURM_JOB_ID', 'local_run')
    
    # Use the project-level intent_results directory for outputs
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "intent_results"
    os.makedirs(results_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = results_dir / f"run-{run_id}"
    os.makedirs(run_dir / "files", exist_ok=True)
    
    # Configure wandb
    os.environ["WANDB_DIR"] = str(results_dir)
    os.environ["WANDB_RUN_ID"] = run_id
    
    wandb.init(
        project=PROJECT_NAME,
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
        model_provider=args.model_provider,
        embedding_model=args.embedding_model,
        device=args.device,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy
    )
    
    # Dispatch processing by step argument

    question = "Who is tim keith ferguson?"

    ## TODO Study the effect of the system prompt on answers

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

    
    # Save final results

    save_results(results, wandb.run.dir, RESULTS_FILENAME)


if __name__ == "__main__":
    main()
