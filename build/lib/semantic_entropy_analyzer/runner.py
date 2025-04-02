#!/usr/bin/env python3
"""Runner script for semantic entropy analysis."""

import os
import argparse
import logging
from pathlib import Path
import wandb
import sys

from sc_analyzer.utils import setup_logger
from semantic_entropy_analyzer import SemanticEntropyCalculator, ResultsAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate and analyze semantic entropy of smart contract intents")
    
    # Required parameters
    parser.add_argument("--results_path", type=str, required=True, 
                        help="Path to the input results pickle file")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save entropy results")
    
    # Entropy calculation parameters
    parser.add_argument("--entailment_model", type=str, default="cross-encoder/nli-deberta-v3-base",
                        help="Entailment model to use")
    parser.add_argument("--entailment_type", type=str, default="nli", choices=["nli", "stsb"],
                        help="Type of entailment model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    
    # Analysis parameters
    parser.add_argument("--original_results_path", type=str, default=None,
                        help="Path to the original results.pkl file for extra analysis")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip the analysis phase")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="smart_contract_intent",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="Directory for W&B files")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    # Logging parameters
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    return parser.parse_args()


def init_wandb(args):
    """Initialize Weights & Biases tracking."""
    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=args.wandb_dir,
            config=vars(args),
            mode="disabled" if args.debug else "online"
        )
        return True
    except ImportError:
        logging.warning("W&B not available, skipping experiment tracking")
        return False
    except Exception as e:
        logging.warning(f"Failed to initialize W&B: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(log_level=log_level)
    
    # Initialize W&B
    use_wandb = init_wandb(args)
    
    try:
        # Ensure results path exists
        results_path = Path(args.results_path)
        if not results_path.exists():
            logging.error(f"Results file not found: {results_path}")
            return 1
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = results_path.parent / "entropy_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info(f"Starting semantic entropy calculation for {results_path}")
        
        # Calculate entropies
        calculator = SemanticEntropyCalculator(
            results_path=args.results_path,
            output_dir=output_dir,
            entailment_model=args.entailment_model,
            entailment_type=args.entailment_type,
            device=args.device
        )
        
        entropy_results = calculator.calculate_entropies()
        entropy_results_path = output_dir / "entropy_results.pkl"
        
        if not args.skip_analysis:
            # Run analysis
            logging.info("Starting entropy results analysis")
            analyzer = ResultsAnalyzer(
                entropy_results_path=entropy_results_path,
                original_results_path=args.original_results_path,
                output_dir=output_dir / "analysis"
            )
            
            analysis_results = analyzer.run_full_analysis()
            
            if use_wandb:
                # Log analysis results to W&B
                wandb.log({"analysis_complete": True})
                
                # Upload analysis files to W&B
                report_path = output_dir / "analysis" / "entropy_analysis_report.md"
                if report_path.exists():
                    wandb.save(str(report_path))
                
                # Upload plots
                for plot_file in (output_dir / "analysis").glob("*.png"):
                    wandb.log({f"plot/{plot_file.stem}": wandb.Image(str(plot_file))})
        
        logging.info("Semantic entropy analysis complete!")
        logging.info(f"Results saved to {output_dir}")
        
        if use_wandb:
            # Log the final paths
            wandb.log({
                "output_dir": str(output_dir),
                "entropy_results_path": str(entropy_results_path)
            })
            
            # Create a summary
            wandb.run.summary.update({
                "num_contracts": entropy_results["summary"]["total_contracts"],
                "mean_overall_entropy": float(entropy_results["summary"]["mean_entropy"].get("overall", 0)),
                "max_overall_entropy": float(entropy_results["summary"]["max_entropy"].get("overall", 0))
            })
            
            wandb.finish()
        
        return 0
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        if use_wandb:
            wandb.finish()
        return 130
    
    except Exception as e:
        logging.exception(f"Error in entropy analysis: {e}")
        if use_wandb:
            wandb.finish()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 