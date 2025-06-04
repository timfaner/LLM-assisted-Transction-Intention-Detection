#!/usr/bin/env python3
"""Runner script for semantic entropy analysis."""

import os
import argparse
import logging
from pathlib import Path
import sys

# texttexttexttext
try:
    from sc_analyzer.utils import setup_logger
except ImportError:
    # texttexttexttexttexttext texttexttexttexttexttexttextsetup_loggertexttext
    def setup_logger(level=logging.INFO):
        """Set up loggingtexttext"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

from semantic_entropy_analyzer.semantic_entropy import SemanticEntropyCalculator
from semantic_entropy_analyzer.results_analyzer import ResultsAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="texttexttexttexttexttexttexttexttexttexttexttexttexttexttext")
    
    # Required parameters
    parser.add_argument("--results_path", type=str, required=True, 
                        help="texttexttexttexttexttexttexttexttexttext")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None,
                        help="texttexttexttexttexttexttexttexttexttext")
    
    # Entropy calculation parameters
    parser.add_argument("--device", type=str, default=None,
                        help="texttexttexttext cudatextcpu ")
    parser.add_argument("--no_api", action="store_true",
                        help="texttexttextAPItexttexttexttexttexttexttexttext texttexttexttextAPI ")
    
    # Analysis parameters
    parser.add_argument("--skip_analysis", action="store_true",
                        help="texttexttexttexttexttext")
    parser.add_argument("--save_detailed", action="store_true",
                        help="texttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--debug", action="store_true",
                        help="texttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext")
    
    # Logging parameters
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="texttexttexttext")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure results path exists
        results_path = Path(args.results_path)
        if not results_path.exists():
            logger.error(f"texttexttexttexttexttexttext: {results_path}")
            return 1
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = results_path.parent / "entropy_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"texttexttexttext {results_path} texttexttexttext")
        logger.info(f"texttextAPItexttexttexttexttexttexttexttext: {not args.no_api}")
        
        # Calculate entropies
        calculator = SemanticEntropyCalculator(
            results_path=str(results_path),
            output_dir=str(output_dir),
            device=args.device,
            use_api_for_equivalence=not args.no_api
        )
        
        entropy_results = calculator.calculate_entropies()
        
        # texttexttexttexttexttext
        logger.info("==== Semantic Entropy Calculation Results Summary ====")
        logger.info(f"Number of Contracts Analyzed: {entropy_results['summary']['num_contracts']}")
        logger.info(f"Overall Average Semantic Entropy: {entropy_results['summary']['avg_overall_entropy']:.4f}")
        logger.info(f"Calculation Time: {entropy_results['summary']['time_taken']:.2f} seconds")
        
        if not args.skip_analysis and 'contracts' in entropy_results and entropy_results['contracts']:
            # Run analysis
            logger.info("==== Starting Semantic Entropy Analysis ====")
            
            # texttexttexttexttexttexttexttext
            for contract in entropy_results['contracts']:
                contract_path = contract['contract_path']
                avg_entropy = contract['avg_overall_entropy']
                logger.info(f"Contract {contract_path}: Average Semantic Entropy = {avg_entropy:.4f}")
                
                # texttexttexttexttexttexttexttext
                if args.save_detailed:
                    for intent in contract['intent_entropies']:
                        intent_id = intent['intent_id']
                        intent_entropy = intent['overall_entropy']
                        logger.info(f"  Intent {intent_id}: Semantic Entropy = {intent_entropy:.4f}")
                        
                        # texttexttexttexttexttexttexttext
                        for section_name, section_entropy in intent['section_entropies'].items():
                            # texttexttexttexttexttext
                            section_translation = {
                                'contract_interaction': 'Contract Interaction',
                                'state_changes': 'State Changes',
                                'events': 'Events',
                                'implications': 'Implications'
                            }
                            english_section = section_translation.get(section_name, section_name)
                            logger.info(f"    {english_section}: {section_entropy:.4f}")
            
            # texttexttexttexttexttext
            analyzer = ResultsAnalyzer(
                entropy_results=entropy_results,
                output_dir=output_dir / "analysis"
            )
            
            analysis_results = analyzer.run_analysis()
            
            logger.info(f"Analysis results saved to {output_dir / 'analysis'}")
        
        logger.info("Semantic entropy analysis completed!")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("texttexttexttexttexttext")
        return 130
    
    except Exception as e:
        logger.exception(f"texttexttexttexttexttexttext: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())