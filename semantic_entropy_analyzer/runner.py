#!/usr/bin/env python3
"""Semantic entropy runner module."""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from sc_analyzer.utils import setup_logger, load_results
from sc_analyzer.data_types import AnalysisResults, EntropyResults
from semantic_entropy_analyzer.semantic_entropy import SemanticEntropyCalculator
from semantic_entropy_analyzer.results_analyzer import ResultsAnalyzer


def setup_argparse():
    """texttexttexttexttexttexttexttexttext """
    parser = argparse.ArgumentParser(description="Semantic entropy analysis tool")
    
    parser.add_argument(
        "--results_path", "-r", type=str,
        help="Path to the intention analysis pickle result file"
    )
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Output directory (default: ./entropy_results)"
    )
    

    
    parser.add_argument(
        "--model_name", type=str, default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    

    
    parser.add_argument(
        "--log_level", "-l", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)"
    )
    
    return parser.parse_args()


def run_entropy_analysis(args) -> Optional[EntropyResults]:
    """texttexttexttexttexttexttext """
    logging.info("Starting semantic entropy analysis...")
    
    # texttexttexttexttexttext
    if not args.results_path:
        logging.error("A result file path is required")
        return None
    
    results: Optional[AnalysisResults] = load_results(args.results_path)
    if not results:
        logging.error(f"Unable to load result file: {args.results_path}")
        return None
    
    # Create semantic entropy calculator
    calculator = SemanticEntropyCalculator(
        results=results,
        model_name=args.model_name,
        mode=args.mode,
        cluster_threshold=args.cluster_threshold,
        debug=args.debug,
        output_dir=args.output_dir
    )
    
    # Calculate semantic entropy
    entropy_results: EntropyResults = calculator.calculate_entropies()
    logging.info(f"Semantic entropy calculation complete; overall entropy: {entropy_results['overall_entropy']:.4f}")
    
    # texttexttexttext
    analyzer = ResultsAnalyzer(
        entropy_results=entropy_results,
        output_dir=args.output_dir
    )
    analysis_results = analyzer.run_analysis()
    
    logging.info("Analysis complete")
    return entropy_results


def main():
    """Main function."""
    # Parse command-line arguments
    args = setup_argparse()
    
    # Set up logging
    setup_logger(args.log_level)
    
    # Run analysis
    entropy_results = run_entropy_analysis(args)
    
    if entropy_results:
        logging.info("Semantic entropy analysis completed successfully")
    else:
        logging.error("Semantic entropy analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())