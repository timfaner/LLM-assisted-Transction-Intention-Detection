#!/usr/bin/env python3
"""语义熵分析的运行器模块。"""

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
    """设置命令行参数解析。"""
    parser = argparse.ArgumentParser(description="语义熵分析工具")
    
    parser.add_argument(
        "--results_path", "-r", type=str,
        help="意图分析结果的pickle文件路径"
    )
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="输出目录 (默认: ./entropy_results)"
    )
    

    
    parser.add_argument(
        "--model_name", type=str, default="all-MiniLM-L6-v2",
        help="嵌入模型名称 (默认: all-MiniLM-L6-v2)"
    )
    

    
    parser.add_argument(
        "--log_level", "-l", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别 (默认: INFO)"
    )
    
    return parser.parse_args()


def run_entropy_analysis(args) -> Optional[EntropyResults]:
    """运行语义熵分析。"""
    logging.info("开始语义熵分析...")
    
    # 加载分析结果
    if not args.results_path:
        logging.error("必须指定结果文件路径")
        return None
    
    results: Optional[AnalysisResults] = load_results(args.results_path)
    if not results:
        logging.error(f"无法加载结果文件: {args.results_path}")
        return None
    
    # 创建语义熵计算器
    calculator = SemanticEntropyCalculator(
        results=results,
        model_name=args.model_name,
        mode=args.mode,
        cluster_threshold=args.cluster_threshold,
        debug=args.debug,
        output_dir=args.output_dir
    )
    
    # 计算语义熵
    entropy_results: EntropyResults = calculator.calculate_entropies()
    logging.info(f"语义熵计算完成，整体熵: {entropy_results['overall_entropy']:.4f}")
    
    # 分析结果
    analyzer = ResultsAnalyzer(
        entropy_results=entropy_results,
        output_dir=args.output_dir
    )
    analysis_results = analyzer.run_analysis()
    
    logging.info("分析完成")
    return entropy_results


def main():
    """主函数。"""
    # 解析命令行参数
    args = setup_argparse()
    
    # 设置日志
    setup_logger(args.log_level)
    
    # 运行分析
    entropy_results = run_entropy_analysis(args)
    
    if entropy_results:
        logging.info("语义熵分析成功完成")
    else:
        logging.error("语义熵分析失败")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())