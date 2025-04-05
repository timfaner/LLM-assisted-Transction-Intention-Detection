#!/usr/bin/env python3
"""Runner script for semantic entropy analysis."""

import os
import argparse
import logging
from pathlib import Path
import sys

# 修改导入
try:
    from sc_analyzer.utils import setup_logger
except ImportError:
    # 如果无法导入，创建一个简单的setup_logger函数
    def setup_logger(level=logging.INFO):
        """设置日志记录"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

from semantic_entropy_analyzer.semantic_entropy import SemanticEntropyCalculator
from semantic_entropy_analyzer.results_analyzer import ResultsAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="计算并分析智能合约意图的语义熵")
    
    # Required parameters
    parser.add_argument("--results_path", type=str, required=True, 
                        help="意图分析结果文件路径")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=None,
                        help="保存语义熵结果的目录")
    
    # Entropy calculation parameters
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备（cuda或cpu）")
    parser.add_argument("--no_api", action="store_true",
                        help="不使用API进行语义等价判断（默认使用API）")
    parser.add_argument("--mode", type=str, choices=["step3", "all"], default="step3",
                        help="生成模式：step3或all")
    
    # Analysis parameters
    parser.add_argument("--skip_analysis", action="store_true",
                        help="跳过分析阶段")
    parser.add_argument("--save_detailed", action="store_true",
                        help="保存详细的熵分析结果")
    parser.add_argument("--debug", action="store_true",
                        help="开启详细调试日志，显示语义熵计算过程")
    
    # Logging parameters
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
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
            logger.error(f"结果文件未找到: {results_path}")
            return 1
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = results_path.parent / "entropy_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"开始计算 {results_path} 的语义熵")
        logger.info(f"使用API进行语义等价判断: {not args.no_api}")
        logger.info(f"使用模式: {args.mode}")
        
        # Calculate entropies
        calculator = SemanticEntropyCalculator(
            results_path=str(results_path),
            output_dir=str(output_dir),
            device=args.device,
            use_api_for_equivalence=not args.no_api,
            mode=args.mode
        )
        
        entropy_results = calculator.calculate_entropies()
        
        # 输出摘要信息
        logger.info("==== Semantic Entropy Calculation Results Summary ====")
        logger.info(f"Mode: {entropy_results['summary']['mode']}")
        logger.info(f"Overall Average Semantic Entropy: {entropy_results['overall_entropy']:.4f}")
        logger.info(f"Calculation Time: {entropy_results['summary']['time_taken']:.2f} seconds")
        
        if not args.skip_analysis and 'questions' in entropy_results and entropy_results['questions']:
            # Run analysis
            logger.info("==== Starting Semantic Entropy Analysis ====")
            
            # 输出每个问题的熵
            for question in entropy_results['questions']:
                question_text = question['question']
                entropy = question['entropy']
                num_clusters = question['num_clusters']
                logger.info(f"Question: {question_text}")
                logger.info(f"  Entropy: {entropy:.4f}")
                logger.info(f"  Clusters: {num_clusters}")
            
            # 保存分析结果
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
        logger.info("用户中断进程")
        return 130
    
    except Exception as e:
        logger.exception(f"语义熵分析出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())