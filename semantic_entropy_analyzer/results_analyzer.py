"""分析语义熵结果的模块。"""

import os
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict


class ResultsAnalyzer:
    """分析语义熵结果。"""
    
    def __init__(
        self, 
        entropy_results: Dict,  # 直接接收计算好的熵结果
        output_dir: Optional[str] = None
    ):
        """
        初始化分析器。
        
        参数:
            entropy_results: 语义熵结果字典
            output_dir: 保存分析结果的目录
        """
        self.entropy_results = entropy_results
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("./analysis_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置图表输出目录
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info(f"分析结果将保存至: {self.output_dir}")
    
    def run_analysis(self) -> Dict:
        """
        运行完整分析。
        
        返回:
            分析结果
        """
        logging.info("开始分析语义熵结果...")
        
        # 分析各合约的熵值分布
        contract_entropy_analysis = self.analyze_contract_entropies()
        
        # 分析各部分的熵值分布
        section_entropy_analysis = self.analyze_section_entropies()
        
        # 生成图表
        self.generate_plots()
        
        # 合并分析结果
        analysis_results = {
            "contract_entropy_analysis": contract_entropy_analysis,
            "section_entropy_analysis": section_entropy_analysis,
            "summary": self.entropy_results["summary"]
        }
        
        # 保存结果
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"分析结果已保存至: {results_path}")
        
        return analysis_results
    
    def analyze_contract_entropies(self) -> Dict:
        """
        分析各合约的熵值分布。
        
        返回:
            合约熵值分析结果
        """
        if not self.entropy_results.get("contracts"):
            return {"error": "没有可分析的合约数据"}
        
        # 收集所有合约的熵值
        contract_entropies = [contract["avg_overall_entropy"] for contract in self.entropy_results["contracts"]]
        
        # 计算统计信息
        stats = {
            "mean": np.mean(contract_entropies) if contract_entropies else 0,
            "median": np.median(contract_entropies) if contract_entropies else 0,
            "std": np.std(contract_entropies) if contract_entropies else 0,
            "min": np.min(contract_entropies) if contract_entropies else 0,
            "max": np.max(contract_entropies) if contract_entropies else 0,
            "count": len(contract_entropies)
        }
        
        # 对合约按熵值排序
        sorted_contracts = sorted(
            [(contract["contract_path"], contract["avg_overall_entropy"]) 
             for contract in self.entropy_results["contracts"]],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "stats": stats,
            "sorted_contracts": sorted_contracts
        }
    
    def analyze_section_entropies(self) -> Dict:
        """
        分析各部分（如合约交互、状态变化等）的熵值分布。
        
        返回:
            各部分熵值分析结果
        """
        if not self.entropy_results.get("contracts"):
            return {"error": "没有可分析的合约数据"}
        
        # 按部分名称收集熵值
        section_entropies = defaultdict(list)
        
        for contract in self.entropy_results["contracts"]:
            for intent in contract["intent_entropies"]:
                for section_name, entropy in intent["section_entropies"].items():
                    section_entropies[section_name].append(entropy)
        
        # 计算每个部分的统计信息
        section_stats = {}
        for section_name, entropies in section_entropies.items():
            section_stats[section_name] = {
                "mean": np.mean(entropies) if entropies else 0,
                "median": np.median(entropies) if entropies else 0,
                "std": np.std(entropies) if entropies else 0,
                "min": np.min(entropies) if entropies else 0,
                "max": np.max(entropies) if entropies else 0,
                "count": len(entropies)
            }
        
        # 对部分按平均熵值排序
        sorted_sections = sorted(
            [(section_name, stats["mean"]) 
             for section_name, stats in section_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "section_stats": section_stats,
            "sorted_sections": sorted_sections
        }
    
    def generate_plots(self):
        """生成可视化图表。"""
        logging.info("生成可视化图表...")
        
        # 生成合约熵值分布图
        self.plot_contract_entropy_distribution()
        
        # 生成部分熵值对比图
        self.plot_section_entropy_comparison()
        
        logging.info(f"图表已保存至: {self.plots_dir}")
    
    def plot_contract_entropy_distribution(self):
        """绘制合约熵值分布图。"""
        if not self.entropy_results.get("contracts"):
            logging.warning("没有合约数据，无法生成分布图")
            return
        
        # 收集所有合约的熵值
        contract_entropies = [contract["avg_overall_entropy"] for contract in self.entropy_results["contracts"]]
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(contract_entropies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Semantic Entropy')
        plt.ylabel('Number of Contracts')
        plt.title('Contract Semantic Entropy Distribution')
        plt.grid(True, alpha=0.3)
        
        # 添加平均值和中位数线
        mean_entropy = np.mean(contract_entropies)
        median_entropy = np.median(contract_entropies)
        plt.axvline(mean_entropy, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_entropy:.4f}')
        plt.axvline(median_entropy, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_entropy:.4f}')
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(self.plots_dir / "contract_entropy_distribution.png", dpi=300)
        plt.close()
    
    def plot_section_entropy_comparison(self):
        """绘制各部分熵值对比图。"""
        if not self.entropy_results.get("contracts"):
            logging.warning("没有合约数据，无法生成对比图")
            return
        
        # 按部分名称收集熵值
        section_entropies = defaultdict(list)
        
        for contract in self.entropy_results["contracts"]:
            for intent in contract["intent_entropies"]:
                for section_name, entropy in intent["section_entropies"].items():
                    section_entropies[section_name].append(entropy)
        
        if not section_entropies:
            logging.warning("没有部分数据，无法生成对比图")
            return
        
        # 计算每个部分的平均熵值
        section_means = {section: np.mean(entropies) for section, entropies in section_entropies.items()}
        
        # 对部分按平均熵值排序
        sorted_sections = sorted(section_means.items(), key=lambda x: x[1], reverse=True)
        sections, means = zip(*sorted_sections)
        
        # 翻译部分名称为英文
        section_translation = {
            'contract_interaction': 'Contract Interaction',
            'state_changes': 'State Changes',
            'events': 'Events',
            'implications': 'Implications'
        }
        
        # 使用翻译后的名称
        english_sections = [section_translation.get(s, s) for s in sections]
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(english_sections, means, color='skyblue', edgecolor='black')
        plt.xlabel('Intent Section')
        plt.ylabel('Average Semantic Entropy')
        plt.title('Section Semantic Entropy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # 在条形上添加值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(self.plots_dir / "section_entropy_comparison.png", dpi=300)
        plt.close() 