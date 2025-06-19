"""texttexttexttexttexttexttexttexttexttext """

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

from sc_analyzer.data_types import EntropyResults


class ResultsAnalyzer:
    """texttexttexttexttexttexttext """
    
    def __init__(
        self, 
        entropy_results: EntropyResults,  # texttexttexttexttexttexttexttexttexttexttext
        output_dir: Optional[str] = None
    ):
        """
        texttexttexttexttexttext 
        
        texttext:
            entropy_results: texttexttexttexttexttexttext
            output_dir: texttexttexttexttexttexttexttexttext
        """
        self.entropy_results = entropy_results
        
        # texttexttexttexttexttext
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("./analysis_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # texttexttexttexttexttexttexttext
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info(f"texttexttexttexttexttexttexttext: {self.output_dir}")
    
    def run_analysis(self) -> Dict:
        """
        texttexttexttexttexttext 
        
        texttext:
            texttexttexttext
        """
        logging.info("texttexttexttexttexttexttexttexttext...")
        
        # texttexttexttexttexttexttexttexttext
        question_entropy_analysis = self.analyze_question_entropies()
        
        # texttexttexttexttexttexttexttexttexttext
        section_entropy_analysis = self.analyze_section_entropies()
        
        # texttexttexttext
        self.generate_plots()
        
        # texttexttexttexttexttext
        analysis_results = {
            "question_entropy_analysis": question_entropy_analysis,
            "section_entropy_analysis": section_entropy_analysis,
            "summary": self.entropy_results.get("summary", {})
        }
        
        # texttexttexttext
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"texttexttexttexttexttexttexttext: {results_path}")
        
        return analysis_results
    
    def analyze_question_entropies(self) -> Dict:
        """
        texttexttexttexttexttexttexttexttext 
        
        texttext:
            texttexttexttexttexttexttexttext
        """
        if not self.entropy_results.get("questions"):
            return {"error": "texttexttexttexttexttexttexttexttexttext"}
        
        # texttexttexttexttexttexttexttexttext
        question_entropies = [question["entropy"] for question in self.entropy_results["questions"]]
        
        # texttexttexttexttexttext
        stats = {
            "mean": np.mean(question_entropies) if question_entropies else 0,
            "median": np.median(question_entropies) if question_entropies else 0,
            "std": np.std(question_entropies) if question_entropies else 0,
            "min": np.min(question_entropies) if question_entropies else 0,
            "max": np.max(question_entropies) if question_entropies else 0,
            "count": len(question_entropies)
        }
        
        # texttexttexttexttexttexttexttext
        sorted_questions = sorted(
            [(question["question"], question["entropy"]) 
             for question in self.entropy_results["questions"]],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "stats": stats,
            "sorted_questions": sorted_questions
        }
    
    def analyze_section_entropies(self) -> Dict:
        """
        texttexttexttexttext texttexttexttexttext texttexttexttexttext texttexttexttexttext 
        
        texttext:
            texttexttexttexttexttexttexttexttext
        """
        if not self.entropy_results.get("questions"):
            return {"error": "texttexttexttexttexttexttexttexttexttext"}
        
        # texttexttexttexttexttexttexttexttext
        section_entropies = defaultdict(list)
        
        for question in self.entropy_results["questions"]:
            section_name = question.get("section_name", "unknown")
            section_entropies[section_name].append(question["entropy"])
        
        # texttexttexttexttexttexttexttexttexttexttext
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
        
        # texttexttexttexttexttexttexttexttexttext
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
        """texttexttexttexttexttexttext """
        logging.info("texttexttexttexttexttexttext...")
        
        # texttexttexttexttexttexttexttexttext
        self.plot_question_entropy_distribution()
        
        # texttexttexttexttexttexttexttexttext
        self.plot_section_entropy_comparison()
        
        logging.info(f"texttexttexttexttexttext: {self.plots_dir}")
    
    def plot_question_entropy_distribution(self):
        """texttexttexttexttexttexttexttexttext """
        if not self.entropy_results.get("questions"):
            logging.warning("texttexttexttexttexttext texttexttexttexttexttexttext")
            return
        
        # texttexttexttexttexttexttexttexttext
        question_entropies = [question["entropy"] for question in self.entropy_results["questions"]]
        
        # texttexttexttexttext
        plt.figure(figsize=(10, 6))
        plt.hist(question_entropies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Semantic Entropy')
        plt.ylabel('Number of Questions')
        plt.title('Question Semantic Entropy Distribution')
        plt.grid(True, alpha=0.3)
        
        # texttexttexttexttexttexttexttexttexttext
        mean_entropy = np.mean(question_entropies)
        median_entropy = np.median(question_entropies)
        plt.axvline(mean_entropy, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_entropy:.4f}')
        plt.axvline(median_entropy, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_entropy:.4f}')
        plt.legend()
        
        # texttexttexttext
        plt.tight_layout()
        plt.savefig(self.plots_dir / "question_entropy_distribution.png", dpi=300)
        plt.close()
    
    def plot_section_entropy_comparison(self):
        """texttexttexttexttexttexttexttexttexttext """
        if not self.entropy_results.get("questions"):
            logging.warning("texttexttexttexttexttext texttexttexttexttexttexttext")
            return
        
        # texttexttexttexttexttexttexttexttext
        section_entropies = defaultdict(list)
        
        for question in self.entropy_results["questions"]:
            section_name = question.get("section_name", "unknown")
            section_entropies[section_name].append(question["entropy"])
        
        if not section_entropies:
            logging.warning("texttexttexttexttexttext texttexttexttexttexttexttext")
            return
        
        # texttexttexttexttexttexttexttexttexttexttext
        section_means = {section: np.mean(entropies) for section, entropies in section_entropies.items()}
        
        # texttexttexttexttexttexttexttexttexttext
        sorted_sections = sorted(section_means.items(), key=lambda x: x[1], reverse=True)
        sections, means = zip(*sorted_sections)
        
        # texttexttexttexttexttexttexttexttext
        section_translation = {
            'contract_interaction': 'Contract Interaction',
            'state_changes': 'State Changes',
            'events': 'Events',
            'implications': 'Implications'
        }
        
        # texttexttexttexttexttexttexttext
        english_sections = [section_translation.get(s, s) for s in sections]
        
        # texttexttexttexttext
        plt.figure(figsize=(12, 6))
        bars = plt.bar(english_sections, means, color='skyblue', edgecolor='black')
        plt.xlabel('Intent Section')
        plt.ylabel('Average Semantic Entropy')
        plt.title('Section Semantic Entropy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # texttexttexttexttexttexttexttexttext
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        # texttexttexttext
        plt.tight_layout()
        plt.savefig(self.plots_dir / "section_entropy_comparison.png", dpi=300)
        plt.close() 