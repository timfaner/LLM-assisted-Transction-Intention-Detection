"""Analyze and visualize results from smart contract intent analysis."""
import os
import logging
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import wandb
from typing import Dict, List, Any, Tuple, Optional

from sc_analyzer.utils import setup_logger


class ResultsAnalyzer:
    """Analyze and visualize entropy results from smart contract intent analysis."""
    
    def __init__(self, entropy_results_path, original_results_path=None, output_dir=None):
        """
        Initialize the results analyzer.
        
        Args:
            entropy_results_path: Path to the entropy_results.pkl file
            original_results_path: Path to the original results.pkl file (optional)
            output_dir: Directory to save output visualizations
        """
        self.entropy_results_path = Path(entropy_results_path)
        self.original_results_path = Path(original_results_path) if original_results_path else None
        self.output_dir = Path(output_dir) if output_dir else self.entropy_results_path.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load entropy results
        with open(self.entropy_results_path, 'rb') as f:
            self.entropy_results = pickle.load(f)
        
        # Load original results if provided
        self.original_results = None
        if self.original_results_path and self.original_results_path.exists():
            with open(self.original_results_path, 'rb') as f:
                self.original_results = pickle.load(f)
        
        # Extract sections from the first contract
        sample_contract = next(iter(self.entropy_results['contracts'].values()))
        self.sections = list(sample_contract['section_entropies'].keys())
        
        # Create a DataFrame for easier analysis
        self.contract_df = self._create_contracts_dataframe()
    
    def _create_contracts_dataframe(self):
        """Create a DataFrame from the contract entropy results."""
        data = []
        
        for contract_key, contract_data in self.entropy_results['contracts'].items():
            row = {
                'contract': contract_key,
                'file_path': contract_data['file_path'],
                'relative_path': contract_data['relative_path'],
                'overall_entropy': contract_data['overall_entropy'],
                'num_intent_analyses': contract_data['num_intent_analyses']
            }
            
            # Add section entropies
            for section in self.sections:
                row[f'{section}_entropy'] = contract_data['section_entropies'].get(section, np.nan)
            
            # Add metadata from original results if available
            if self.original_results:
                original_contract = self.original_results['contract_intents'].get(contract_key, {})
                row['content_length'] = original_contract.get('content_length', 0)
                row['content_hash'] = original_contract.get('content_hash', '')
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_summary_stats(self):
        """Generate summary statistics for entropy values."""
        summary = self.entropy_results['summary']
        
        stats = {
            'overall': {
                'mean': summary['mean_entropy'].get('overall', np.nan),
                'min': summary['min_entropy'].get('overall', np.nan),
                'max': summary['max_entropy'].get('overall', np.nan),
                'median': self.contract_df['overall_entropy'].median(),
                'std': self.contract_df['overall_entropy'].std()
            }
        }
        
        # Add section stats
        for section in self.sections:
            section_values = self.contract_df[f'{section}_entropy'].dropna()
            if len(section_values) > 0:
                stats[section] = {
                    'mean': summary['mean_entropy'].get(section, np.nan),
                    'min': summary['min_entropy'].get(section, np.nan),
                    'max': summary['max_entropy'].get(section, np.nan),
                    'median': section_values.median(),
                    'std': section_values.std()
                }
        
        # Save stats as JSON
        stats_path = self.output_dir / "entropy_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def plot_overall_entropy_distribution(self):
        """Plot the distribution of overall entropy values."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.contract_df['overall_entropy'].dropna(), kde=True)
        plt.title('Distribution of Overall Entropy Across Contracts')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_entropy_distribution.png", dpi=300)
        plt.close()
    
    def plot_section_entropy_comparison(self):
        """Plot a comparison of entropy values across different sections."""
        # Prepare data
        section_data = []
        for section in self.sections:
            values = self.contract_df[f'{section}_entropy'].dropna()
            if len(values) > 0:
                for value in values:
                    section_data.append({'Section': section, 'Entropy': value})
        
        section_df = pd.DataFrame(section_data)
        
        # Create plot
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='Section', y='Entropy', data=section_df)
        plt.title('Entropy Comparison Across Contract Sections')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "section_entropy_comparison.png", dpi=300)
        plt.close()
        
        # Also create a bar chart with means and error bars
        plt.figure(figsize=(12, 7))
        section_means = section_df.groupby('Section')['Entropy'].mean()
        section_std = section_df.groupby('Section')['Entropy'].std()
        
        section_means.plot(kind='bar', yerr=section_std, capsize=4, figsize=(12, 7))
        plt.title('Mean Entropy by Contract Section (with Standard Deviation)')
        plt.ylabel('Mean Entropy')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "section_entropy_means.png", dpi=300)
        plt.close()
    
    def plot_heatmap(self):
        """Plot a heatmap of entropy values across contracts and sections."""
        # Prepare data
        heatmap_data = self.contract_df.set_index('contract')
        section_columns = [f'{section}_entropy' for section in self.sections]
        heatmap_data = heatmap_data[section_columns]
        
        # Limit to 30 contracts if there are too many
        if len(heatmap_data) > 30:
            logging.info(f"Limiting heatmap to 30 contracts (out of {len(heatmap_data)})")
            # Sort by overall entropy and take highest 30
            top_contracts = self.contract_df.sort_values('overall_entropy', ascending=False).head(30)['contract']
            heatmap_data = heatmap_data.loc[top_contracts]
        
        # Rename columns to be more readable
        heatmap_data.columns = [col.replace('_entropy', '') for col in heatmap_data.columns]
        
        # Create plot
        plt.figure(figsize=(12, max(8, len(heatmap_data) * 0.3)))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title('Entropy Heatmap by Contract and Section')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_heatmap.png", dpi=300)
        plt.close()
    
    def identify_high_entropy_contracts(self, threshold_percentile=90):
        """Identify contracts with high entropy values."""
        threshold = np.percentile(self.contract_df['overall_entropy'].dropna(), threshold_percentile)
        high_entropy_contracts = self.contract_df[self.contract_df['overall_entropy'] >= threshold]
        
        # Sort by entropy in descending order
        high_entropy_contracts = high_entropy_contracts.sort_values('overall_entropy', ascending=False)
        
        # Save to CSV
        csv_path = self.output_dir / f"high_entropy_contracts_p{threshold_percentile}.csv"
        high_entropy_contracts.to_csv(csv_path, index=False)
        
        return high_entropy_contracts
    
    def analyze_cluster_patterns(self):
        """Analyze patterns in semantic clusters across contracts."""
        cluster_stats = {
            'avg_clusters_per_section': {},
            'max_clusters': {},
            'single_cluster_percentage': {}
        }
        
        for section in self.sections:
            section_clusters = []
            single_cluster_count = 0
            max_clusters = 0
            
            for contract_key, contract_data in self.entropy_results['contracts'].items():
                clusters = contract_data['section_clusters'].get(section, [])
                if clusters:
                    unique_clusters = len(set(clusters))
                    section_clusters.append(unique_clusters)
                    max_clusters = max(max_clusters, unique_clusters)
                    
                    if unique_clusters == 1:
                        single_cluster_count += 1
            
            if section_clusters:
                cluster_stats['avg_clusters_per_section'][section] = np.mean(section_clusters)
                cluster_stats['max_clusters'][section] = max_clusters
                cluster_stats['single_cluster_percentage'][section] = (single_cluster_count / len(section_clusters)) * 100
        
        # Save stats
        cluster_stats_path = self.output_dir / "cluster_stats.json"
        with open(cluster_stats_path, 'w') as f:
            json.dump(cluster_stats, f, indent=2)
        
        # Create bar chart of average clusters
        plt.figure(figsize=(12, 6))
        sections = list(cluster_stats['avg_clusters_per_section'].keys())
        avg_clusters = [cluster_stats['avg_clusters_per_section'][s] for s in sections]
        
        plt.bar(sections, avg_clusters)
        plt.title('Average Number of Semantic Clusters per Section')
        plt.xlabel('Section')
        plt.ylabel('Average Clusters')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "avg_clusters_by_section.png", dpi=300)
        plt.close()
        
        return cluster_stats
    
    def create_correlation_matrix(self):
        """Create and visualize correlation matrix between section entropies."""
        # Prepare correlation data
        corr_columns = [f'{section}_entropy' for section in self.sections]
        corr_data = self.contract_df[corr_columns].corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Between Section Entropies')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / "entropy_correlation_matrix.png", dpi=300)
        plt.close()
        
        return corr_data
    
    def run_full_analysis(self):
        """Run all analyses and generate a comprehensive report."""
        logging.info("Starting comprehensive analysis of entropy results")
        
        # Generate all analyses
        summary_stats = self.generate_summary_stats()
        self.plot_overall_entropy_distribution()
        self.plot_section_entropy_comparison()
        self.plot_heatmap()
        high_entropy_contracts = self.identify_high_entropy_contracts()
        cluster_stats = self.analyze_cluster_patterns()
        correlation_matrix = self.create_correlation_matrix()
        
        # Generate a text report
        report = ["# Smart Contract Intent Entropy Analysis", ""]
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("### Overall Entropy")
        report.append(f"- Mean: {summary_stats['overall']['mean']:.4f}")
        report.append(f"- Median: {summary_stats['overall']['median']:.4f}")
        report.append(f"- Standard Deviation: {summary_stats['overall']['std']:.4f}")
        report.append(f"- Min: {summary_stats['overall']['min']:.4f}")
        report.append(f"- Max: {summary_stats['overall']['max']:.4f}")
        report.append("")
        
        # Section statistics
        report.append("### Section Entropy Statistics")
        for section in self.sections:
            if section in summary_stats:
                report.append(f"#### {section}")
                report.append(f"- Mean: {summary_stats[section]['mean']:.4f}")
                report.append(f"- Median: {summary_stats[section]['median']:.4f}")
                report.append(f"- Standard Deviation: {summary_stats[section]['std']:.4f}")
                report.append(f"- Min: {summary_stats[section]['min']:.4f}")
                report.append(f"- Max: {summary_stats[section]['max']:.4f}")
                report.append("")
        
        # High entropy contracts
        report.append("## High Entropy Contracts")
        report.append(f"Contracts with entropy in the top 10%:")
        for idx, row in high_entropy_contracts.head(10).iterrows():
            report.append(f"- {row['contract']}: {row['overall_entropy']:.4f}")
        report.append("")
        
        # Cluster analysis
        report.append("## Semantic Cluster Analysis")
        report.append("### Average Clusters per Section")
        for section, avg_clusters in cluster_stats['avg_clusters_per_section'].items():
            report.append(f"- {section}: {avg_clusters:.2f}")
        report.append("")
        
        report.append("### Single Cluster Percentage")
        report.append("Percentage of contracts with only one semantic cluster per section:")
        for section, percentage in cluster_stats['single_cluster_percentage'].items():
            report.append(f"- {section}: {percentage:.2f}%")
        report.append("")
        
        # Correlation analysis
        report.append("## Section Entropy Correlations")
        report.append("Strongest correlations between section entropies:")
        
        # Find top 3 correlations
        corr_pairs = []
        for i, section1 in enumerate(self.sections):
            for j, section2 in enumerate(self.sections):
                if i < j:  # Avoid duplicates and self-correlations
                    corr = correlation_matrix.loc[f"{section1}_entropy", f"{section2}_entropy"]
                    corr_pairs.append((section1, section2, corr))
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for section1, section2, corr in corr_pairs[:3]:
            report.append(f"- {section1} and {section2}: {corr:.4f}")
        report.append("")
        
        # Save the report
        report_path = self.output_dir / "entropy_analysis_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logging.info(f"Comprehensive analysis completed. Report saved to {report_path}")
        return {
            "summary_stats": summary_stats,
            "high_entropy_contracts": high_entropy_contracts,
            "cluster_stats": cluster_stats,
            "correlation_matrix": correlation_matrix
        } 