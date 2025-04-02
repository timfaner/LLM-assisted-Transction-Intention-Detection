"""Module for calculating semantic entropy of smart contract intent analyses."""

import os
import logging
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import wandb
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import json
import time
import math

from sc_analyzer.utils import setup_logger


class SemanticEntropyCalculator:
    """Calculate semantic entropy across different intent analysis tests."""
    
    def __init__(
        self, 
        results_path: str, 
        output_dir: Optional[str] = None,
        entailment_model: str = "cross-encoder/nli-deberta-v3-base",
        entailment_type: str = "nli",
        device: Optional[str] = None
    ):
        """
        Initialize the semantic entropy calculator.
        
        Args:
            results_path: Path to the pickle file with intent analysis results
            output_dir: Directory to save entropy results
            entailment_model: Name or path of the entailment model to use
            entailment_type: Type of entailment model ('nli' or 'stsb')
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.results_path = Path(results_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_path.parent / "entropy_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load results
        with open(self.results_path, "rb") as f:
            self.results = pickle.load(f)
            
        # Set up entailment model
        self.entailment_model_name = entailment_model
        self.entailment_type = entailment_type
        logging.info(f"Loading entailment model {entailment_model} on {self.device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(entailment_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(entailment_model)
        
        # Set threshold based on model type
        if entailment_type == "nli":
            # For NLI models, we'll use the probability of entailment or contradiction
            self.threshold = 0.5  # Probability threshold
        elif entailment_type == "stsb":
            # For STS-B models, we'll use the similarity score
            self.threshold = 0.8  # Similarity threshold
        else:
            raise ValueError(f"Unknown entailment type: {entailment_type}")
            
        # Initialize sections for intent parsing
        self.sections = [
            "function_intent",
            "constraints",
            "parameters",
            "execution_flow",
            "key_events",
            "state_values"
        ]
    
    def parse_intent_sections(self, intent_text: str) -> Dict[str, str]:
        """
        Parse intent text into sections.
        
        Args:
            intent_text: The full intent text
            
        Returns:
            Dictionary with sections as keys and the section text as values
        """
        # Initialize sections
        sections = {section: "" for section in self.sections}
        
        # Split intent into lines
        lines = intent_text.strip().split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            lower_line = line.lower()
            for section in self.sections:
                # Convert section name from snake_case to Title Case with spaces
                section_title = " ".join(word.capitalize() for word in section.split("_"))
                if lower_line.startswith(section_title.lower()) and ":" in lower_line:
                    current_section = section
                    # Extract text after the colon if there is any
                    if ":" in line:
                        sections[current_section] += line.split(":", 1)[1].strip() + " "
                    break
            else:
                # If not a section header and we have a current section, add to that section
                if current_section:
                    sections[current_section] += line + " "
        
        # Clean up sections
        for section in sections:
            sections[section] = sections[section].strip()
            
        return sections
    
    def are_equivalent(self, text1: str, text2: str) -> bool:
        """
        Check if two text segments are semantically equivalent using the entailment model.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            True if texts are semantically equivalent, False otherwise
        """
        if not text1 or not text2:
            return False
            
        if text1 == text2:
            return True
            
        # Encode the texts
        inputs = self.tokenizer(
            [text1, text2],
            [text2, text1],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process outputs based on model type
        if self.entailment_type == "nli":
            # NLI models typically have 3 outputs: entailment, neutral, contradiction
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Check if either direction shows entailment (idx 0 usually = entailment)
            entailment_prob_1to2 = probs[0, 0].item()
            entailment_prob_2to1 = probs[1, 0].item()
            
            # Bidirectional entailment indicates equivalence
            return entailment_prob_1to2 > self.threshold and entailment_prob_2to1 > self.threshold
            
        elif self.entailment_type == "stsb":
            # STS-B models output a single similarity score
            similarity = torch.sigmoid(outputs.logits).squeeze().item()
            return similarity > self.threshold
            
        return False
    
    def get_semantic_ids(self, texts: List[str]) -> List[int]:
        """
        Group texts into semantic clusters and assign IDs.
        
        Args:
            texts: List of text segments to cluster
            
        Returns:
            List of cluster IDs for each text
        """
        if not texts or all(not text for text in texts):
            return []
            
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        
        if not valid_texts:
            return [0] * len(texts)
            
        # Initialize clusters
        clusters = []
        cluster_ids = [-1] * len(texts)
        
        # Assign texts to clusters
        for idx, text in valid_texts:
            # Try to find a matching cluster
            for cluster_id, cluster_texts in enumerate(clusters):
                # Check if text is equivalent to at least one text in the cluster
                for cluster_text in cluster_texts:
                    if self.are_equivalent(text, cluster_text):
                        clusters[cluster_id].append(text)
                        cluster_ids[idx] = cluster_id
                        break
                if cluster_ids[idx] != -1:
                    break
                    
            # If no matching cluster, create a new one
            if cluster_ids[idx] == -1:
                clusters.append([text])
                cluster_ids[idx] = len(clusters) - 1
        
        # Assign 0 to empty texts
        for i in range(len(texts)):
            if cluster_ids[i] == -1:
                cluster_ids[i] = 0
                
        return cluster_ids
    
    def cluster_assignment_entropy(self, cluster_ids: List[int]) -> float:
        """
        Calculate entropy from cluster assignments.
        
        Args:
            cluster_ids: List of cluster IDs
            
        Returns:
            Entropy value
        """
        if not cluster_ids:
            return 0.0
            
        # Count occurrences of each cluster
        counts = defaultdict(int)
        for cluster_id in cluster_ids:
            counts[cluster_id] += 1
            
        # Calculate probabilities
        total = len(cluster_ids)
        probabilities = [count / total for count in counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in probabilities)
        
        return entropy
    
    def process_contract(self, contract_key: str) -> Dict[str, Any]:
        """
        Process a single contract to calculate entropy for its intent analyses.
        
        Args:
            contract_key: Key of the contract in the results dictionary
            
        Returns:
            Dictionary with entropy results for the contract
        """
        logging.info(f"Processing contract {contract_key}")
        contract_data = self.results["contract_intents"].get(contract_key)
        
        if not contract_data or "test_results" not in contract_data:
            logging.warning(f"No test results found for contract {contract_key}")
            return {}
            
        test_results = contract_data["test_results"]
        file_path = contract_data.get("file_path", "")
        relative_path = contract_data.get("relative_path", "")
        
        # Extract intent texts
        intent_texts = []
        for test in test_results:
            if "intent_text" in test and test["intent_text"]:
                intent_texts.append(test["intent_text"])
                
        num_intent_analyses = len(intent_texts)
        if num_intent_analyses < 2:
            logging.warning(f"Not enough intent analyses for contract {contract_key} (found {num_intent_analyses})")
            return {
                "file_path": file_path,
                "relative_path": relative_path,
                "overall_entropy": 0.0,
                "section_entropies": {},
                "section_clusters": {},
                "num_intent_analyses": num_intent_analyses
            }
            
        # Parse sections from each intent text
        parsed_intents = [self.parse_intent_sections(text) for text in intent_texts]
        
        # Calculate entropy for each section
        section_entropies = {}
        section_clusters = {}
        
        for section in self.sections:
            # Get section texts
            section_texts = [intent.get(section, "") for intent in parsed_intents]
            
            # Skip sections with no content
            if not any(text for text in section_texts):
                continue
                
            # Get semantic clusters
            cluster_ids = self.get_semantic_ids(section_texts)
            section_clusters[section] = cluster_ids
            
            # Calculate entropy
            entropy = self.cluster_assignment_entropy(cluster_ids)
            section_entropies[section] = entropy
            
        # Calculate overall entropy (average of section entropies)
        if section_entropies:
            overall_entropy = sum(section_entropies.values()) / len(section_entropies)
        else:
            overall_entropy = 0.0
            
        return {
            "file_path": file_path,
            "relative_path": relative_path,
            "overall_entropy": overall_entropy,
            "section_entropies": section_entropies,
            "section_clusters": section_clusters,
            "num_intent_analyses": num_intent_analyses
        }
        
    def calculate_entropies(self) -> Dict[str, Any]:
        """
        Calculate entropies for all contracts and save results.
        
        Returns:
            Dictionary with entropy results for all contracts
        """
        logging.info("Starting entropy calculation")
        start_time = time.time()
        
        contracts = self.results["contract_intents"].keys()
        entropy_results = {
            "contracts": {},
            "summary": {
                "total_contracts": 0,
                "contracts_with_multiple_tests": 0,
                "mean_entropy": {},
                "max_entropy": {},
                "min_entropy": {}
            }
        }
        
        # Track section entropies for summary statistics
        section_entropies = defaultdict(list)
        overall_entropies = []
        
        # Process each contract
        for contract_key in contracts:
            contract_result = self.process_contract(contract_key)
            
            if contract_result:
                entropy_results["contracts"][contract_key] = contract_result
                entropy_results["summary"]["total_contracts"] += 1
                
                if contract_result["num_intent_analyses"] >= 2:
                    entropy_results["summary"]["contracts_with_multiple_tests"] += 1
                    
                # Collect entropies for summary
                overall_entropy = contract_result["overall_entropy"]
                overall_entropies.append(overall_entropy)
                
                for section, entropy in contract_result["section_entropies"].items():
                    section_entropies[section].append(entropy)
        
        # Calculate summary statistics
        if overall_entropies:
            entropy_results["summary"]["mean_entropy"]["overall"] = np.mean(overall_entropies)
            entropy_results["summary"]["max_entropy"]["overall"] = max(overall_entropies)
            entropy_results["summary"]["min_entropy"]["overall"] = min(overall_entropies)
            
        for section, entropies in section_entropies.items():
            if entropies:
                entropy_results["summary"]["mean_entropy"][section] = np.mean(entropies)
                entropy_results["summary"]["max_entropy"][section] = max(entropies)
                entropy_results["summary"]["min_entropy"][section] = min(entropies)
        
        # Add time taken
        end_time = time.time()
        entropy_results["summary"]["time_taken"] = end_time - start_time
        
        # Save results
        results_path = self.output_dir / "entropy_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(entropy_results, f)
            
        # Also save a JSON summary
        summary_path = self.output_dir / "entropy_summary.json"
        with open(summary_path, "w") as f:
            # Create a copy of the summary with numpy values converted to native Python
            summary = entropy_results["summary"].copy()
            for key in ["mean_entropy", "max_entropy", "min_entropy"]:
                summary[key] = {k: float(v) for k, v in summary[key].items()}
            json.dump(summary, f, indent=2)
            
        logging.info(f"Entropy calculation completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Results saved to {results_path}")
        logging.info(f"Summary saved to {summary_path}")
        
        return entropy_results


def main():
    """Run semantic entropy calculation from command line."""
    parser = argparse.ArgumentParser(description="Calculate semantic entropy of smart contract intents")
    
    parser.add_argument("--results_path", type=str, required=True, 
                        help="Path to the input results pickle file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save entropy results")
    parser.add_argument("--entailment_model", type=str, default="cross-encoder/nli-deberta-v3-base",
                        help="Entailment model to use")
    parser.add_argument("--entailment_type", type=str, default="nli", choices=["nli", "stsb"],
                        help="Type of entailment model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    parser.add_argument("--wandb_project", type=str, default="smart_contract_intent",
                        help="W&B project name")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="Directory for W&B files")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(log_level=log_level)
    
    # Initialize W&B if available
    use_wandb = False
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            dir=args.wandb_dir,
            config=vars(args),
            mode="disabled" if args.debug else "online"
        )
        use_wandb = True
    except ImportError:
        logging.warning("W&B not available, skipping experiment tracking")
    
    try:
        # Initialize and run entropy calculator
        calculator = SemanticEntropyCalculator(
            results_path=args.results_path,
            output_dir=args.output_dir,
            entailment_model=args.entailment_model,
            entailment_type=args.entailment_type,
            device=args.device
        )
        
        # Calculate entropies
        results = calculator.calculate_entropies()
        
        # Log summary to W&B
        if use_wandb:
            wandb.log(results["summary"])
            
            # Create and log figures
            if len(results["contracts"]) > 0:
                # Log distribution of overall entropy
                overall_entropies = [c["overall_entropy"] for c in results["contracts"].values()]
                wandb.log({"overall_entropy_histogram": wandb.Histogram(overall_entropies)})
                
                # Log section entropies
                for section in calculator.sections:
                    section_values = [c["section_entropies"].get(section, 0) 
                                     for c in results["contracts"].values() 
                                     if section in c["section_entropies"]]
                    if section_values:
                        wandb.log({f"{section}_entropy_histogram": wandb.Histogram(section_values)})
    
    finally:
        # Finish W&B run
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main() 