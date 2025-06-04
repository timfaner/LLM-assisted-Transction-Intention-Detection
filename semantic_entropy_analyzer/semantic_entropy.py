"""texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """

import os
import logging
import pickle
import argparse
from pathlib import Path
import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import json
import time
import torch

# texttexttexttextsetup_logger texttexttexttexttexttexttexttexttexttexttexttexttexttext
try:
    from sc_analyzer.utils import setup_logger
except ImportError:
    def setup_logger(level=logging.INFO):
        """Set up loggingtexttext"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


class SemanticEntropyCalculator:
    """texttexttexttexttexttexttexttexttexttexttexttexttexttext """
    
    def __init__(
        self, 
        results_path: str, 
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_api_for_equivalence: bool = True
    ):
        """
        texttexttexttexttexttexttexttexttext 
        
        texttext:
            results_path: texttextPath to the intention analysis pickle result file
            output_dir: texttexttexttexttexttexttexttext
            device: texttexttexttexttexttexttext('cuda'text'cpu')
            use_api_for_equivalence: texttexttexttextLLM APItexttexttexttexttexttexttexttext
        """
        self.results_path = Path(results_path)
        self.use_api_for_equivalence = use_api_for_equivalence
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_path.parent / "entropy_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load results
        with open(self.results_path, "rb") as f:
            self.results = pickle.load(f)
            
        logging.info(f"texttexttexttexttexttexttext: {self.results_path}")
        
        # APItexttexttexttext texttextuse_api_for_equivalencetextTruetexttexttext 
        self.api_client = None
        if self.use_api_for_equivalence:
            try:
                import openai
                from openai import OpenAI
                
                # texttextAPItexttext
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logging.warning("texttexttextOpenAI APItexttext texttexttexttexttexttexttexttexttexttexttexttexttext")
                    self.use_api_for_equivalence = False
                else:
                    self.api_client = OpenAI(api_key=api_key)
                    logging.info("texttexttexttextOpenAI APItexttexttext")
            except ImportError:
                logging.warning("texttexttextopenaitext texttexttexttexttexttexttexttexttexttexttexttexttext")
                self.use_api_for_equivalence = False
                
        # texttexttexttext
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"texttexttexttext: {self.device}")
    
    def are_equivalent_llm_api(self, text1: str, text2: str) -> bool:
        """
        texttextLLM APItexttexttexttexttexttexttexttexttexttexttexttext 
        
        texttext:
            text1: texttexttexttexttext
            text2: texttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttexttexttexttextTrue texttexttextFalse
        """
        # texttexttexttext
        if not text1 or not text2:
            return False
        
        if text1 == text2:
            return True
        
        if not self.api_client:
            logging.warning("APItexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttext")
            return text1.lower() == text2.lower()
        
        # texttexttexttexttext
        prompt = f"""
        texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttext  
        
        texttext1:
        "{text1}"
        
        texttext2:
        "{text2}"
        
        texttexttexttext"text"text"text" texttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext texttexttexttext"text" 
        """
        
        try:
            response = self.api_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            
            # texttexttexttext
            if "text" in answer or "texttext" in answer or "texttext" in answer:
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"APItexttexttexttext: {e}")
            # texttexttexttexttexttexttext
            return text1.lower() == text2.lower()
    
    def get_semantic_ids(self, texts: List[str]) -> List[int]:
        """
        texttexttexttexttexttexttexttexttexttexttexttextID 
    
        texttext:
            texts: texttexttexttexttexttexttexttexttext
        
        texttext:
            texttexttexttexttexttextIDtexttext
        """
        if not texts or all(not text for text in texts):
            return []
            
        # texttexttexttexttexttext
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
                    if self.use_api_for_equivalence:
                        if self.are_equivalent_llm_api(text, cluster_text):
                            # texttexttexttexttexttext
                            clusters[cluster_id].append(text)
                            cluster_ids[idx] = cluster_id
                            break
                    else:
                        # texttexttexttext texttexttexttextAPItexttext
                        if text.lower() == cluster_text.lower():
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
    
    def calculate_entropy_from_logprobs(self, log_probs: List[float]) -> float:
        """
        texttexttexttexttexttexttexttexttexttext 
        
        texttext:
            log_probs: texttexttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttext
        """
        if not log_probs or len(log_probs) <= 1:
            return 0.0
            
        # texttexttexttexttext
        valid_log_probs = [lp for lp in log_probs if lp is not None and not math.isnan(lp) and not math.isinf(lp)]
        if not valid_log_probs:
            return 0.0
            
        # texttexttexttexttexttexttexttexttext
        max_log_prob = max(valid_log_probs)
        probs = [math.exp(lp - max_log_prob) for lp in valid_log_probs]
        
        # texttexttexttexttext
        total_prob = sum(probs)
        if total_prob <= 0:
            return 0.0
            
        normalized_probs = [p / total_prob for p in probs]
        
        # texttexttext
        entropy = -sum(p * math.log2(p) for p in normalized_probs if p > 0)
        return entropy
    
    def logsumexp(self, log_probs: List[float]) -> float:
        """
        texttextlog(sum(exp(x))) texttexttexttexttexttexttexttext 
        
        texttext:
            log_probs: texttexttexttexttexttext
            
        texttext:
            log(sum(exp(x)))texttext
        """
        if not log_probs:
            return float('-inf')
            
        max_log_prob = max(log_probs)
        sum_exp = sum(math.exp(lp - max_log_prob) for lp in log_probs)
        
        return max_log_prob + math.log(sum_exp)
    
    def calculate_cluster_entropy(self, cluster_ids: List[int], log_probs: List[float]) -> float:
        """
        texttexttexttextIDtexttexttexttexttextCalculate semantic entropy texttextRaotexttext  
        
        texttext:
            cluster_ids: texttextIDtexttext
            log_probs: texttexttexttexttexttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttexttext
        """
        if not cluster_ids or not log_probs or len(cluster_ids) != len(log_probs):
            return 0.0
            
        # texttexttexttexttextlog_probstexttexttexttextIDtexttext
        valid_data = [(cid, lp) for cid, lp in zip(cluster_ids, log_probs) 
                     if lp is not None and not math.isnan(lp) and not math.isinf(lp)]
        
        if not valid_data:
            return 0.0
            
        # texttexttextIDtexttexttexttexttexttext
        cluster_log_probs = defaultdict(list)
        for cluster_id, log_prob in valid_data:
            cluster_log_probs[cluster_id].append(log_prob)
        
        # texttexttexttexttexttexttextlogsumexptexttexttexttexttexttext
        aggregated_log_probs = []
        for log_probs_list in cluster_log_probs.values():
            if log_probs_list:
                aggregated_log_probs.append(self.logsumexp(log_probs_list))
        
        # texttexttexttexttexttexttexttexttexttexttexttexttexttext
        if aggregated_log_probs:
            # texttextlogsumexptexttextlogtexttexttexttexttexttext
            total_log_prob = self.logsumexp(aggregated_log_probs)
            normalized_log_probs = [lp - total_log_prob for lp in aggregated_log_probs]
            
            # texttextRaotexttexttexttexttexttexttext: -sum(exp(log_p) * log_p)
            entropy = -sum(math.exp(log_p) * log_p for log_p in normalized_log_probs)
            return entropy
        
        return 0.0
    
    def calculate_question_entropy(self, question_data: Dict) -> float:
        """
        texttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttext 
        
        texttext:
            question_data: texttexttexttexttexttexttexttexttexttexttexttext
            
        texttext:
            texttexttexttexttexttext
        """
        # texttexttexttexttexttexttexttexttexttexttext
        answers = []
        log_probs = []
        
        for answer in question_data.get('answers', []):
            answer_text = answer.get('answer', '')
            logprob = answer.get('avg_logprob')
            
            if answer_text and logprob is not None:
                answers.append(answer_text)
                log_probs.append(logprob)
        
        if not answers or not log_probs:
            return 0.0
        
        # texttexttexttexttexttexttexttexttext
        cluster_ids = self.get_semantic_ids(answers)
        
        # texttexttexttexttexttexttexttextCalculate semantic entropy
        entropy = self.calculate_cluster_entropy(cluster_ids, log_probs)
        
        # texttexttexttexttexttext
        logging.debug(f"texttext: {question_data.get('question', 'texttexttexttext')}")
        logging.debug(f"texttexttexttext: {len(answers)}")
        logging.debug(f"texttexttexttext: {cluster_ids}")
        logging.debug(f"texttexttexttext: {log_probs}")
        logging.debug(f"texttexttexttexttexttexttexttext: {entropy}")
        
        return entropy
    
    def calculate_section_entropy(self, section_data: Dict) -> float:
        """
        texttexttexttexttexttexttexttexttext texttexttexttext texttexttexttexttext texttexttexttext 
        
        texttext:
            section_data: texttexttexttexttexttexttexttexttexttexttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttext texttexttexttexttexttexttexttexttext 
        """
        question_entropies = []
        
        # texttexttexttexttexttexttexttext
        for question in section_data.get('questions', []):
            question_entropy = self.calculate_question_entropy(question)
            question_entropies.append(question_entropy)
        
        # texttexttexttexttext
        if not question_entropies:
            return 0.0
        
        return sum(question_entropies) / len(question_entropies)
    
    def calculate_intent_entropy(self, intent_data: Dict) -> Dict:
        """
        texttexttexttexttexttexttexttexttexttext 
        
        texttext:
            intent_data: texttexttexttexttexttexttexttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttexttext
        """
        # texttextintent_datatexttexttexttexttexttexttext
        if not isinstance(intent_data, dict):
            logging.warning(f"texttexttexttexttexttexttexttext: {type(intent_data)}, texttexttexttext: dict")
            return {
                'intent_id': 'unknown',
                'overall_entropy': 0.0,
                'section_entropies': {}
            }
            
        section_entropies = {}
        
        # texttextintent_datatexttexttexttexttextsectionstexttext
        sections = intent_data.get('sections', [])
        if not sections:
            logging.warning(f"texttexttexttexttexttexttextsectionstexttexttexttexttext")
            
        # texttexttexttexttexttexttexttext
        for section in sections:
            if not isinstance(section, dict):
                logging.warning(f"texttexttexttexttexttexttexttext: {type(section)}, texttexttexttext: dict")
                continue
                
            section_name = section.get('section_name', 'unknown')
            section_entropy = self.calculate_section_entropy(section)
            section_entropies[section_name] = section_entropy
        
        # texttexttexttexttexttexttext
        if not section_entropies:
            avg_section_entropy = 0.0
        else:
            avg_section_entropy = sum(section_entropies.values()) / len(section_entropies)
        
        return {
            'intent_id': intent_data.get('intent_id', 'unknown'),
            'overall_entropy': avg_section_entropy,
            'section_entropies': section_entropies
        }
    
    def calculate_contract_entropies(self, contract_data: Dict) -> Dict:
        """
        texttexttexttexttexttexttexttexttexttexttexttexttexttexttext 
        
        texttext:
            contract_data: texttexttexttexttexttext texttexttest_resultstexttext
            
        texttext:
            texttexttexttexttexttexttexttext
        """
        intent_entropies = []
        
        # texttextcontract_datatexttexttexttexttext
        if not isinstance(contract_data, dict):
            logging.warning(f"texttexttexttexttexttexttexttext: {type(contract_data)}, texttexttexttext: dict")
            return {
                'contract_path': 'unknown',
                'avg_overall_entropy': 0.0,
                'intent_entropies': []
            }
            
        # texttexttest_resultstexttext texttexttexttexttexttext
        test_results = contract_data.get('test_results', [])
        if not test_results:
            logging.warning(f"texttexttexttexttexttexttexttest_resultstexttexttexttexttext")
            
        # texttexttexttexttexttexttexttexttexttext
        for intent_data in test_results:
            intent_entropy = self.calculate_intent_entropy(intent_data)
            intent_entropies.append(intent_entropy)
        
        # texttexttexttexttext
        overall_entropies = [e['overall_entropy'] for e in intent_entropies]
        avg_overall_entropy = sum(overall_entropies) / len(overall_entropies) if overall_entropies else 0.0
        
        # texttextrelative_pathtexttextcontract_path
        contract_path = contract_data.get('relative_path', 'unknown')
        
        return {
            'contract_path': contract_path,
            'avg_overall_entropy': avg_overall_entropy,
            'intent_entropies': intent_entropies
        }
    
    def calculate_all_entropies(self) -> Dict:
        """
        texttexttexttexttexttexttexttexttexttext 
        
        texttext:
            texttexttexttexttexttexttexttexttexttext
        """
        contract_results = self.results.get('contract_intents', {})
        
        entropy_results = {
            'contracts': [],
            'summary': {
                'num_contracts': len(contract_results),
                'avg_overall_entropy': 0.0
            }
        }
        
        # texttexttexttexttexttexttexttext
        total_entropy = 0.0
        for contract_path, contract_data in contract_results.items():
            logging.info(f"texttexttexttexttexttexttexttext: {contract_path}")
            contract_entropy = self.calculate_contract_entropies(contract_data)
            entropy_results['contracts'].append(contract_entropy)
            total_entropy += contract_entropy['avg_overall_entropy']
        
        # texttexttexttexttext
        if entropy_results['contracts']:
            entropy_results['summary']['avg_overall_entropy'] = total_entropy / len(entropy_results['contracts'])
        
        return entropy_results
    
    def calculate_entropies(self) -> Dict:
        """
        Calculate semantic entropytexttexttexttexttext 
        
        texttext:
            texttexttexttexttext
        """
        logging.info("texttextCalculate semantic entropy...")
        start_time = time.time()
        
        # texttexttexttexttexttext
        entropy_results = self.calculate_all_entropies()
        
        # texttexttexttexttexttext
        end_time = time.time()
        entropy_results["summary"]["time_taken"] = end_time - start_time
        
        # texttexttexttext
        results_path = self.output_dir / "entropy_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(entropy_results, f)
        
        # texttextJSONtexttexttexttexttext
        summary_path = self.output_dir / "entropy_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(entropy_results["summary"], f, ensure_ascii=False, indent=2)
        
        logging.info(f"texttexttexttexttexttexttext texttext: {end_time - start_time:.2f}text")
        logging.info(f"texttexttexttexttexttext: {results_path}")
        logging.info(f"texttexttexttexttexttext: {summary_path}")
        logging.info(f"texttexttexttexttexttexttext: {entropy_results['summary']['avg_overall_entropy']:.4f}")
        
        return entropy_results


def main():
    """Run semantic entropy calculation from command line."""
    parser = argparse.ArgumentParser(description="texttexttexttexttexttexttexttexttexttexttexttext")
    
    parser.add_argument("--results_path", type=str, required=True, 
                        help="texttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="texttexttexttexttexttexttexttext")
    parser.add_argument("--device", type=str, default=None,
                        help="texttexttexttext (cuda text cpu)")
    parser.add_argument("--no_api", action="store_true",
                        help="texttexttextAPItexttexttexttexttexttexttexttext texttexttexttextAPI ")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="texttexttexttext")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize and run entropy calculator
        calculator = SemanticEntropyCalculator(
            results_path=args.results_path,
            output_dir=args.output_dir,
            device=args.device,
            use_api_for_equivalence=not args.no_api
        )
        
        # Calculate entropies
        calculator.calculate_entropies()
        
    except Exception as e:
        logger.exception(f"texttexttexttexttexttexttext: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 