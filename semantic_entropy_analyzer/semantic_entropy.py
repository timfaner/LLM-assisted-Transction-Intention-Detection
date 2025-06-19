"""Calculate semantic entropytexttexttext """
import os
import logging
import time
import pickle
import json
from pathlib import Path
import math
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np


from sc_analyzer.data_types import (
    AnalysisResults, ContractData, IntentData, 
    EntropyResults, QuestionEntropy, ContractEntropy,
    IntentEntropy, EntropySummary,AnswerWithArgLogprobList
)

# texttexttexttextsetup_logger texttexttexttexttexttexttexttexttexttexttexttexttexttext

from sc_analyzer.utils import setup_logger
from sc_analyzer.models import get_model


class EntailmentModel:
    """texttexttexttexttexttexttexttext """
    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        self.model_name = model_name
        self.model_provider = model_provider
        self.entailment_model = get_model(model_type="api", model_name=model_name, model_provider=model_provider)
    

    ## texttexttexttexttextprompt
    def equivalence_prompt(self,question:str, text1:str, text2:str):
        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Only Respond with entailment, contradiction, or neutral, no need to explain."""
        return prompt

    def predict(self, question:str, text1:str, text2:str, temperature=0.1):
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.equivalence_prompt(question, text1, text2)}
        ]
        return self.entailment_model.get_response(messages, temperature=temperature) 

class SemanticEntropyCalculator:
    """texttexttexttexttexttexttexttexttexttexttexttexttexttext """
    
    def __init__(
        self,
        results: AnalysisResults,
        entailment_model: EntailmentModel,
    ):
        """
        texttexttexttexttexttexttexttexttext 
        
        texttext:
            results: texttext3texttexttexttexttexttext
            model_name: texttexttexttexttexttexttexttext
        """
        self.results = results
        self.entailment_model = entailment_model
        self.logger = logging.getLogger("SE-cal")
        self.logger.setLevel(logging.DEBUG)

    # todo texttexttexttexttextentailment
    def are_equivalent_llm_api(self, context: str,text1: str, text2: str) -> bool:
        """
        texttextLLM APItexttexttexttexttexttexttexttexttexttexttexttext 
        
        texttext:
            context: texttexttext
            text1: texttexttexttexttext
            text2: texttexttexttexttext
            
        texttext:
            texttexttexttexttexttexttexttexttexttextTrue texttexttextFalse
        """
        # texttexttexttext
        if not context or not text1 or not text2:
            self.logger.error("texttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttext")
            return False
        
        result = self.entailment_model.predict(context, text1, text2).lower()
        self.logger.debug(f"texttexttexttexttexttexttexttext: {result}")


        ## todo texttexttexttexttext neutral texttexttext
        if 'entailment' in result:
            return True
        elif 'contradiction' in result:
            return False
        else:
            self.logger.warning(f"texttexttexttexttexttexttexttexttexttext: {result}")
            return False

    
    def get_semantic_ids(self,context:str, texts: List[str]) -> List[int]:
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
                    if self.are_equivalent_llm_api(context,text, cluster_text):
                        # texttexttexttexttexttext
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

        for text in texts:
            self.logger.debug(f"texttexttexttext: {text}")
        
        for cluster in clusters:
            self.logger.debug(f"cluster: {cluster}")
        self.logger.debug(f"texttexttexttext: {cluster_ids}")

        return cluster_ids
     
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
            self.logger.error("texttextIDtexttexttexttexttexttexttexttexttext texttextCalculate semantic entropy. cluster_ids: {cluster_ids}, log_probs: {log_probs}")
            raise ValueError("texttextIDtexttexttexttexttexttexttexttexttext texttextCalculate semantic entropy")
            
        avg_log_probs = [sum(log_prob) / len(log_prob) for log_prob in log_probs]


        
        # texttexttexttexttextlog_probstexttexttexttexttexttextID
        valid_data = []
        for cid, lp in zip(cluster_ids, avg_log_probs):
            if lp is not None and not math.isnan(lp) and not math.isinf(lp):
                valid_data.append((cid, lp))
            else:
                self.logger.warning(f"texttexttexttext: cid: {cid}, lp: {lp}")
        
        if not valid_data:
            self.logger.error("texttexttexttexttexttexttext texttextCalculate semantic entropy, cluster_ids: {cluster_ids}, log_probs: {log_probs}")
            raise ValueError("texttexttexttexttexttexttext texttextCalculate semantic entropy")
            
        # texttexttexttexttexttexttextIDtexttexttexttexttext
        valid_cluster_ids = [cid for cid, _ in valid_data]
        valid_log_probs = [lp for _, lp in valid_data]
        
        # texttexttexttexttexttexttextIDtexttexttexttexttexttext
        unique_ids = sorted(list(set(valid_cluster_ids)))
        if unique_ids != list(range(len(unique_ids))):
            self.logger.warning("texttextIDtexttexttext texttexttexttexttext")
            # texttexttexttexttexttextIDtexttexttexttexttexttext
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            valid_cluster_ids = [id_mapping[cid] for cid in valid_cluster_ids]
            unique_ids = list(range(len(unique_ids)))
        
        # texttexttexttexttexttextIDtextlogsumexp
        log_likelihood_per_semantic_id = []
        for uid in unique_ids:
            # texttexttexttexttexttextuidtexttexttexttexttext
            id_indices = [pos for pos, x in enumerate(valid_cluster_ids) if x == uid]
            # texttexttexttexttexttexttexttexttexttexttext
            id_log_likelihoods = [valid_log_probs[i] for i in id_indices]
            
            # texttexttexttexttexttexttext
  
            # log( sum(p) )
            total_log_prob = math.log(sum(math.exp(lp) for lp in valid_log_probs))
            
            # texttexttexttexttexttexttext
            # [ log( p/sum(p) ) ]
            log_lik_norm = [lp - total_log_prob for lp in id_log_likelihoods]
            
            # texttextlogsumexp
            #log(  (p1+ p3)/sum(p) )
            logsumexp_value = math.log(sum(math.exp(lp) for lp in log_lik_norm))
            
            log_likelihood_per_semantic_id.append( round(logsumexp_value, 8))
        
        # texttextRaotexttexttexttexttext
        entropy = -sum(math.exp(log_p) * log_p for log_p in log_likelihood_per_semantic_id)
        entropy = round(entropy, 6)

        sum_of_rao_p = sum( [math.exp(log_p) for log_p in log_likelihood_per_semantic_id])
        sum_of_rao_p = round(sum_of_rao_p, 6)

        self.logger.debug(f"texttextID: {valid_cluster_ids}")
        self.logger.debug(f"texttexttexttexttexttext: {valid_log_probs}")
        self.logger.debug(f"texttexttexttexttexttexttexttexttext: {[lp - total_log_prob for lp in valid_log_probs]}")
        self.logger.debug(f"texttexttexttextIDtexttexttexttexttexttexttext: {log_likelihood_per_semantic_id}")
        self.logger.debug(f"texttexttexttexttexttexttexttext: {entropy}")
        self.logger.debug(f"sum of rao p: {sum_of_rao_p}")

        
        if sum_of_rao_p!= 1:
            self.logger.error(f"sth wrong when cal entropy , cluster_ids:{cluster_ids}, entropy:{entropy}")
            raise ValueError(f"sth wrong when cal entropy , cluster_ids:{cluster_ids},  entropy:{entropy}")
        
        return entropy
    
