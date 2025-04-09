"""计算语义熵的模块。"""
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

# 尝试导入setup_logger，如果不存在则创建一个简单版本

from sc_analyzer.utils import setup_logger
from sc_analyzer.models import get_model


class EntailmentModel:
    """语义蕴含检测模型。"""
    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        self.model_name = model_name
        self.model_provider = model_provider
        self.entailment_model = get_model(model_type="api", model_name=model_name, model_provider=model_provider)
    

    ## 构建不同的prompt
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
    """计算不同意图分析测试的语义熵。"""
    
    def __init__(
        self,
        results: AnalysisResults,
        entailment_model: EntailmentModel,
    ):
        """
        初始化语义熵计算器。
        
        参数:
            results: 步骤3分析结果字典
            model_name: 语义蕴含检测模型
        """
        self.results = results
        self.entailment_model = entailment_model
        self.logger = logging.getLogger("SE-cal")
        self.logger.setLevel(logging.DEBUG)

    # todo 不同模型做entailment
    def are_equivalent_llm_api(self, context: str,text1: str, text2: str) -> bool:
        """
        使用LLM API判断两段文本是否语义等价。
        
        参数:
            context: 上下文
            text1: 第一段文本
            text2: 第二段文本
            
        返回:
            如果文本语义等价则为True，否则为False
        """
        # 基本检查
        if not context or not text1 or not text2:
            self.logger.error("上下文或文本为空，无法进行语义等价判断")
            return False
        
        result = self.entailment_model.predict(context, text1, text2).lower()
        self.logger.debug(f"语义等价判断结果: {result}")


        ## todo 怎么样处理 neutral 的情况
        if 'entailment' in result:
            return True
        elif 'contradiction' in result:
            return False
        else:
            self.logger.warning(f"语义等价判断结果异常: {result}")
            return False

    
    def get_semantic_ids(self,context:str, texts: List[str]) -> List[int]:
        """
        将文本分组为语义簇并分配ID。
    
        参数:
            texts: 要聚类的文本段列表
        
        返回:
            每个文本的簇ID列表
        """
        if not texts or all(not text for text in texts):
            return []
            
        # 过滤掉空文本
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
                        # 处理等价情况
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
            self.logger.debug(f"输入文本: {text}")
        
        for cluster in clusters:
            self.logger.debug(f"cluster: {cluster}")
        self.logger.debug(f"聚类结果: {cluster_ids}")

        return cluster_ids
     
    def calculate_cluster_entropy(self, cluster_ids: List[int], log_probs: List[float]) -> float:
        """
        根据聚类ID和对数概率计算语义熵（使用Rao方法）。
        
        参数:
            cluster_ids: 聚类ID列表
            log_probs: 对应的对数概率列表
            
        返回:
            计算得到的语义熵
        """
        if not cluster_ids or not log_probs or len(cluster_ids) != len(log_probs):
            self.logger.error("聚类ID或对数概率列表无效，无法计算语义熵. cluster_ids: {cluster_ids}, log_probs: {log_probs}")
            raise ValueError("聚类ID或对数概率列表无效，无法计算语义熵")
            
        avg_log_probs = [sum(log_prob) / len(log_prob) for log_prob in log_probs]


        
        # 筛选有效的log_probs和对应的聚类ID
        valid_data = []
        for cid, lp in zip(cluster_ids, avg_log_probs):
            if lp is not None and not math.isnan(lp) and not math.isinf(lp):
                valid_data.append((cid, lp))
            else:
                self.logger.warning(f"无效数据: cid: {cid}, lp: {lp}")
        
        if not valid_data:
            self.logger.error("没有有效的数据，无法计算语义熵, cluster_ids: {cluster_ids}, log_probs: {log_probs}")
            raise ValueError("没有有效的数据，无法计算语义熵")
            
        # 提取有效的聚类ID和对数概率
        valid_cluster_ids = [cid for cid, _ in valid_data]
        valid_log_probs = [lp for _, lp in valid_data]
        
        # 获取唯一的聚类ID并按顺序排序
        unique_ids = sorted(list(set(valid_cluster_ids)))
        if unique_ids != list(range(len(unique_ids))):
            self.logger.warning("聚类ID不连续，将重新映射")
            # 重新映射聚类ID为连续的整数
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            valid_cluster_ids = [id_mapping[cid] for cid in valid_cluster_ids]
            unique_ids = list(range(len(unique_ids)))
        
        # 计算每个语义ID的logsumexp
        log_likelihood_per_semantic_id = []
        for uid in unique_ids:
            # 找到属于当前uid的所有位置
            id_indices = [pos for pos, x in enumerate(valid_cluster_ids) if x == uid]
            # 获取这些位置的对数概率
            id_log_likelihoods = [valid_log_probs[i] for i in id_indices]
            
            # 计算归一化因子
  
            # log( sum(p) )
            total_log_prob = math.log(sum(math.exp(lp) for lp in valid_log_probs))
            
            # 归一化对数概率
            # [ log( p/sum(p) ) ]
            log_lik_norm = [lp - total_log_prob for lp in id_log_likelihoods]
            
            # 计算logsumexp
            #log(  (p1+ p3)/sum(p) )
            logsumexp_value = math.log(sum(math.exp(lp) for lp in log_lik_norm))
            
            log_likelihood_per_semantic_id.append( round(logsumexp_value, 8))
        
        # 使用Rao方法计算熵
        entropy = -sum(math.exp(log_p) * log_p for log_p in log_likelihood_per_semantic_id)
        entropy = round(entropy, 6)

        sum_of_rao_p = sum( [math.exp(log_p) for log_p in log_likelihood_per_semantic_id])
        sum_of_rao_p = round(sum_of_rao_p, 6)

        self.logger.debug(f"聚类ID: {valid_cluster_ids}")
        self.logger.debug(f"原始对数概率: {valid_log_probs}")
        self.logger.debug(f"归一化后的对数概率: {[lp - total_log_prob for lp in valid_log_probs]}")
        self.logger.debug(f"每个语义ID的聚合对数概率: {log_likelihood_per_semantic_id}")
        self.logger.debug(f"计算得到的语义熵: {entropy}")
        self.logger.debug(f"sum of rao p: {sum_of_rao_p}")

        
        if sum_of_rao_p!= 1:
            self.logger.error(f"sth wrong when cal entropy , cluster_ids:{cluster_ids}, entropy:{entropy}")
            raise ValueError(f"sth wrong when cal entropy , cluster_ids:{cluster_ids},  entropy:{entropy}")
        
        return entropy
    
