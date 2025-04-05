"""用于计算智能合约意图分析的语义熵的模块。"""

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

# 尝试导入setup_logger，如果不存在则创建一个简单版本
try:
    from sc_analyzer.utils import setup_logger
except ImportError:
    def setup_logger(level=logging.INFO):
        """设置日志记录"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


class SemanticEntropyCalculator:
    """计算不同意图分析测试的语义熵。"""
    
    def __init__(
        self, 
        results_path: str, 
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_api_for_equivalence: bool = True,
        mode: str = "step3"
    ):
        """
        初始化语义熵计算器。
        
        参数:
            results_path: 包含意图分析结果的pickle文件路径
            output_dir: 保存熵结果的目录
            device: 运行模型的设备('cuda'或'cpu')
            use_api_for_equivalence: 是否使用LLM API进行语义等价判断
            mode: 生成模式 ("step3" 或 "all")
        """
        self.results_path = Path(results_path)
        self.use_api_for_equivalence = use_api_for_equivalence
        self.mode = mode
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_path.parent / "entropy_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load results
        with open(self.results_path, "rb") as f:
            self.results = pickle.load(f)
            
        logging.info(f"已加载结果文件: {self.results_path}")
        logging.info(f"使用模式: {self.mode}")
        
        # API相关设置（仅在use_api_for_equivalence为True时使用）
        self.api_client = None
        if self.use_api_for_equivalence:
            try:
                import openai
                from openai import OpenAI
                
                # 设置API密钥
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logging.warning("未找到OpenAI API密钥，语义等价判断将使用简单比较")
                    self.use_api_for_equivalence = False
                else:
                    self.api_client = OpenAI(api_key=api_key)
                    logging.info("已初始化OpenAI API客户端")
            except ImportError:
                logging.warning("未安装openai库，语义等价判断将使用简单比较")
                self.use_api_for_equivalence = False
                
        # 设置设备
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"使用设备: {self.device}")
    
    def are_equivalent_llm_api(self, text1: str, text2: str) -> bool:
        """
        使用LLM API判断两段文本是否语义等价。
        
        参数:
            text1: 第一段文本
            text2: 第二段文本
            
        返回:
            如果文本语义等价则为True，否则为False
        """
        # 基本检查
        if not text1 or not text2:
            return False
        
        if text1 == text2:
            return True
        
        if not self.api_client:
            logging.warning("API客户端未初始化，无法进行语义等价判断")
            return text1.lower() == text2.lower()
        
        # 构建提示词
        prompt = f"""
        请判断以下两段文本是否表达相同的含义（语义等价）。
        
        文本1:
        "{text1}"
        
        文本2:
        "{text2}"
        
        请只回答"是"或"否"，不要解释理由。如果两段文本表达的核心含义相同，即使用词或结构不同，也应回答"是"。
        """
        
        try:
            response = self.api_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个精确判断文本语义等价的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            
            # 处理回答
            if "是" in answer or "相同" in answer or "等价" in answer:
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"API调用出错: {e}")
            # 降级为简单比较
            return text1.lower() == text2.lower()
    
    def get_semantic_ids(self, texts: List[str]) -> List[int]:
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
                    if self.use_api_for_equivalence:
                        if self.are_equivalent_llm_api(text, cluster_text):
                            # 处理等价情况
                            clusters[cluster_id].append(text)
                            cluster_ids[idx] = cluster_id
                            break
                    else:
                        # 简单比较，如果没有API可用
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
        根据对数概率计算熵值。
        
        参数:
            log_probs: 对数概率列表
            
        返回:
            计算得到的熵值
        """
        if not log_probs or len(log_probs) <= 1:
            return 0.0
            
        # 去除无效值
        valid_log_probs = [lp for lp in log_probs if lp is not None and not math.isnan(lp) and not math.isinf(lp)]
        if not valid_log_probs:
            return 0.0
            
        # 对数概率转换为概率
        max_log_prob = max(valid_log_probs)
        probs = [math.exp(lp - max_log_prob) for lp in valid_log_probs]
        
        # 归一化概率
        total_prob = sum(probs)
        if total_prob <= 0:
            return 0.0
            
        normalized_probs = [p / total_prob for p in probs]
        
        # 计算熵
        entropy = -sum(p * math.log2(p) for p in normalized_probs if p > 0)
        return entropy
    
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
            return 0.0
            
        # 筛选有效的log_probs和对应的聚类ID
        valid_data = [(cid, lp) for cid, lp in zip(cluster_ids, log_probs) 
                     if lp is not None and not math.isnan(lp) and not math.isinf(lp)]
        
        if not valid_data:
            return 0.0
            
        # 提取有效的聚类ID和对数概率
        valid_cluster_ids = [cid for cid, _ in valid_data]
        valid_log_probs = [lp for _, lp in valid_data]
        
        # 获取唯一的聚类ID并按顺序排序
        unique_ids = sorted(list(set(valid_cluster_ids)))
        if unique_ids != list(range(len(unique_ids))):
            logging.warning("聚类ID不连续，将重新映射")
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
            total_log_prob = math.log(sum(math.exp(lp) for lp in valid_log_probs))
            
            # 归一化对数概率
            log_lik_norm = [lp - total_log_prob for lp in id_log_likelihoods]
            
            # 计算logsumexp
            logsumexp_value = math.log(sum(math.exp(lp) for lp in log_lik_norm))
            
            log_likelihood_per_semantic_id.append(logsumexp_value)
        
        # 使用Rao方法计算熵
        entropy = -sum(math.exp(log_p) * log_p for log_p in log_likelihood_per_semantic_id)
        
        # 记录调试信息
        logging.debug(f"聚类ID: {valid_cluster_ids}")
        logging.debug(f"原始对数概率: {valid_log_probs}")
        logging.debug(f"归一化后的对数概率: {[lp - total_log_prob for lp in valid_log_probs]}")
        logging.debug(f"每个语义ID的聚合对数概率: {log_likelihood_per_semantic_id}")
        logging.debug(f"计算得到的语义熵: {entropy}")
        
        return entropy
    
    def calculate_question_entropy(self, question_data: Dict) -> float:
        """
        计算单个问题的语义熵，基于其三个答案的聚类和对数概率。
        
        参数:
            question_data: 包含问题和答案信息的字典
            
        返回:
            问题的语义熵
        """
        # 提取答案文本和对数概率
        answers = []
        log_probs = []
        
        for answer in question_data.get('answers', []):
            answer_text = answer.get('answer', '')
            logprob = answer.get('avg_logprob')
            
            if answer_text and logprob is not None:
                answers.append(answer_text)
                log_probs.append(logprob)
        
        if not answers or not log_probs:
            logging.warning(f"问题 '{question_data.get('question', '未知问题')}' 没有有效的答案或对数概率")
            return 0.0
        
        # 对答案进行语义聚类
        cluster_ids = self.get_semantic_ids(answers)
        
        # 检查聚类结果
        if len(set(cluster_ids)) == 1:
            logging.warning(f"问题 '{question_data.get('question', '未知问题')}' 的所有答案被聚类到同一组: {cluster_ids}")
        
        # 检查对数概率是否相同
        if len(set(log_probs)) == 1:
            logging.warning(f"问题 '{question_data.get('question', '未知问题')}' 的所有答案对数概率相同: {log_probs[0]}")
        
        # 使用修改后的方法计算语义熵
        entropy = self.calculate_cluster_entropy(cluster_ids, log_probs)
        
        # 记录详细的调试信息
        logging.debug(f"问题: {question_data.get('question', '未知问题')}")
        logging.debug(f"答案数量: {len(answers)}")
        logging.debug(f"答案文本: {answers}")
        logging.debug(f"对数概率: {log_probs}")
        logging.debug(f"聚类结果: {cluster_ids}")
        logging.debug(f"计算得到的语义熵: {entropy}")
        
        return entropy
    
    def calculate_section_entropy(self, section_data: Dict) -> float:
        """
        计算意图的一个部分（合约交互、状态变化等）的语义熵。
        
        参数:
            section_data: 包含部分内容和问题列表的字典
            
        返回:
            该部分的语义熵（三个问题熵的平均值）
        """
        question_entropies = []
        
        # 计算每个问题的熵
        for question in section_data.get('questions', []):
            question_entropy = self.calculate_question_entropy(question)
            question_entropies.append(question_entropy)
        
        # 计算平均熵
        if not question_entropies:
            return 0.0
        
        return sum(question_entropies) / len(question_entropies)
    
    def calculate_intent_entropy(self, intent_data: Dict) -> Dict:
        """
        计算完整意图的语义熵。
        
        参数:
            intent_data: 包含意图和各部分的字典
            
        返回:
            意图的语义熵结果
        """
        # 检查intent_data是否为字典类型
        if not isinstance(intent_data, dict):
            logging.warning(f"意图数据类型错误: {type(intent_data)}, 期望类型: dict")
            return {
                'intent_id': 'unknown',
                'overall_entropy': 0.0,
                'section_entropies': {}
            }
            
        section_entropies = {}
        
        # 检查intent_data中是否存在sections字段
        sections = intent_data.get('sections', [])
        if not sections:
            logging.warning(f"意图数据中没有sections字段或为空")
            
        # 计算每个部分的熵
        for section in sections:
            if not isinstance(section, dict):
                logging.warning(f"部分数据类型错误: {type(section)}, 期望类型: dict")
                continue
                
            section_name = section.get('section_name', 'unknown')
            section_entropy = self.calculate_section_entropy(section)
            section_entropies[section_name] = section_entropy
        
        # 计算平均部分熵
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
        计算单个合约的所有测试的语义熵。
        
        参数:
            contract_data: 合约数据字典，包含test_results字段
            
        返回:
            合约的语义熵结果
        """
        intent_entropies = []
        
        # 检查contract_data是否为字典
        if not isinstance(contract_data, dict):
            logging.warning(f"合约数据类型错误: {type(contract_data)}, 期望类型: dict")
            return {
                'contract_path': 'unknown',
                'avg_overall_entropy': 0.0,
                'intent_entropies': []
            }
            
        # 获取test_results字段，这是一个列表
        test_results = contract_data.get('test_results', [])
        if not test_results:
            logging.warning(f"合约数据中没有test_results字段或为空")
            
        # 对每个意图结果计算熵
        for intent_data in test_results:
            intent_entropy = self.calculate_intent_entropy(intent_data)
            intent_entropies.append(intent_entropy)
        
        # 计算平均值
        overall_entropies = [e['overall_entropy'] for e in intent_entropies]
        avg_overall_entropy = sum(overall_entropies) / len(overall_entropies) if overall_entropies else 0.0
        
        # 使用relative_path作为contract_path
        contract_path = contract_data.get('relative_path', 'unknown')
        
        return {
            'contract_path': contract_path,
            'avg_overall_entropy': avg_overall_entropy,
            'intent_entropies': intent_entropies
        }
    
    def calculate_all_entropies(self) -> Dict:
        """
        计算所有合约的语义熵。
        
        返回:
            所有合约的语义熵结果
        """
        contract_results = self.results.get('contract_intents', {})
        
        entropy_results = {
            'contracts': [],
            'summary': {
                'num_contracts': len(contract_results),
                'avg_overall_entropy': 0.0
            }
        }
        
        # 计算每个合约的熵
        total_entropy = 0.0
        for contract_path, contract_data in contract_results.items():
            logging.info(f"计算合约的语义熵: {contract_path}")
            contract_entropy = self.calculate_contract_entropies(contract_data)
            entropy_results['contracts'].append(contract_entropy)
            total_entropy += contract_entropy['avg_overall_entropy']
        
        # 计算平均熵
        if entropy_results['contracts']:
            entropy_results['summary']['avg_overall_entropy'] = total_entropy / len(entropy_results['contracts'])
        
        return entropy_results
    
    def calculate_entropies(self) -> Dict:
        """
        计算语义熵并保存结果。
        
        返回:
            语义熵结果
        """
        logging.info("开始计算语义熵...")
        start_time = time.time()
        
        # 检查结果中是否包含必要的数据
        if "contract_intents" not in self.results:
            raise ValueError("结果中缺少 'contract_intents' 字段")
        
        # 检查是否包含step3的数据（问题和答案）
        has_step3_data = False
        for contract_data in self.results["contract_intents"].values():
            for intent_data in contract_data.get("test_results", []):
                if "sections" in intent_data:
                    for section in intent_data["sections"]:
                        if "questions" in section:
                            for question in section["questions"]:
                                if "answers" in question:
                                    has_step3_data = True
                                    break
        
        if not has_step3_data:
            raise ValueError("结果中缺少step3的数据（问题和答案），无法计算语义熵")
        
        # 根据模式计算熵值
        if self.mode == "step3":
            entropy_results = self.calculate_entropy_from_step3_results(self.results)
        elif self.mode == "all":
            # 对于all模式，我们仍然使用step3的方法，因为只有step3有答案和logprob
            entropy_results = self.calculate_entropy_from_step3_results(self.results)
        else:
            raise ValueError(f"不支持的模式: {self.mode}，只支持 'step3' 或 'all'")
        
        # 添加时间信息
        end_time = time.time()
        entropy_results["summary"] = {
            "time_taken": end_time - start_time,
            "mode": self.mode
        }
        
        # 保存结果
        results_path = self.output_dir / "entropy_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(entropy_results, f)
        
        # 保存JSON格式的摘要
        summary_path = self.output_dir / "entropy_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(entropy_results["summary"], f, ensure_ascii=False, indent=2)
        
        logging.info(f"语义熵计算完成，耗时: {end_time - start_time:.2f}秒")
        logging.info(f"结果已保存至: {results_path}")
        logging.info(f"摘要已保存至: {summary_path}")
        logging.info(f"整体平均语义熵: {entropy_results['overall_entropy']:.4f}")
        
        return entropy_results

    def calculate_entropy_from_step3_results(self, results: Dict) -> Dict:
        """
        从step3的结果计算语义熵。
        
        参数:
            results: step3生成的结果字典
            
        返回:
            包含语义熵计算结果的字典
        """
        entropy_results = {
            'questions': [],
            'overall_entropy': 0.0
        }
        
        total_entropy = 0.0
        question_count = 0
        
        # 遍历所有合约
        for contract_path, contract_data in results['contract_intents'].items():
            # 遍历每个意图的测试结果
            for intent_data in contract_data['test_results']:
                if 'sections' not in intent_data:
                    continue
                    
                # 遍历每个部分
                for section_data in intent_data['sections']:
                    # 遍历每个问题
                    for question_data in section_data['questions']:
                        if 'answers' not in question_data:
                            continue
                            
                        # 提取答案文本和对数概率
                        answers = []
                        log_probs = []
                        
                        for answer in question_data['answers']:
                            answer_text = answer.get('answer', '')
                            logprob = answer.get('avg_logprob')
                            
                            if answer_text and logprob is not None:
                                answers.append(answer_text)
                                log_probs.append(logprob)
                        
                        if not answers or not log_probs:
                            continue
                            
                        # 对答案进行语义聚类
                        cluster_ids = self.get_semantic_ids(answers)
                        
                        # 计算该问题的语义熵
                        question_entropy = self.calculate_cluster_entropy(cluster_ids, log_probs)
                        
                        # 记录问题级别的熵
                        entropy_results['questions'].append({
                            'question_id': question_data['question_id'],
                            'question': question_data['question'],
                            'section_name': section_data['section_name'],
                            'entropy': question_entropy,
                            'num_answers': len(answers),
                            'num_clusters': len(set(cluster_ids))
                        })
                        
                        total_entropy += question_entropy
                        question_count += 1
        
        # 计算整体平均熵
        if question_count > 0:
            entropy_results['overall_entropy'] = total_entropy / question_count
        
        return entropy_results


def main():
    """Run semantic entropy calculation from command line."""
    parser = argparse.ArgumentParser(description="计算智能合约意图的语义熵")
    
    parser.add_argument("--results_path", type=str, required=True, 
                        help="意图分析结果文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="保存熵结果的目录")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备 (cuda 或 cpu)")
    parser.add_argument("--no_api", action="store_true",
                        help="不使用API进行语义等价判断（默认使用API）")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别")
    
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
        logger.exception(f"语义熵分析出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 