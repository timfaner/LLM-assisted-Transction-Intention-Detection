"""用于智能合约意图生成的模型。"""
import os
import logging
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import httpx

from sc_analyzer.data_types import AnswerWithArgLogprobList,AnswerWithArgLogprob

class BaseModel(ABC):
    """所有意图生成模型的基类。"""
    
    def __init__(self):
        self.intent_prompts = self._default_intent_prompts()
    
    @abstractmethod
    def generate_intent(self, contract_content, transaction_data=""):
        """为给定的智能合约和交易数据生成意图描述。"""
        pass
    
    @abstractmethod
    def generate_questions(self, intent):
        """为给定的意图生成最可能的三个问题。"""
        pass
    
    @abstractmethod
    def generate_questions_for_section(self, section_content, section_name):
        """为意图的特定部分生成三个问题。"""
        pass
    
    @abstractmethod
    def generate_answers(self, question, intent):
        """为给定的问题和相关意图生成三次答案。"""
        pass
    
    def get_prompts_for_log(self):
        """返回用于日志记录的提示。"""
        return self.intent_prompts
    
    def get_model_info(self):
        """返回模型信息。"""
        return {
            "model_type": self.__class__.__name__
        }
    
    def _prepare_format_safe(self, text):
        """确保文本安全用于字符串格式化，处理花括号。"""
        if text is None:
            return ""
        return str(text).replace("{", "{{").replace("}", "}}")
    
    def _format_prompt_safely(self, contract_content, transaction_data):
        """安全地格式化提示词，包含错误处理。"""
        try:
            # 处理花括号以防止格式化错误
            safe_contract = self._prepare_format_safe(contract_content)
            safe_transaction = self._prepare_format_safe(transaction_data)
            
            # 使用格式化
            prompt = self.intent_prompts["generate_intent"].format(
                contract_content=safe_contract,
                transaction_data=safe_transaction
            )
            return prompt
        except (IndexError, KeyError, ValueError) as e:
            logging.error(f"格式化提示时发生错误: {e}")
            logging.error(f"尝试使用替代方法")
            
            # 备用方法：直接替换而不使用format
            prompt = self.intent_prompts["generate_intent"]
            prompt = prompt.replace("{contract_content}", str(contract_content or ""))
            prompt = prompt.replace("{transaction_data}", str(transaction_data or ""))
            return prompt
        except Exception as e:
            logging.error(f"格式化提示时发生未预期的错误: {type(e).__name__}: {e}")
            raise
    
    def _default_intent_prompts(self):
        """Default prompts for intent generation."""
        return {
            "generate_intent": """
Analyze the information carefully and provide your interpretation in a specific structured format.

Here is the transaction data:
<transaction_data>
{transaction_data}
</transaction_data>

Here is the corresponding smart contract code:
<smart_contract_code>
{contract_content}
</smart_contract_code>

Please follow these steps to analyze the transaction:

1. Examine the transaction data carefully, noting details such as the sender, receiver, value transferred, gas price, and any input data.
2. Review the smart contract code and identify which function(s) are being called by this transaction.
3. Interpret how the transaction interacts with the smart contract, including any state changes or events that may be triggered.
4. Consider any potential implications or consequences of this transaction on the blockchain state.

Provide your analysis in the following structured format:

<transaction_analysis>

<contract_interaction>
[Describe how the transaction interacts with the smart contract, including which function(s) are called and their effects]
</contract_interaction>

<state_changes>
[List any state changes in the smart contract that result from this transaction]
</state_changes>

<events>
[List any events that are emitted as a result of this transaction]
</events>

<implications>
[Discuss any broader implications or consequences of this transaction]
</implications>

</transaction_analysis>

Ensure that your analysis is thorough, accurate, and based solely on the provided transaction data and smart contract code. If there's any information you cannot determine from the given data, state this clearly in your analysis.
            """.strip(),
            
            "generate_questions": """
Given the following transaction analysis of a smart contract, generate 3 concise and specific questions that focus on key aspects of the contract's functionality and security.

Transaction Analysis:
{intent}

Generate exactly 3 short, clear questions that cover different important aspects of the contract. Each question should be direct and to the point.

Format your response as a JSON array of exactly 3 questions:
["Question 1", "Question 2", "Question 3"]
            """.strip(),
            
            "generate_questions_for_section": """
Given the following section from a smart contract transaction analysis, generate 3 concise questions about this specific aspect.

Section Type: {section_name}

Section Content:
{section_content}

Generate exactly 3 short, clear questions that focus on key points of this {section_name}. Each question should be direct and to the point.

Format your response as a JSON array of exactly 3 questions:
["Question 1", "Question 2", "Question 3"]
            """.strip(),
            
            "generate_answers": """
You are analyzing a smart contract transaction. Below is the original transaction analysis and a specific question about it. 

Transaction Analysis:
{intent}

Question:
{question}

Provide a detailed, accurate answer to this question based ONLY on the information in the transaction analysis. Be specific and thorough.

Your answer should demonstrate deep technical understanding of blockchain mechanics, smart contract execution, and their implications.
            """.strip()
        }


class LocalLLMModel(BaseModel):
    """使用本地下载的大语言模型。"""
    
    def __init__(self, model_path=None, device="cuda"):
        super().__init__()
        self.device = device
        self.model_path = model_path or "meta-llama/Llama-2-13b-chat-hf"
        
        logging.info(f"Loading local model from {self.model_path} on {self.device}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            # Handle case where model is a local path or HF model identifier
            if Path(self.model_path).exists():
                model_identifier = self.model_path
            else:
                model_identifier = self.model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                device=device if device != "cuda" else 0
            )
            logging.info("本地模型加载成功")
        except Exception as e:
            logging.error(f"加载本地模型时出错: {e}")
            raise
    
    def generate_intent(self, contract_content, transaction_data=""):
        """Generate intent using local LLM."""
        try:
            # 使用基类方法安全格式化提示词
            prompt = self._format_prompt_safely(contract_content, transaction_data)
            
            # 处理可能过长的输入
            max_input_tokens = self.tokenizer.model_max_length - 1024  # 为输出预留空间
            input_ids = self.tokenizer.encode(prompt)
            
            if len(input_ids) > max_input_tokens:
                logging.warning(f"输入过长（{len(input_ids)}个token），截断为{max_input_tokens}个token")
                input_ids = input_ids[:max_input_tokens]
                prompt = self.tokenizer.decode(input_ids)
            
            logging.info(f"使用本地模型生成意图（输入长度：{len(input_ids)}个token）")
            
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            # Extract generated text
            intent = result[0]['generated_text'].strip()
            return intent
        except Exception as e:
            logging.error(f"使用本地模型生成意图时出错: {e}")
            raise
    
    def get_model_info(self):
        """Return model information."""
        info = super().get_model_info()
        info.update({
            "model_path": self.model_path,
            "device": self.device
        })
        return info

    def generate_questions(self, intent):
        """为给定的意图生成三个问题。LocalLLM实现。"""
        try:
            # 格式化提示词
            prompt = self.intent_prompts["generate_questions"].format(intent=intent)
            
            # 使用本地模型生成
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            questions_response = result[0]['generated_text'].strip()
            
            # 尝试解析JSON格式的问题列表
            import json
            try:
                questions = json.loads(questions_response)
                if not isinstance(questions, list) or len(questions) != 3:
                    raise ValueError("格式不正确")
                return questions
            except (json.JSONDecodeError, ValueError):
                # 备用方案：手动提取问题
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"问题{i+1}：此智能合约有何潜在风险？" for i in range(3 - len(fallback_questions))])
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"使用本地模型生成问题时出错: {e}")
            # 返回默认问题
            return [
                "此智能合约的主要功能是什么？",
                "合约交互可能存在哪些安全风险？",
                "此交易会如何影响合约的状态？"
            ]
    
    def generate_answers(self, question, intent):
        """为给定的问题和意图生成答案。LocalLLM实现。"""
        answers_with_logprobs = []
        
        try:
            for i in range(3):  # 生成3次回答
                # 格式化提示词
                prompt = self.intent_prompts["generate_answers"].format(
                    intent=intent,
                    question=question
                )
                
                # 使用本地模型生成
                result = self.pipeline(
                    prompt,
                    do_sample=True,
                    return_full_text=False
                )
                
                answer = result[0]['generated_text'].strip()
                
                # 本地模型可能无法获取logprobs，因此使用None
                answers_with_logprobs.append({
                    'answer': answer,
                    'logprobs': [],
                    'avg_logprob': None
                })
                
                logging.info(f"为问题生成第{i+1}次回答，长度：{len(answer)}字符")
                
        except Exception as e:
            logging.error(f"使用本地模型生成答案时出错: {e}")
            # 返回默认答案
            return [
                {'answer': "无法根据提供的信息回答此问题。", 'logprobs': [], 'avg_logprob': None},
                {'answer': "提供的交易数据不足以确定答案。", 'logprobs': [], 'avg_logprob': None},
                {'answer': "需要更多上下文信息才能准确回答。", 'logprobs': [], 'avg_logprob': None}
            ]
            
        return answers_with_logprobs

    def generate_questions_for_section(self, section_content, section_name):
        """为意图的特定部分生成问题。LocalLLM实现。"""
        try:
            # 格式化提示词
            prompt = self.intent_prompts["generate_questions_for_section"].format(
                section_content=section_content,
                section_name=section_name
            )
            
            # 使用本地模型生成
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            questions_response = result[0]['generated_text'].strip()
            
            # 尝试解析JSON格式的问题列表
            import json
            try:
                questions = json.loads(questions_response)
                if not isinstance(questions, list) or len(questions) != 3:
                    raise ValueError("格式不正确")
                return questions
            except (json.JSONDecodeError, ValueError):
                # 备用方案：手动提取问题
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"关于此{section_name}，请问有何潜在风险？" for i in range(3 - len(fallback_questions))])
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"使用本地模型为{section_name}部分生成问题时出错: {e}")
            # 返回默认问题
            return [
                f"这个{section_name}的主要功能或影响是什么？",
                f"此{section_name}可能存在哪些安全风险或边缘情况？",
                f"如何改进或优化此{section_name}的实现？"
            ]


class LegacyAPIModel(BaseModel):
    """Model using LLM API."""
    
    def __init__(self, model_name="gpt-4", api_key=None, embedding_model="text-embedding-ada-002", 
                 http_proxy=None, https_proxy=None):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # 只支持OpenAI API
        self.api_type = "openai"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # 代理设置
        self.http_proxy = http_proxy or os.environ.get("HTTP_PROXY")
        self.https_proxy = https_proxy or os.environ.get("HTTPS_PROXY")
        
        logging.info(f"使用API模型：通过{self.api_type} API使用{model_name}，嵌入模型：{embedding_model}")
        if self.http_proxy or self.https_proxy:
            logging.debug(f"使用代理配置：HTTP:{self.http_proxy}, HTTPS:{self.https_proxy}")
            
        self._setup_client()
    
    def _setup_client(self):
        """设置OpenAI客户端。"""
        try:
            from openai import OpenAI
            
            # 创建客户端时设置代理
            client_kwargs = {"api_key": self.api_key}
            
            # 设置代理
            if self.http_proxy or self.https_proxy:
                # 根据httpx文档，直接使用字符串形式的代理
                proxy = None
                if self.https_proxy:  # 优先使用HTTPS代理
                    proxy = self.https_proxy
                elif self.http_proxy:
                    proxy = self.http_proxy
                
                if proxy:
                    # 使用httpx的正确代理设置语法：使用proxy参数而非proxies
                    client_kwargs["http_client"] = httpx.Client(proxy=proxy)
                
            self.client = OpenAI(**client_kwargs)
            logging.info(f"API客户端初始化成功")
        except Exception as e:
            logging.error(f"初始化API客户端时出错：{e}")
            raise
    
    def generate_intent(self, contract_content, transaction_data=""):
        """使用OpenAI API生成意图并获取token概率和嵌入向量。"""
        try:
            # 使用基类方法安全格式化提示词
            prompt = self._format_prompt_safely(contract_content, transaction_data)
            
            # 生成意图文本
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an intelligent contract engineer with a deep understanding of blockchain transaction structures. Your task is to interpret a given transaction using the provided transaction data and corresponding smart contract code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                logprobs=True,
                top_logprobs=5
            )
            intent = response.choices[0].message.content.strip()
            
            # 提取token对数似然概率
            token_log_likelihoods = []
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
                for token_info in response.choices[0].logprobs.content:
                    if hasattr(token_info, 'logprob'):
                        token_log_likelihoods.append(token_info.logprob)
            
            # 获取意图文本的嵌入向量
            try:
                embedding_response = self.client.embeddings.create(
                    model=self.embedding_model,  # 使用指定的嵌入模型
                    input=intent
                )
                embedding = embedding_response.data[0].embedding
                logging.info(f"成功获取意图嵌入向量，维度: {len(embedding)}")
            except Exception as e:
                logging.warning(f"获取嵌入向量时出错: {e}")
                embedding = None
            
            return intent, token_log_likelihoods, embedding
            
        except Exception as e:
            logging.error(f"通过OpenAI API生成意图时出错: {e}")
            raise
    
    def generate_questions(self, intent):
        """根据给定的意图生成三个最可能的问题。"""
        try:
            # 格式化提示词
            prompt = self.intent_prompts["generate_questions"].format(intent=intent)
            
            # 生成问题
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个区块链智能合约审计专家，擅长针对智能合约交易提出有洞察力的问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            questions_response = response.choices[0].message.content.strip()
            
            # 解析JSON格式的问题列表
            import json
            try:
                questions = json.loads(questions_response)
                # 确保正好有3个问题
                if not isinstance(questions, list):
                    raise ValueError("返回格式不是列表")
                
                # 如果不是恰好3个问题，记录警告并修正
                if len(questions) != 3:
                    logging.warning(f"API返回了{len(questions)}个问题，而不是预期的3个")
                    if len(questions) < 3:
                        # 补足到3个问题
                        for i in range(3 - len(questions)):
                            questions.append(f"问题{len(questions) + 1}：此智能合约的安全性如何？")
                    else:
                        # 截取前3个问题
                        questions = questions[:3]
                
                return questions
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"解析问题时出错: {e}")
                logging.error(f"API返回的原始内容: {questions_response}")
                
                # 备用方案：手动提取问题（简单分割）
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"问题{i+1}：此智能合约有何潜在风险？" for i in range(3 - len(fallback_questions))])
                
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"生成问题时出错: {e}")
            # 返回默认问题
            return [
                "此智能合约的主要功能是什么？",
                "合约交互可能存在哪些安全风险？",
                "此交易会如何影响合约的状态？"
            ]
    
    def generate_answers(self, question, intent):
        """为给定的问题和意图生成答案，并返回答案及其logprob。"""
        answers_with_logprobs = []
        
        try:
            for i in range(3):  # 生成3次回答
                # 格式化提示词
                prompt = self.intent_prompts["generate_answers"].format(
                    intent=intent,
                    question=question
                )
                
                # 生成答案
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个区块链智能合约专家，擅长解答关于智能合约交易的技术问题。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    logprobs=True,
                    top_logprobs=5
                )
                
                answer = response.choices[0].message.content.strip()
                
                # 提取token对数似然概率
                token_log_likelihoods = []
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
                    for token_info in response.choices[0].logprobs.content:
                        if hasattr(token_info, 'logprob'):
                            token_log_likelihoods.append(token_info.logprob)
                
                # 计算答案的平均logprob
                avg_logprob = sum(token_log_likelihoods) / len(token_log_likelihoods) if token_log_likelihoods else None
                
                answers_with_logprobs.append({
                    'answer': answer,
                    'logprobs': token_log_likelihoods,
                    'avg_logprob': avg_logprob
                })
                
                logging.info(f"为问题生成第{i+1}次回答，长度：{len(answer)}字符，平均logprob：{avg_logprob}")
                
        except Exception as e:
            logging.error(f"生成答案时出错: {e}")
            # 如果已经有一些成功的答案，返回已有的
            if answers_with_logprobs:
                return answers_with_logprobs
            
            # 否则返回默认答案
            return [
                {'answer': "无法根据提供的信息回答此问题。", 'logprobs': [], 'avg_logprob': None},
                {'answer': "提供的交易数据不足以确定答案。", 'logprobs': [], 'avg_logprob': None},
                {'answer': "需要更多上下文信息才能准确回答。", 'logprobs': [], 'avg_logprob': None}
            ]
            
        return answers_with_logprobs
    
    def get_model_info(self):
        """Return model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "api_type": self.api_type,
            "embedding_model": self.embedding_model
        })
        return info

    def generate_questions_for_section(self, section_content, section_name):
        """根据给定的意图部分生成三个最可能的问题。"""
        try:
            # 格式化提示词
            prompt = self.intent_prompts["generate_questions_for_section"].format(
                section_content=section_content,
                section_name=section_name
            )
            
            # 生成问题
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"你是一个区块链智能合约审计专家，擅长针对智能合约交易的{section_name}部分提出有洞察力的问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            questions_response = response.choices[0].message.content.strip()
            
            # 解析JSON格式的问题列表
            import json
            try:
                questions = json.loads(questions_response)
                # 确保正好有3个问题
                if not isinstance(questions, list):
                    raise ValueError("返回格式不是列表")
                
                # 如果不是恰好3个问题，记录警告并修正
                if len(questions) != 3:
                    logging.warning(f"API返回了{len(questions)}个问题，而不是预期的3个")
                    if len(questions) < 3:
                        # 补足到3个问题
                        for i in range(3 - len(questions)):
                            questions.append(f"问题{len(questions) + 1}：此{section_name}的安全性如何？")
                    else:
                        # 截取前3个问题
                        questions = questions[:3]
                
                return questions
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"解析问题时出错: {e}")
                logging.error(f"API返回的原始内容: {questions_response}")
                
                # 备用方案：手动提取问题（简单分割）
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"问题{i+1}：此{section_name}有何潜在风险？" for i in range(3 - len(fallback_questions))])
                
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"为{section_name}生成问题时出错: {e}")
            # 返回默认问题
            return [
                f"这个{section_name}的主要功能或影响是什么？",
                f"此{section_name}可能存在哪些安全风险或边缘情况？",
                f"如何改进或优化此{section_name}的实现？"
            ]


## todo 兼容 openai, deepseek, claude
class APIModel():
    """Model using LLM API."""
    
    def __init__(self,model_provider="openai", model_name="gpt-4", api_key=None, embedding_model="text-embedding-ada-002", 
                 http_proxy=None, https_proxy=None):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # 只支持OpenAI API
        self.api_type = "openai"
        self.api_key = api_key
        
        # 代理设置
        self.http_proxy = http_proxy 
        self.https_proxy = https_proxy
        
        logging.info(f"使用API模型：通过{self.api_type} API使用{model_name}，嵌入模型：{embedding_model}")
        if self.http_proxy or self.https_proxy:
            logging.debug(f"使用代理配置：HTTP:{self.http_proxy}, HTTPS:{self.https_proxy}")
            
        self._setup_client()
    
    def _setup_client(self):
        """设置OpenAI客户端。"""
        try:
            from openai import OpenAI
            
            # 创建客户端时设置代理
            client_kwargs = {"api_key": self.api_key}
            
            # 设置代理
            if self.http_proxy or self.https_proxy:
                # 根据httpx文档，直接使用字符串形式的代理
                proxy = None
                if self.https_proxy:  # 优先使用HTTPS代理
                    proxy = self.https_proxy
                elif self.http_proxy:
                    proxy = self.http_proxy
                
                if proxy:
                    # 使用httpx的正确代理设置语法：使用proxy参数而非proxies
                    client_kwargs["http_client"] = httpx.Client(proxy=proxy)
                
            self.client = OpenAI(**client_kwargs)
            logging.info(f"API客户端初始化成功")
        except Exception as e:
            logging.error(f"初始化API客户端时出错：{e}")
            raise
    
    def get_multiple_answers(self, answers_num = 3, answers_temperature = 0.7, system_prompt = "You are an expert in smart contract analysis.",
                             question = None):
        """为给定的问题和意图生成答案，并返回答案及其logprob。"""
        answers_with_logprobs:AnswerWithArgLogprobList = []
        
        try:
            for i in range(answers_num):  
                
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]
                
                output,token_log_likelihoods = self.get_response_with_probs(messages, answers_temperature, 512)
                avg_logprob = sum(token_log_likelihoods) / len(token_log_likelihoods)
                
                answers_with_logprobs.append(
                    AnswerWithArgLogprob(
                        answer_id = i,
                        answer=output,
                        token_log_likelihoods=token_log_likelihoods,
                        avg_logprob=avg_logprob
                    )
                )
                
                logging.info(f"为问题生成第{i+1}次回答，平均logprob：{avg_logprob}")
                
        except Exception as e:
            logging.error(f"生成答案时出错: {e}")
            
        return answers_with_logprobs
    
    def get_response(self, messages, temperature, max_tokens=512):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output = response.choices[0].message.content
        
        return output
    
    def get_response_with_probs(self, messages, temperature, max_tokens=512):


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5
            )
        token_log_likelihoods = []
        output = response.choices[0].message.content

        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
            for token_info in response.choices[0].logprobs.content:
                if hasattr(token_info, 'logprob'):
                    token_log_likelihoods.append(token_info.logprob)
        else:
            logging.error(f"没有找到logprobs")
            token_log_likelihoods = None
            
        return output,token_log_likelihoods

    
    def get_model_info(self):
        """Return model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "api_type": self.api_type,
            "embedding_model": self.embedding_model
        })
        return info


from sc_analyzer.config import ApiKeyConfig

keys = ApiKeyConfig()

def get_model(model_type:str, 
              model_name:str=None, 
              model_provider:str="openai", 
              local_model_path:str=None, 
              device:str="cuda", 
              embedding_model:str="text-embedding-ada-002",
              http_proxy:str=None, 
              https_proxy:str=None):
    """Factory function to get the appropriate model."""
    if model_type == "local":
        return LocalLLMModel(model_path=local_model_path, device=device)
    elif model_type == "api":
        return APIModel(model_name=model_name,
                        model_provider=model_provider,
                        api_key=keys.get_api_key(model_provider), 
                        embedding_model=embedding_model,
                        http_proxy=http_proxy, 
                        https_proxy=https_proxy)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
