"""texttexttexttexttexttexttexttexttexttexttexttexttext """
import os
import logging
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import httpx

from sc_analyzer.data_types import AnswerWithArgLogprobList,AnswerWithArgLogprob

class BaseModel(ABC):
    """texttexttexttexttexttexttexttexttexttexttext """
    
    def __init__(self):
        self.intent_prompts = self._default_intent_prompts()
    
    @abstractmethod
    def generate_intent(self, contract_content, transaction_data=""):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
        pass
    
    @abstractmethod
    def generate_questions(self, intent):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
        pass
    
    @abstractmethod
    def generate_questions_for_section(self, section_content, section_name):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttext """
        pass
    
    @abstractmethod
    def generate_answers(self, question, intent):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
        pass
    
    def get_prompts_for_log(self):
        """texttexttexttexttexttexttexttexttexttexttext """
        return self.intent_prompts
    
    def get_model_info(self):
        """texttexttexttexttexttext """
        return {
            "model_type": self.__class__.__name__
        }
    
    def _prepare_format_safe(self, text):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttext """
        if text is None:
            return ""
        return str(text).replace("{", "{{").replace("}", "}}")
    
    def _format_prompt_safely(self, contract_content, transaction_data):
        """texttexttexttexttexttexttexttexttext texttexttexttexttexttext """
        try:
            # texttexttexttexttexttexttexttexttexttexttexttexttext
            safe_contract = self._prepare_format_safe(contract_content)
            safe_transaction = self._prepare_format_safe(transaction_data)
            
            # texttexttexttexttext
            prompt = self.intent_prompts["generate_intent"].format(
                contract_content=safe_contract,
                transaction_data=safe_transaction
            )
            return prompt
        except (IndexError, KeyError, ValueError) as e:
            logging.error(f"texttexttexttexttexttexttexttexttexttext: {e}")
            logging.error(f"texttexttexttexttexttexttexttext")
            
            # texttexttexttext texttexttexttexttexttexttexttextformat
            prompt = self.intent_prompts["generate_intent"]
            prompt = prompt.replace("{contract_content}", str(contract_content or ""))
            prompt = prompt.replace("{transaction_data}", str(transaction_data or ""))
            return prompt
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttexttexttexttexttexttexttexttext: {type(e).__name__}: {e}")
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
    """texttexttexttexttexttexttexttexttexttexttexttext """
    
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
            logging.info("texttexttexttexttexttexttexttext")
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttexttexttext: {e}")
            raise
    
    def generate_intent(self, contract_content, transaction_data=""):
        """Generate intent using local LLM."""
        try:
            # texttexttexttexttexttexttexttexttexttexttexttexttexttext
            prompt = self._format_prompt_safely(contract_content, transaction_data)
            
            # texttexttexttexttexttexttexttexttext
            max_input_tokens = self.tokenizer.model_max_length - 1024  # texttexttexttexttexttexttext
            input_ids = self.tokenizer.encode(prompt)
            
            if len(input_ids) > max_input_tokens:
                logging.warning(f"texttexttexttext {len(input_ids)}texttoken  texttexttext{max_input_tokens}texttoken")
                input_ids = input_ids[:max_input_tokens]
                prompt = self.tokenizer.decode(input_ids)
            
            logging.info(f"texttexttexttexttexttexttexttexttexttext texttexttexttext {len(input_ids)}texttoken ")
            
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            # Extract generated text
            intent = result[0]['generated_text'].strip()
            return intent
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttexttexttexttexttexttexttext: {e}")
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
        """texttexttexttexttexttexttexttexttexttexttexttext LocalLLMtexttext """
        try:
            # texttexttexttexttexttext
            prompt = self.intent_prompts["generate_questions"].format(intent=intent)
            
            # texttexttexttexttexttexttexttext
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            questions_response = result[0]['generated_text'].strip()
            
            # texttexttexttextJSONtexttexttexttexttexttexttext
            import json
            try:
                questions = json.loads(questions_response)
                if not isinstance(questions, list) or len(questions) != 3:
                    raise ValueError("texttexttexttexttext")
                return questions
            except (json.JSONDecodeError, ValueError):
                # texttexttexttext texttexttexttexttexttext
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"texttext{i+1} texttexttexttexttexttexttexttexttexttexttext " for i in range(3 - len(fallback_questions))])
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttexttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttext
            return [
                "texttexttexttexttexttexttexttexttexttexttexttexttext ",
                "texttexttexttexttexttexttexttexttexttexttexttexttexttext ",
                "texttexttexttexttexttexttexttexttexttexttexttexttext "
            ]
    
    def generate_answers(self, question, intent):
        """texttexttexttexttexttexttexttexttexttexttexttexttext LocalLLMtexttext """
        answers_with_logprobs = []
        
        try:
            for i in range(3):  # texttext3texttexttext
                # texttexttexttexttexttext
                prompt = self.intent_prompts["generate_answers"].format(
                    intent=intent,
                    question=question
                )
                
                # texttexttexttexttexttexttexttext
                result = self.pipeline(
                    prompt,
                    do_sample=True,
                    return_full_text=False
                )
                
                answer = result[0]['generated_text'].strip()
                
                # texttexttexttexttexttexttexttexttexttextlogprobs texttexttexttextNone
                answers_with_logprobs.append({
                    'answer': answer,
                    'logprobs': [],
                    'avg_logprob': None
                })
                
                logging.info(f"texttexttexttexttexttext{i+1}texttexttext texttext {len(answer)}texttext")
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttexttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttext
            return [
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None},
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None},
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None}
            ]
            
        return answers_with_logprobs

    def generate_questions_for_section(self, section_content, section_name):
        """texttexttexttexttexttexttexttexttexttexttexttext LocalLLMtexttext """
        try:
            # texttexttexttexttexttext
            prompt = self.intent_prompts["generate_questions_for_section"].format(
                section_content=section_content,
                section_name=section_name
            )
            
            # texttexttexttexttexttexttexttext
            result = self.pipeline(
                prompt,
                do_sample=True,
                return_full_text=False
            )
            
            questions_response = result[0]['generated_text'].strip()
            
            # texttexttexttextJSONtexttexttexttexttexttexttext
            import json
            try:
                questions = json.loads(questions_response)
                if not isinstance(questions, list) or len(questions) != 3:
                    raise ValueError("texttexttexttexttext")
                return questions
            except (json.JSONDecodeError, ValueError):
                # texttexttexttext texttexttexttexttexttext
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"texttexttext{section_name} texttexttexttexttexttexttexttext " for i in range(3 - len(fallback_questions))])
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttext{section_name}texttexttexttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttext
            return [
                f"texttext{section_name}texttexttexttexttexttexttexttexttexttexttext ",
                f"text{section_name}texttexttexttexttexttexttexttexttexttexttexttexttexttexttext ",
                f"texttexttexttexttexttexttexttext{section_name}texttexttext "
            ]


class LegacyAPIModel(BaseModel):
    """Model using LLM API."""
    
    def __init__(self, model_name="gpt-4", api_key=None, embedding_model="text-embedding-ada-002", 
                 http_proxy=None, https_proxy=None):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # texttexttextOpenAI API
        self.api_type = "openai"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # texttexttexttext
        self.http_proxy = http_proxy or os.environ.get("HTTP_PROXY")
        self.https_proxy = https_proxy or os.environ.get("HTTPS_PROXY")
        
        logging.info(f"texttextAPItexttext texttext{self.api_type} APItexttext{model_name} texttexttexttext {embedding_model}")
        if self.http_proxy or self.https_proxy:
            logging.debug(f"texttexttexttexttexttext HTTP:{self.http_proxy}, HTTPS:{self.https_proxy}")
            
        self._setup_client()
    
    def _setup_client(self):
        """texttextOpenAItexttexttext """
        try:
            from openai import OpenAI
            
            # texttexttexttexttexttexttexttexttexttext
            client_kwargs = {"api_key": self.api_key}
            
            # texttexttexttext
            if self.http_proxy or self.https_proxy:
                # texttexthttpxtexttext texttexttexttexttexttexttexttexttexttexttexttext
                proxy = None
                if self.https_proxy:  # texttexttexttextHTTPStexttext
                    proxy = self.https_proxy
                elif self.http_proxy:
                    proxy = self.http_proxy
                
                if proxy:
                    # texttexthttpxtexttexttexttexttexttexttexttexttext texttextproxytexttexttexttextproxies
                    client_kwargs["http_client"] = httpx.Client(proxy=proxy)
                
            self.client = OpenAI(**client_kwargs)
            logging.info(f"APItexttexttexttexttexttexttexttext")
        except Exception as e:
            logging.error(f"texttexttextAPItexttexttexttexttexttext {e}")
            raise
    
    def generate_intent(self, contract_content, transaction_data=""):
        """texttextOpenAI APItexttexttexttexttexttexttexttokentexttexttexttexttexttexttext """
        try:
            # texttexttexttexttexttexttexttexttexttexttexttexttexttext
            prompt = self._format_prompt_safely(contract_content, transaction_data)
            
            # texttexttexttexttexttext
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
            
            # texttexttokentexttexttexttexttexttext
            token_log_likelihoods = []
            if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
                for token_info in response.choices[0].logprobs.content:
                    if hasattr(token_info, 'logprob'):
                        token_log_likelihoods.append(token_info.logprob)
            
            # texttexttexttexttexttexttexttexttexttexttext
            try:
                embedding_response = self.client.embeddings.create(
                    model=self.embedding_model,  # texttexttexttexttexttexttexttexttext
                    input=intent
                )
                embedding = embedding_response.data[0].embedding
                logging.info(f"texttexttexttexttexttexttexttexttexttext texttext: {len(embedding)}")
            except Exception as e:
                logging.warning(f"texttexttexttexttexttexttexttexttext: {e}")
                embedding = None
            
            return intent, token_log_likelihoods, embedding
            
        except Exception as e:
            logging.error(f"texttextOpenAI APItexttexttexttexttexttexttext: {e}")
            raise
    
    def generate_questions(self, intent):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
        try:
            # texttexttexttexttexttext
            prompt = self.intent_prompts["generate_questions"].format(intent=intent)
            
            # texttexttexttext
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "texttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            questions_response = response.choices[0].message.content.strip()
            
            # texttextJSONtexttexttexttexttexttexttext
            import json
            try:
                questions = json.loads(questions_response)
                # texttexttexttexttext3texttexttext
                if not isinstance(questions, list):
                    raise ValueError("texttexttexttexttexttexttexttext")
                
                # texttexttexttexttexttext3texttexttext texttexttexttexttexttexttext
                if len(questions) != 3:
                    logging.warning(f"APItexttexttext{len(questions)}texttexttext texttexttexttexttexttext3text")
                    if len(questions) < 3:
                        # texttexttext3texttexttext
                        for i in range(3 - len(questions)):
                            questions.append(f"texttext{len(questions) + 1} texttexttexttexttexttexttexttexttexttexttext ")
                    else:
                        # texttexttext3texttexttext
                        questions = questions[:3]
                
                return questions
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"texttexttexttexttexttexttext: {e}")
                logging.error(f"APItexttexttexttexttexttexttext: {questions_response}")
                
                # texttexttexttext texttexttexttexttexttext texttexttexttext 
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"texttext{i+1} texttexttexttexttexttexttexttexttexttexttext " for i in range(3 - len(fallback_questions))])
                
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttext
            return [
                "texttexttexttexttexttexttexttexttexttexttexttexttext ",
                "texttexttexttexttexttexttexttexttexttexttexttexttexttext ",
                "texttexttexttexttexttexttexttexttexttexttexttexttext "
            ]
    
    def generate_answers(self, question, intent):
        """texttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttextlogprob """
        answers_with_logprobs = []
        
        try:
            for i in range(3):  # texttext3texttexttext
                # texttexttexttexttexttext
                prompt = self.intent_prompts["generate_answers"].format(
                    intent=intent,
                    question=question
                )
                
                # texttexttexttext
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "texttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext "},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    logprobs=True,
                    top_logprobs=5
                )
                
                answer = response.choices[0].message.content.strip()
                
                # texttexttokentexttexttexttexttexttext
                token_log_likelihoods = []
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
                    for token_info in response.choices[0].logprobs.content:
                        if hasattr(token_info, 'logprob'):
                            token_log_likelihoods.append(token_info.logprob)
                
                # texttexttexttexttexttexttextlogprob
                avg_logprob = sum(token_log_likelihoods) / len(token_log_likelihoods) if token_log_likelihoods else None
                
                answers_with_logprobs.append({
                    'answer': answer,
                    'logprobs': token_log_likelihoods,
                    'avg_logprob': avg_logprob
                })
                
                logging.info(f"texttexttexttexttexttext{i+1}texttexttext texttext {len(answer)}texttext texttextlogprob {avg_logprob}")
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttext
            if answers_with_logprobs:
                return answers_with_logprobs
            
            # texttexttexttexttexttexttexttext
            return [
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None},
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None},
                {'answer': "texttexttexttexttexttexttexttexttexttexttexttexttexttexttext ", 'logprobs': [], 'avg_logprob': None}
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
        """texttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttexttext """
        try:
            # texttexttexttexttexttext
            prompt = self.intent_prompts["generate_questions_for_section"].format(
                section_content=section_content,
                section_name=section_name
            )
            
            # texttexttexttext
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"texttexttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttexttexttext{section_name}texttexttexttexttexttexttexttexttexttexttext "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            questions_response = response.choices[0].message.content.strip()
            
            # texttextJSONtexttexttexttexttexttexttext
            import json
            try:
                questions = json.loads(questions_response)
                # texttexttexttexttext3texttexttext
                if not isinstance(questions, list):
                    raise ValueError("texttexttexttexttexttexttexttext")
                
                # texttexttexttexttexttext3texttexttext texttexttexttexttexttexttext
                if len(questions) != 3:
                    logging.warning(f"APItexttexttext{len(questions)}texttexttext texttexttexttexttexttext3text")
                    if len(questions) < 3:
                        # texttexttext3texttexttext
                        for i in range(3 - len(questions)):
                            questions.append(f"texttext{len(questions) + 1} text{section_name}texttexttexttexttexttext ")
                    else:
                        # texttexttext3texttexttext
                        questions = questions[:3]
                
                return questions
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"texttexttexttexttexttexttext: {e}")
                logging.error(f"APItexttexttexttexttexttexttext: {questions_response}")
                
                # texttexttexttext texttexttexttexttexttext texttexttexttext 
                fallback_questions = questions_response.strip().split('\n')[:3]
                if len(fallback_questions) < 3:
                    fallback_questions.extend([f"texttext{i+1} text{section_name}texttexttexttexttexttext " for i in range(3 - len(fallback_questions))])
                
                return fallback_questions[:3]
                
        except Exception as e:
            logging.error(f"text{section_name}texttexttexttexttexttexttext: {e}")
            # texttexttexttexttexttext
            return [
                f"texttext{section_name}texttexttexttexttexttexttexttexttexttexttext ",
                f"text{section_name}texttexttexttexttexttexttexttexttexttexttexttexttexttexttext ",
                f"texttexttexttexttexttexttexttext{section_name}texttexttext "
            ]


## todo texttext openai, deepseek, claude
class APIModel():
    """Model using LLM API."""
    
    def __init__(self,model_provider="openai", model_name="gpt-4", api_key=None, embedding_model="text-embedding-ada-002", 
                 http_proxy=None, https_proxy=None):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # texttexttextOpenAI API
        self.api_type = "openai"
        self.api_key = api_key
        
        # texttexttexttext
        self.http_proxy = http_proxy 
        self.https_proxy = https_proxy
        
        logging.info(f"texttextAPItexttext texttext{self.api_type} APItexttext{model_name} texttexttexttext {embedding_model}")
        if self.http_proxy or self.https_proxy:
            logging.debug(f"texttexttexttexttexttext HTTP:{self.http_proxy}, HTTPS:{self.https_proxy}")
            
        self._setup_client()
    
    def _setup_client(self):
        """texttextOpenAItexttexttext """
        try:
            from openai import OpenAI
            
            # texttexttexttexttexttexttexttexttexttext
            client_kwargs = {"api_key": self.api_key}
            
            # texttexttexttext
            if self.http_proxy or self.https_proxy:
                # texttexthttpxtexttext texttexttexttexttexttexttexttexttexttexttexttext
                proxy = None
                if self.https_proxy:  # texttexttexttextHTTPStexttext
                    proxy = self.https_proxy
                elif self.http_proxy:
                    proxy = self.http_proxy
                
                if proxy:
                    # texttexthttpxtexttexttexttexttexttexttexttexttext texttextproxytexttexttexttextproxies
                    client_kwargs["http_client"] = httpx.Client(proxy=proxy)
                
            self.client = OpenAI(**client_kwargs)
            logging.info(f"APItexttexttexttexttexttexttexttext")
        except Exception as e:
            logging.error(f"texttexttextAPItexttexttexttexttexttext {e}")
            raise
    
    def get_multiple_answers(self, answers_num = 3, answers_temperature = 0.7, system_prompt = "You are an expert in smart contract analysis.",
                             question = None):
        """texttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttextlogprob """
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
                
                logging.info(f"texttexttexttexttexttext{i+1}texttexttext texttextlogprob {avg_logprob}")
                
        except Exception as e:
            logging.error(f"texttexttexttexttexttexttext: {e}")
            
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
            logging.error(f"texttexttexttextlogprobs")
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
        raise ValueError(f"texttexttexttexttexttexttexttext: {model_type}")
