"""用于智能合约意图生成的模型。"""
import os
import logging
from abc import ABC, abstractmethod
import torch
from pathlib import Path

class BaseModel(ABC):
    """所有意图生成模型的基类。"""
    
    def __init__(self):
        self.intent_prompts = self._default_intent_prompts()
    
    @abstractmethod
    def generate_intent(self, contract_content):
        """为给定的智能合约生成意图描述。"""
        pass
    
    def get_prompts_for_log(self):
        """返回用于日志记录的提示。"""
        return self.intent_prompts
    
    def get_model_info(self):
        """返回模型信息。"""
        return {
            "model_type": self.__class__.__name__
        }
    
    def _default_intent_prompts(self):
        """Default prompts for intent generation."""
        return {
            "generate_intent": """
You are an expert in blockchain and smart contract analysis.
Given the following Solidity smart contract code, provide a comprehensive explanation of its purpose and functionality.
Your explanation should include:
1. The main purpose of the contract
2. Key functions and their roles
3. Notable design patterns used
4. Potential security considerations
5. The blockchain ecosystem it's designed for (if apparent)

Please be detailed but concise.

Smart Contract:
```solidity
{contract_content}
```
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
    
    def generate_intent(self, contract_content):
        """Generate intent using local LLM."""
        prompt = self.intent_prompts["generate_intent"].format(contract_content=contract_content)
        
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
    
    def get_model_info(self):
        """Return model information."""
        info = super().get_model_info()
        info.update({
            "model_path": self.model_path,
            "device": self.device
        })
        return info


class APIModel(BaseModel):
    """Model using LLM API."""
    
    def __init__(self, model_name="gpt-4", api_key=None):
        super().__init__()
        self.model_name = model_name
        
        # 根据模型名称确定API类型
        if "gpt" in model_name.lower():
            self.api_type = "openai"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif "claude" in model_name.lower():
            self.api_type = "anthropic"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"API不支持的模型类型：{model_name}")
        
        logging.info(f"使用API模型：通过{self.api_type} API使用{model_name}")
        self._setup_client()
    
    def _setup_client(self):
        """根据API类型设置相应的客户端。"""
        try:
            if self.api_type == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.api_type == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            logging.info(f"API客户端初始化成功")
        except Exception as e:
            logging.error(f"初始化API客户端时出错：{e}")
            raise
    
    def generate_intent(self, contract_content):
        """使用API生成意图。"""
        prompt = self.intent_prompts["generate_intent"].format(contract_content=contract_content)
        
        try:
            if self.api_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert smart contract analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                intent = response.choices[0].message.content
            
            elif self.api_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0.7,
                    system="You are an expert smart contract analyzer.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                intent = response.content[0].text
            
            return intent.strip()
            
        except Exception as e:
            logging.error(f"Error generating intent via API: {e}")
            raise
    
    def get_model_info(self):
        """Return model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "api_type": self.api_type
        })
        return info


def get_model(model_type, model_name=None, api_key=None, use_local=False, 
              local_model_path=None, device="cuda"):
    """Factory function to get the appropriate model."""
    if use_local or model_type == "local":
        return LocalLLMModel(model_path=local_model_path, device=device)
    else:
        return APIModel(model_name=model_name, api_key=api_key)
