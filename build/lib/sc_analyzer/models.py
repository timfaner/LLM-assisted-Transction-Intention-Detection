"""texttexttexttexttexttexttexttexttexttexttexttexttext """
import os
import logging
from abc import ABC, abstractmethod
import torch
from pathlib import Path

class BaseModel(ABC):
    """texttexttexttexttexttexttexttexttexttexttext """
    
    def __init__(self):
        self.intent_prompts = self._default_intent_prompts()
    
    @abstractmethod
    def generate_intent(self, contract_content):
        """texttexttexttexttexttexttexttexttexttexttexttexttexttext """
        pass
    
    def get_prompts_for_log(self):
        """texttexttexttexttexttexttexttexttexttexttext """
        return self.intent_prompts
    
    def get_model_info(self):
        """texttexttexttexttexttext """
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
    
    def generate_intent(self, contract_content):
        """Generate intent using local LLM."""
        prompt = self.intent_prompts["generate_intent"].format(contract_content=contract_content)
        
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
        
        # texttexttexttexttexttexttexttextAPItexttext
        if "gpt" in model_name.lower():
            self.api_type = "openai"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif "claude" in model_name.lower():
            self.api_type = "anthropic"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"APItexttexttexttexttexttexttexttext {model_name}")
        
        logging.info(f"texttextAPItexttext texttext{self.api_type} APItexttext{model_name}")
        self._setup_client()
    
    def _setup_client(self):
        """texttextAPItexttexttexttexttexttexttexttexttexttext """
        try:
            if self.api_type == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.api_type == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            logging.info(f"APItexttexttexttexttexttexttexttext")
        except Exception as e:
            logging.error(f"texttexttextAPItexttexttexttexttexttext {e}")
            raise
    
    def generate_intent(self, contract_content):
        """texttextAPItexttexttexttext """
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
