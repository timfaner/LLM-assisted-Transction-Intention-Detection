"""texttexttexttexttexttexttexttexttexttexttexttexttexttext """
import os
import argparse
import logging
import json
from pathlib import Path
import torch
from typing import Dict, Any, Optional, List


class Config:
    """texttexttexttexttext texttexttexttexttexttexttexttexttexttexttexttexttexttext """
    
    def __init__(self):
        """texttexttexttexttexttexttexttext """
        self.parser = self._create_arg_parser()
        self.args = None
    
    def _create_arg_parser(self) -> argparse.ArgumentParser:
        """texttexttexttexttexttexttext 
        
        Returns:
            texttexttexttexttexttexttext
        """
        parser = argparse.ArgumentParser(description="texttexttexttexttexttexttexttexttext")
        
        # texttexttexttexttexttext
        parser.add_argument("--config", type=str, default=None,
                            help="texttexttexttexttexttext texttextJSONtextYAMLtexttext ")
        
        # texttext/texttexttexttext
        parser.add_argument("--input_dir", type=str,
                            help="texttexttexttexttexttexttexttexttexttexttexttext")
        parser.add_argument("--wandb_dir", type=str, default=None,
                            help="wandbtexttexttexttexttext texttext: intent_resultstexttext ")
        parser.add_argument("--max_contracts", type=int, default=0,
                            help="texttexttexttexttexttexttexttexttexttext 0texttexttexttexttext ")
        parser.add_argument("--save_interval", type=int, default=10,
                            help="texttexttexttexttexttexttexttexttexttexttexttexttext 0texttexttexttext ")
        parser.add_argument("--num_tests", type=int, default=6,
                            help="texttexttexttexttexttexttexttexttexttexttext")
        
        # texttexttexttexttexttext
        parser.add_argument("--step", type=str, choices=["step1", "step2", "step3", "all"], default="all",
                            help="texttexttexttext step1(texttexttexttext), step2(texttexttexttext), step3(texttexttexttext), all(texttexttexttext)")
        parser.add_argument("--input_results", type=str, default=None,
                            help="texttexttexttexttexttexttexttexttexttexttext texttextstep2textstep3 ")
        
        # texttexttexttext
        parser.add_argument("--model_type", type=str, default="api",
                            choices=["api", "local"], help="texttexttexttexttexttexttext")
        parser.add_argument("--model_name", type=str, default="gpt-4",
                            help="OpenAItexttexttexttext texttext gpt-3.5-turbo, gpt-4 ")
        parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                            help="OpenAItexttexttexttexttexttext")
        parser.add_argument("--api_key", type=str, default=None,
                            help="texttexttexttexttextAPItexttext")
        parser.add_argument("--use_local", action="store_true",
                            help="texttexttexttexttexttexttexttexttexttexttexttext")
        parser.add_argument("--local_model_path", type=str, default=None,
                            help="texttexttexttexttexttexttexttexttext")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="texttexttexttexttexttexttexttexttext")
        
        # texttexttexttext
        parser.add_argument("--http_proxy", type=str, default=None,
                            help="HTTPtexttexttexttext texttext http://127.0.0.1:7890 ")
        parser.add_argument("--https_proxy", type=str, default=None,
                            help="HTTPStexttexttexttext texttext http://127.0.0.1:7890 ")
        
        # texttexttexttext
        parser.add_argument("--log_level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help="texttexttexttext")
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line argumentstexttexttexttexttext 
        
        Args:
            args: texttexttexttexttexttexttext Nonetexttexttexttextsys.argv
            
        Returns:
            texttexttexttexttexttexttexttexttexttext
        """
        # texttexttexttext texttexttexttexttexttexttexttext
        pre_args, remaining_argv = self.parser.parse_known_args(args)
        
        # texttexttexttexttexttexttexttexttext texttexttexttexttexttexttexttexttext 
        if pre_args.config:
            config_data = self._load_config_file(pre_args.config)
            # texttexttexttexttexttexttexttext
            self.parser.set_defaults(**config_data)
        
        # texttexttexttext texttexttexttexttexttext
        self.args = self.parser.parse_args(remaining_argv)
        
        # texttexttexttext
        self._validate_args()
        
        return self.args
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """texttexttexttexttexttexttexttexttext 
        
        Args:
            config_path: texttexttexttexttexttext
            
        Returns:
            texttexttexttexttexttexttext
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"texttexttexttext {config_path} texttexttext")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("texttextYAMLtexttexttexttexttexttexttexttextPyYAML: pip install pyyaml")
        else:
            raise ValueError(f"texttexttexttexttexttexttexttexttexttext: {config_path.suffix}")
        
        logging.info(f"texttexttexttexttext {config_path} texttexttexttexttext")
        return config
    
    def _validate_args(self) -> None:
        """texttexttexttexttexttexttexttexttexttexttexttext """
        # texttextinput_dir

        # texttexttexttexttexttexttexttext texttexttexttexttexttext 
        if self.args.model_type == "local" and self.args.use_local and not self.args.local_model_path:
            logging.warning("texttexttexttexttexttexttexttexttexttexttexttexttexttext texttexttexttexttexttexttext")
    
    def get_config(self) -> argparse.Namespace:
        """texttexttexttexttexttext 
        
        Returns:
            texttexttexttexttexttexttexttexttexttext
        """
        if self.args is None:
            self.parse_args()
        return self.args
    
    def to_dict(self) -> Dict[str, Any]:
        """texttexttexttexttexttexttexttext 
        
        Returns:
            texttexttexttexttexttexttext
        """
        if self.args is None:
            self.parse_args()
        return vars(self.args)


# texttexttexttexttexttexttexttexttext texttexttexttext
config_manager = Config()


def get_config(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Load configurationtexttexttexttexttext 
    
    Args:
        args: texttexttexttexttexttexttext Nonetexttexttexttextsys.argv
        
    Returns:
        texttexttexttexttexttexttexttexttexttext
    """
    return config_manager.parse_args(args) 


class ApiKeyConfig:
    """APItexttexttexttexttext texttexttexttextAPItexttexttexttexttexttexttexttext """
    
    def __init__(self):
        """texttexttextAPItexttexttexttexttexttexttext """

        with open("api_keys.json", "r") as f:
            self.api_keys = json.load(f)

    def get_api_key(self, model_provider:str) -> str:
        """texttexttexttexttexttexttextAPItexttext 
        
        Args:
            model_provider: texttexttexttexttext
        
        Returns:
            texttexttextAPItexttext
        """

        supported_providers = ['deepseek', 'openai', 'claude']
        if model_provider not in supported_providers:
            raise ValueError(f"texttexttexttexttexttexttexttexttext: {model_provider}, texttexttexttexttexttext: {supported_providers}")
        return self.api_keys.get(model_provider)


class PromptConfig:

    
    def __init__(self):
        """texttexttexttexttexttexttexttexttexttext """
        with open("prompts.json", "r") as f:
            self.prompts = json.load(f)

    def get_system_prompt(self, prompt_name:str) -> str:
        """texttexttexttexttexttext 
        
        Args:
            prompt_name: texttexttexttext
        """
        return self.prompts.get("system_prompts").get(prompt_name)
