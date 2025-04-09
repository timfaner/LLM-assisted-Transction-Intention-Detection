"""智能合约分析器的配置管理模块。"""
import os
import argparse
import logging
import json
from pathlib import Path
import torch
from typing import Dict, Any, Optional, List


class Config:
    """配置管理类，负责处理命令行参数和配置文件。"""
    
    def __init__(self):
        """初始化配置管理器。"""
        self.parser = self._create_arg_parser()
        self.args = None
    
    def _create_arg_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器。
        
        Returns:
            参数解析器对象
        """
        parser = argparse.ArgumentParser(description="智能合约意图分析器")
        
        # 配置文件选项
        parser.add_argument("--config", type=str, default=None,
                            help="配置文件路径（支持JSON或YAML格式）")
        
        # 输入/输出选项
        parser.add_argument("--input_dir", type=str,
                            help="包含智能合约文件夹的目录")
        parser.add_argument("--wandb_dir", type=str, default=None,
                            help="wandb文件的目录（默认: intent_results目录）")
        parser.add_argument("--max_contracts", type=int, default=0,
                            help="要处理的最大合约数量（0表示无限制）")
        parser.add_argument("--save_interval", type=int, default=10,
                            help="每处理这么多合约后保存结果（0表示禁用）")
        parser.add_argument("--num_tests", type=int, default=6,
                            help="每个合约运行的测试次数")
        
        # 步骤执行选项
        parser.add_argument("--step", type=str, choices=["step1", "step2", "step3", "all"], default="all",
                            help="执行步骤：step1(生成意图), step2(生成问题), step3(生成答案), all(完整流程)")
        parser.add_argument("--input_results", type=str, default=None,
                            help="上一步骤的结果文件路径（用于step2和step3）")
        
        # 模型选项
        parser.add_argument("--model_type", type=str, default="api",
                            choices=["api", "local"], help="使用的模型类型")
        parser.add_argument("--model_name", type=str, default="gpt-4",
                            help="OpenAI模型名称（例如，gpt-3.5-turbo, gpt-4）")
        parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                            help="OpenAI嵌入模型名称")
        parser.add_argument("--api_key", type=str, default=None,
                            help="模型服务的API密钥")
        parser.add_argument("--use_local", action="store_true",
                            help="使用本地下载的大语言模型")
        parser.add_argument("--local_model_path", type=str, default=None,
                            help="本地下载模型的路径")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="运行本地模型的设备")
        
        # 代理设置
        parser.add_argument("--http_proxy", type=str, default=None,
                            help="HTTP代理地址（例如：http://127.0.0.1:7890）")
        parser.add_argument("--https_proxy", type=str, default=None,
                            help="HTTPS代理地址（例如：http://127.0.0.1:7890）")
        
        # 日志选项
        parser.add_argument("--log_level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            help="日志级别")
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """解析命令行参数和配置文件。
        
        Args:
            args: 命令行参数列表，None表示使用sys.argv
            
        Returns:
            解析后的参数命名空间
        """
        # 第一阶段：解析配置文件参数
        pre_args, remaining_argv = self.parser.parse_known_args(args)
        
        # 从配置文件加载配置（如果指定了配置文件）
        if pre_args.config:
            config_data = self._load_config_file(pre_args.config)
            # 更新解析器默认值
            self.parser.set_defaults(**config_data)
        
        # 第二阶段：解析所有参数
        self.args = self.parser.parse_args(remaining_argv)
        
        # 验证参数
        self._validate_args()
        
        return self.args
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """从配置文件加载参数。
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            包含配置的字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("使用YAML配置文件需要安装PyYAML: pip install pyyaml")
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logging.info(f"从配置文件 {config_path} 加载了配置")
        return config
    
    def _validate_args(self) -> None:
        """验证参数的有效性和完整性。"""
        # 检查input_dir

        # 检查本地模型路径（对于本地模式）
        if self.args.model_type == "local" and self.args.use_local and not self.args.local_model_path:
            logging.warning("使用本地模型但未指定模型路径，将使用默认路径")
    
    def get_config(self) -> argparse.Namespace:
        """获取当前配置。
        
        Returns:
            解析后的参数命名空间
        """
        if self.args is None:
            self.parse_args()
        return self.args
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典。
        
        Returns:
            包含配置的字典
        """
        if self.args is None:
            self.parse_args()
        return vars(self.args)


# 创建配置管理器实例，方便导入
config_manager = Config()


def get_config(args: Optional[List[str]] = None) -> argparse.Namespace:
    """获取配置的便捷函数。
    
    Args:
        args: 命令行参数列表，None表示使用sys.argv
        
    Returns:
        解析后的参数命名空间
    """
    return config_manager.parse_args(args) 


class ApiKeyConfig:
    """API密钥配置类，负责处理API密钥的获取和验证。"""
    
    def __init__(self):
        """初始化API密钥配置管理器。"""

        with open("api_keys.json", "r") as f:
            self.api_keys = json.load(f)

    def get_api_key(self, model_provider:str) -> str:
        """获取指定模型的API密钥。
        
        Args:
            model_provider: 模型提供商
        
        Returns:
            对应的API密钥
        """

        supported_providers = ['deepseek', 'openai', 'claude']
        if model_provider not in supported_providers:
            raise ValueError(f"不支持的模型提供商: {model_provider}, 支持的提供商: {supported_providers}")
        return self.api_keys.get(model_provider)


class PromptConfig:

    
    def __init__(self):
        """初始化提示配置管理器。"""
        with open("prompts.json", "r") as f:
            self.prompts = json.load(f)

    def get_system_prompt(self, prompt_name:str) -> str:
        """获取指定提示。
        
        Args:
            prompt_name: 提示名称
        """
        return self.prompts.get("system_prompts").get(prompt_name)
