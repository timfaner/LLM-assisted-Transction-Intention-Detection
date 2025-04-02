#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="smart-contract-analyzer",
    version="0.1.0",
    description="智能合约意图分析与语义熵计算工具",
    author="NTU Research Team",
    author_email="hosen@ntu.edu.sg",
    packages=find_packages(include=["sc_analyzer", "sc_analyzer.*", 
                                   "semantic_entropy_analyzer", "semantic_entropy_analyzer.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "wandb>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "entropy-analyzer=semantic_entropy_analyzer.runner:main",
        ],
    },
) 