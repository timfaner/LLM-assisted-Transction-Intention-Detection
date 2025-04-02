#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查看智能合约意图分析结果脚本
"""

import os
import pickle
import argparse
import glob
import statistics
from pathlib import Path


def format_analysis(text, width=100):
    """格式化分析文本以便于阅读"""
    if not text:
        return "未提供内容"
    
    lines = []
    for line in text.split('\n'):
        if len(line) > width:
            # 对过长的行进行换行处理
            chunks = [line[i:i+width] for i in range(0, len(line), width)]
            lines.extend(chunks)
        else:
            lines.append(line)
    return '\n'.join(lines)


def get_latest_results_file():
    """获取最新的结果文件"""
    # 查找结果文件的可能位置
    script_dir = Path(__file__).resolve().parent
    project_results = list(Path(script_dir / "intent_results").glob("run-*/files/results.pkl"))
    
    # 传统路径（向后兼容）
    wandb_results = glob.glob('/tmp/*/sc_intent/wandb/run-*/files/results.pkl')
    
    # 合并所有找到的结果文件
    results_files = project_results + wandb_results
    
    if not results_files:
        print("未找到任何结果文件！")
        return None
    
    # 按文件修改时间排序，获取最新的文件
    latest_file = max(results_files, key=os.path.getmtime)
    print(f"找到最新结果文件: {latest_file}")
    return str(latest_file)


def format_logprobs(logprobs):
    """格式化logprob值的统计信息"""
    if not logprobs or all(lp is None for lp in logprobs):
        return "无可用的logprob值"
    
    # 过滤掉None值
    valid_logprobs = [lp for lp in logprobs if lp is not None]
    
    if not valid_logprobs:
        return "无可用的logprob值"
    
    try:
        avg = sum(valid_logprobs) / len(valid_logprobs)
        if len(valid_logprobs) > 1:
            stdev = statistics.stdev(valid_logprobs)
            return f"平均: {avg:.4f}, 标准差: {stdev:.4f}, 范围: [{min(valid_logprobs):.4f}, {max(valid_logprobs):.4f}]"
        else:
            return f"值: {avg:.4f}"
    except Exception as e:
        return f"计算统计时出错: {e}"


def view_results(results_file=None, output_file=None, max_sections=None, max_questions=None, max_answers=None, full_text=False):
    """查看智能合约分析结果"""
    if not results_file:
        results_file = get_latest_results_file()
        if not results_file:
            return
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"无法加载结果文件 {results_file}: {e}")
        return
    
    # 准备输出内容
    output = []
    
    # 输出模型信息
    model_info = results.get('model_info', {})
    output.append("=" * 80)
    output.append("模型信息:")
    output.append("-" * 80)
    for key, value in model_info.items():
        output.append(f"  {key}: {value}")
    
    # 输出合约分析结果
    contracts = results.get('contract_intents', {})
    output.append("\n" + "=" * 80)
    output.append(f"分析了 {len(contracts)} 个智能合约")
    output.append("=" * 80)
    
    for contract_name, contract_data in contracts.items():
        output.append(f"\n\n{'='*40} 合约: {contract_name} {'='*40}")
        output.append(f"文件: {contract_data.get('contract_file')}")
        output.append(f"交易数据: {contract_data.get('transaction_file')}")
        output.append(f"合约大小: {contract_data.get('content_length')} 字节")
        output.append(f"合约哈希: {contract_data.get('content_hash')}")
        
        for test_idx, test in enumerate(contract_data.get('test_results', [])):
            output.append(f"\n{'-'*30} 意图分析 {test_idx+1} {'-'*30}")
            output.append(f"时间戳: {test.get('timestamp')}")
            output.append(f"意图ID: {test.get('intent_id')}")
            
            # 输出完整意图内容或摘要
            intent = test.get('intent', '')
            if full_text:
                output.append("\n完整意图分析:")
                output.append(format_analysis(intent))
            else:
                # 截取前200个字符作为摘要
                intent_summary = intent[:200] + "..." if len(intent) > 200 else intent
                output.append("\n意图摘要:")
                output.append(format_analysis(intent_summary))
            
            # 统计信息
            token_log_likelihoods = test.get('token_log_likelihoods', [])
            if token_log_likelihoods:
                avg_ll = sum(token_log_likelihoods) / len(token_log_likelihoods)
                output.append(f"\n模型置信度统计:")
                output.append(f"  - 平均对数似然: {avg_ll:.4f}")
                output.append(f"  - 总token数: {len(token_log_likelihoods)}")
            
            # 输出各个部分的内容
            sections = test.get('sections', [])
            output.append(f"\n意图分解为 {len(sections)} 个部分:")
            
            section_limit = len(sections) if max_sections is None else min(len(sections), max_sections)
            for i, section in enumerate(sections[:section_limit]):
                section_name = section.get('section_name', f'部分{i+1}')
                output.append(f"\n{'+'*20} {section_name} {'+'*20}")
                
                # 输出部分内容
                section_content = section.get('content', '')
                section_summary = section_content[:150] + "..." if len(section_content) > 150 and not full_text else section_content
                output.append(format_analysis(section_summary))
                
                # 输出部分问题
                questions = section.get('questions', [])
                output.append(f"\n该部分生成了 {len(questions)} 个问题:")
                
                q_limit = len(questions) if max_questions is None else min(len(questions), max_questions)
                for q_idx, question in enumerate(questions[:q_limit]):
                    q_text = question.get('question', f'问题{q_idx+1}')
                    output.append(f"\n[问题 {q_idx+1}] {q_text}")
                    
                    # 问题的logprob统计
                    q_logprobs = question.get('log_likelihoods', [])
                    if q_logprobs:
                        output.append(f"    LogProb统计: {format_logprobs(q_logprobs)}")
                    
                    # 输出答案
                    answers = question.get('answers', [])
                    output.append(f"    该问题有 {len(answers)} 个答案:")
                    
                    a_limit = len(answers) if max_answers is None else min(len(answers), max_answers)
                    for a_idx, answer in enumerate(answers[:a_limit]):
                        a_text = answer.get('answer', '')
                        a_logprob = answer.get('avg_logprob', None)
                        
                        if full_text:
                            logprob_display = f"{a_logprob:.4f}" if a_logprob is not None else "N/A"
                            output.append(f"\n    [答案 {a_idx+1}] (logprob: {logprob_display})")
                            output.append(f"    {format_analysis(a_text)}")
                        else:
                            # 截取答案摘要
                            a_summary = a_text[:100] + "..." if len(a_text) > 100 else a_text
                            logprob_display = f"{a_logprob:.4f}" if a_logprob is not None else "N/A"
                            output.append(f"    [答案 {a_idx+1}] {a_summary} (logprob: {logprob_display})")
                    
                    if len(answers) > a_limit:
                        output.append(f"    ... 还有 {len(answers) - a_limit} 个答案未显示")
                
                if len(questions) > q_limit:
                    output.append(f"\n... 还有 {len(questions) - q_limit} 个问题未显示")
            
            if len(sections) > section_limit:
                output.append(f"\n... 还有 {len(sections) - section_limit} 个部分未显示")
    
    # 输出到终端或文件
    full_output = '\n'.join(output)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"结果已保存到: {output_file}")
    else:
        print(full_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看智能合约意图分析结果")
    parser.add_argument("--file", type=str, help="结果pickle文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径（如不指定则输出到终端）")
    parser.add_argument("--max-sections", type=int, help="每个意图最多显示的部分数量")
    parser.add_argument("--max-questions", type=int, help="每个部分最多显示的问题数量")
    parser.add_argument("--max-answers", type=int, help="每个问题最多显示的答案数量")
    parser.add_argument("--full-text", action="store_true", help="显示完整文本而非摘要")
    
    args = parser.parse_args()
    view_results(
        args.file, 
        args.output, 
        args.max_sections,
        args.max_questions,
        args.max_answers,
        args.full_text
    )