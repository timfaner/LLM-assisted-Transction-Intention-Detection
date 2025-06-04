#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
texttexttexttexttexttexttexttexttexttexttexttexttexttext
"""

import os
import pickle
import argparse
import glob
import statistics
from pathlib import Path


def format_analysis(text, width=100):
    """texttexttexttexttexttexttexttexttexttexttexttext"""
    if not text:
        return "texttexttexttexttext"
    
    lines = []
    for line in text.split('\n'):
        if len(line) > width:
            # texttexttexttexttexttexttexttexttexttexttext
            chunks = [line[i:i+width] for i in range(0, len(line), width)]
            lines.extend(chunks)
        else:
            lines.append(line)
    return '\n'.join(lines)


def get_latest_results_file():
    """texttexttexttexttexttexttexttexttext"""
    # texttexttexttexttexttexttexttexttexttexttext
    script_dir = Path(__file__).resolve().parent
    project_results = list(Path(script_dir / "intent_results").glob("run-*/files/results.pkl"))
    
    # texttexttexttext texttexttexttext 
    wandb_results = glob.glob('/tmp/*/sc_intent/wandb/run-*/files/results.pkl')
    
    # texttexttexttexttexttexttexttexttexttexttext
    results_files = project_results + wandb_results
    
    if not results_files:
        print("texttexttexttexttexttexttexttexttext ")
        return None
    
    # texttexttexttexttexttexttexttexttext texttexttexttexttexttexttext
    latest_file = max(results_files, key=os.path.getmtime)
    print(f"texttexttexttexttexttexttexttext: {latest_file}")
    return str(latest_file)


def format_logprobs(logprobs):
    """texttexttextlogprobtexttexttexttexttexttext"""
    if not logprobs or all(lp is None for lp in logprobs):
        return "texttexttexttextlogprobtext"
    
    # texttexttextNonetext
    valid_logprobs = [lp for lp in logprobs if lp is not None]
    
    if not valid_logprobs:
        return "texttexttexttextlogprobtext"
    
    try:
        avg = sum(valid_logprobs) / len(valid_logprobs)
        if len(valid_logprobs) > 1:
            stdev = statistics.stdev(valid_logprobs)
            return f"texttext: {avg:.4f}, texttexttext: {stdev:.4f}, texttext: [{min(valid_logprobs):.4f}, {max(valid_logprobs):.4f}]"
        else:
            return f"text: {avg:.4f}"
    except Exception as e:
        return f"texttexttexttexttexttexttext: {e}"


def view_results(results_file=None, output_file=None, max_sections=None, max_questions=None, max_answers=None, full_text=False):
    """texttexttexttexttexttexttexttexttexttext"""
    if not results_file:
        results_file = get_latest_results_file()
        if not results_file:
            return
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Unable to load result file {results_file}: {e}")
        return
    
    # texttexttexttexttexttext
    output = []
    
    # texttexttexttexttexttext
    model_info = results.get('model_info', {})
    output.append("=" * 80)
    output.append("texttexttexttext:")
    output.append("-" * 80)
    for key, value in model_info.items():
        output.append(f"  {key}: {value}")
    
    # texttexttexttexttexttexttexttext
    contracts = results.get('contract_intents', {})
    output.append("\n" + "=" * 80)
    output.append(f"texttexttext {len(contracts)} texttexttexttexttext")
    output.append("=" * 80)
    
    for contract_name, contract_data in contracts.items():
        output.append(f"\n\n{'='*40} texttext: {contract_name} {'='*40}")
        output.append(f"texttext: {contract_data.get('contract_file')}")
        output.append(f"texttexttexttext: {contract_data.get('transaction_file')}")
        output.append(f"texttexttexttext: {contract_data.get('content_length')} texttext")
        output.append(f"texttexttexttext: {contract_data.get('content_hash')}")
        
        for test_idx, test in enumerate(contract_data.get('test_results', [])):
            output.append(f"\n{'-'*30} texttexttexttext {test_idx+1} {'-'*30}")
            output.append(f"texttexttext: {test.get('timestamp')}")
            output.append(f"texttextID: {test.get('intent_id')}")
            
            # texttexttexttexttexttexttexttexttexttexttext
            intent = test.get('intent', '')
            if full_text:
                output.append("\ntexttexttexttexttexttext:")
                output.append(format_analysis(intent))
            else:
                # texttexttext200texttexttexttexttexttexttext
                intent_summary = intent[:200] + "..." if len(intent) > 200 else intent
                output.append("\ntexttexttexttext:")
                output.append(format_analysis(intent_summary))
            
            # texttexttexttext
            token_log_likelihoods = test.get('token_log_likelihoods', [])
            if token_log_likelihoods:
                avg_ll = sum(token_log_likelihoods) / len(token_log_likelihoods)
                output.append(f"\ntexttexttexttexttexttexttext:")
                output.append(f"  - texttexttexttexttexttext: {avg_ll:.4f}")
                output.append(f"  - texttokentext: {len(token_log_likelihoods)}")
            
            # texttexttexttexttexttexttexttexttext
            sections = test.get('sections', [])
            output.append(f"\ntexttexttexttexttext {len(sections)} texttexttext:")
            
            section_limit = len(sections) if max_sections is None else min(len(sections), max_sections)
            for i, section in enumerate(sections[:section_limit]):
                section_name = section.get('section_name', f'texttext{i+1}')
                output.append(f"\n{'+'*20} {section_name} {'+'*20}")
                
                # texttexttexttexttexttext
                section_content = section.get('content', '')
                section_summary = section_content[:150] + "..." if len(section_content) > 150 and not full_text else section_content
                output.append(format_analysis(section_summary))
                
                # texttexttexttexttexttext
                questions = section.get('questions', [])
                output.append(f"\ntexttexttexttexttexttext {len(questions)} texttexttext:")
                
                q_limit = len(questions) if max_questions is None else min(len(questions), max_questions)
                for q_idx, question in enumerate(questions[:q_limit]):
                    q_text = question.get('question', f'texttext{q_idx+1}')
                    output.append(f"\n[texttext {q_idx+1}] {q_text}")
                    
                    # texttexttextlogprobtexttext
                    q_logprobs = question.get('log_likelihoods', [])
                    if q_logprobs:
                        output.append(f"    LogProbtexttext: {format_logprobs(q_logprobs)}")
                    
                    # texttexttexttext
                    answers = question.get('answers', [])
                    output.append(f"    texttexttexttext {len(answers)} texttexttext:")
                    
                    a_limit = len(answers) if max_answers is None else min(len(answers), max_answers)
                    for a_idx, answer in enumerate(answers[:a_limit]):
                        a_text = answer.get('answer', '')
                        a_logprob = answer.get('avg_logprob', None)
                        
                        if full_text:
                            logprob_display = f"{a_logprob:.4f}" if a_logprob is not None else "N/A"
                            output.append(f"\n    [texttext {a_idx+1}] (logprob: {logprob_display})")
                            output.append(f"    {format_analysis(a_text)}")
                        else:
                            # texttexttexttexttexttext
                            a_summary = a_text[:100] + "..." if len(a_text) > 100 else a_text
                            logprob_display = f"{a_logprob:.4f}" if a_logprob is not None else "N/A"
                            output.append(f"    [texttext {a_idx+1}] {a_summary} (logprob: {logprob_display})")
                    
                    if len(answers) > a_limit:
                        output.append(f"    ... texttext {len(answers) - a_limit} texttexttexttexttexttext")
                
                if len(questions) > q_limit:
                    output.append(f"\n... texttext {len(questions) - q_limit} texttexttexttexttexttext")
            
            if len(sections) > section_limit:
                output.append(f"\n... texttext {len(sections) - section_limit} texttexttexttexttexttext")
    
    # texttexttexttexttexttexttexttext
    full_output = '\n'.join(output)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"texttexttexttexttexttext: {output_file}")
    else:
        print(full_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="texttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--file", type=str, help="texttextpickletexttexttexttext")
    parser.add_argument("--output", type=str, help="texttexttexttexttexttext texttexttexttexttexttexttexttexttexttext ")
    parser.add_argument("--max-sections", type=int, help="texttexttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--max-questions", type=int, help="texttexttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--max-answers", type=int, help="texttexttexttexttexttexttexttexttexttexttexttexttext")
    parser.add_argument("--full-text", action="store_true", help="texttexttexttexttexttexttexttexttexttext")
    
    args = parser.parse_args()
    view_results(
        args.file, 
        args.output, 
        args.max_sections,
        args.max_questions,
        args.max_answers,
        args.full_text
    )