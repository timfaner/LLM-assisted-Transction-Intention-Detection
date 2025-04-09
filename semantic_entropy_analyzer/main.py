import argparse,pickle


from sc_analyzer.data_types import Question
from semantic_entropy_analyzer.semantic_entropy import SemanticEntropyCalculator, EntailmentModel
def main():
    """Run semantic entropy calculation from command line."""
    parser = argparse.ArgumentParser(description="计算智能合约意图的语义熵")
    
    parser.add_argument("--results_path", type=str, required=True, 
                        help="意图分析结果文件路径")
    
    args = parser.parse_args()

    with open(args.results_path, "rb") as f:
        results = pickle.load(f)

    calculator = SemanticEntropyCalculator(
        results=results,
        entailment_model=EntailmentModel()
    )
    q: Question = results[0]

    context = q["question"]
    texts = [a["answer"] for a in q["answers"]]
    log_probs = [a["token_log_likelihoods"] for a in q["answers"]]
    cluster_ids = calculator.get_semantic_ids(context, texts)
    entropy = calculator.calculate_cluster_entropy(cluster_ids, log_probs)
    print(entropy)
    

if __name__ == "__main__":
    main()