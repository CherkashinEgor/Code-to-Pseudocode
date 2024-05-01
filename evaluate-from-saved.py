import json
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# Load JSON data
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Calculate metrics
def calculate_metrics(data):
    ground_truths = data['pseudo_test']
    results = {}

    # Initialize models and scorers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Iterate over each configuration and calculate scores
    for key, translations in data.items():
        if key == 'pseudo_test':
            continue
        config_results = {}
        for method, generations in translations.items():
            method_scores = {
                'bleu': [],
                'rouge1': [],
                'rouge2': [],
                'rougeL': [],
                'semantic_similarity': []
            }

            for pred, truth in zip(generations, ground_truths):
                # BLEU score
                method_scores['bleu'].append(corpus_bleu([[truth]], [pred]))

                # Rouge scores
                rouge_scores = scorer.score(truth, pred)
                method_scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
                method_scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
                method_scores['rougeL'].append(rouge_scores['rougeL'].fmeasure)

                # Semantic similarity
                embeddings1 = model.encode(truth, convert_to_tensor=True)
                embeddings2 = model.encode(pred, convert_to_tensor=True)
                method_scores['semantic_similarity'].append(util.pytorch_cos_sim(embeddings1, embeddings2).item())

            # Average scores for each method
            for score_type in method_scores:
                method_scores[score_type] = sum(method_scores[score_type]) / len(method_scores[score_type])

            config_results[method] = method_scores
        results[key] = config_results

    return results

# Save results to JSON
def save_results(results, output_path):
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

# Main execution function
def main():
    input_path = 'saved_info/all_generations.json'
    output_path = 'saved_info/performance.json'

    data = load_data(input_path)
    results = calculate_metrics(data)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
