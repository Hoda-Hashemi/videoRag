import json
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

def load_golden_test_set(golden_test_file):
    print("Loading the Queries ... ")
    with open(golden_test_file, 'r') as f:
        golden_queries = json.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Encode queries and ensure output is on CPU
    query_embeddings = model.encode(
        [q['question'] for q in golden_queries],
        convert_to_tensor=True,
        device='cpu'
    )
    print("Queries are Loaded")
    return golden_queries, query_embeddings

def load_retrieval_results(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results

def plot_similarity_scores(results, output_file, k=5):
    for result in results:
        if 'distances' not in result or not isinstance(result['distances'], list):
            print(f"Error: Query {result.get('id', 'unknown')} missing valid distances.")
            return
        if len(result['distances']) != k:
            print(f"Error: Query {result['id']} has {len(result['distances'])} distances, expected {k}.")
            return
    
    num_queries = len(results)
    cols = 3
    rows = (num_queries + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        distances = result['distances']
        label = f"Q{result['id']}: {result['question'][:50]}{'...' if len(result['question']) > 50 else ''}"
        
        axes[i].bar(range(1, len(distances) + 1), distances)
        axes[i].set_title(label, fontsize=10)
        axes[i].set_xlabel('Rank')
        axes[i].set_ylabel('Similarity')
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(range(1, len(distances) + 1))
    
    for i in range(num_queries, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Similarity plot saved to {output_file}")
    
