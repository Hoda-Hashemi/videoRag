#%%
import json
import matplotlib.pyplot as plt
import numpy as np
import app as st
import os

# File paths
json_files = [
    '../output/faiss_retrieval_results.json',
    '../output/ivfflat_retrieval_results.json',
    '../output/hnsw_retrieval_results.json',
    '../output/tfidf_retrieval_results.json',
    '../output/bm25_retrieval_results.json'
]
video_file = '../data/video.mp4'

# Custom color palette and method names
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
file_names = ['FAISS', 'IVFFlat', 'HNSW', 'TF-IDF', 'BM25']

# Rating emoji function
def get_rating_emoji(distance):
    if distance <= 0.3:
        return ':smile:'  # Great
    elif distance <= 0.5:
        return ':neutral:'  # Good
    else:
        return ':frown:'  # Okay

# Initialize data
question_data = {}  # {q_id: [(method, distance, color, chunk, timestamp), ...]}
question_texts = {}  # {q_id: question_text}
top_segments = {}  # {q_id: {question, method, chunk, distance, timestamp}}

# Load and process JSON files
for idx, file_path in enumerate(json_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            if 'id' not in entry or 'question' not in entry or 'distances' not in entry or not entry['distances'] or 'retrieved_chunks' not in entry or 'timestamp' not in entry:
                print(f"Warning: Skipping invalid entry in {file_path} (missing required fields)")
                continue
            q_id = entry['id']
            question = entry['question']
            distance = entry['distances'][0]
            chunk = entry['retrieved_chunks'][0]
            timestamp = entry['timestamp']
            
            if q_id not in question_texts:
                question_texts[q_id] = question
            if q_id not in question_data:
                question_data[q_id] = []
            question_data[q_id].append((file_names[idx], distance, colors[idx], chunk, timestamp))
        
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Skipping.")
        continue
    except json.JSONDecodeError:
        print(f"Warning: File {file_path} contains invalid JSON. Skipping.")
        continue

# Check for valid data
if not question_data:
    print("Error: No valid data found in any JSON files.")
    exit()

# Find top segment
for q_id in question_data:
    min_entry = min(question_data[q_id], key=lambda x: x[1])  # Lowest distance
    top_segments[q_id] = {
        'question': question_texts[q_id],
        'method': min_entry[0],
        'chunk': min_entry[3],
        'distance': min_entry[1],
        'timestamp': min_entry[4]
    }

# Save top segments
with open('../output/top_segments.json', 'w') as f:
    json.dump(top_segments, f, indent=4)
print("Saved top segments to '../output/top_segments.json'")

# Plotting
question_ids = sorted(question_data.keys())[:15]
fig, axes = plt.subplots( 5,3, figsize=(20, 12), sharey=True)
axes = axes.flatten()

for idx, q_id in enumerate(question_ids):
    if idx >= 15:
        break
    
    file_names_for_q = [item[0] for item in question_data[q_id]]
    distances = [item[1] for item in question_data[q_id]]
    line_colors = [item[2] for item in question_data[q_id]]
    
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    
    axes[idx].plot(range(len(file_names_for_q)), distances, 
                   linestyle='-', linewidth=2, marker='o', markersize=8, 
                   color=line_colors[0], alpha=0.85)
    axes[idx].plot(min_index, min_distance, marker='o', markersize=12, 
                   color=line_colors[min_index], markeredgecolor='black')
    
    for x, y in enumerate(distances):
        emoji = get_rating_emoji(y)
        axes[idx].text(x, y + 0.01, f'{y:.3f}\n{emoji}', 
                       ha='center', va='bottom', fontsize=8)
    
    question_text = question_texts.get(q_id, f'Question ID {q_id}')
    axes[idx].set_title(question_text, fontsize=10, pad=10, wrap=True)
    axes[idx].set_xticks(range(len(file_names_for_q)))
    axes[idx].set_xticklabels(file_names_for_q, rotation=45, ha='right', fontsize=8)
    axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if idx % 5 == 0:
        axes[idx].set_ylabel('Cosine Distance', fontsize=10)

for idx in range(len(question_ids), 15):
    axes[idx].set_visible(False)

fig.suptitle('First Cosine Similarity Across Retrieval Methods (Lower = More Similar)', fontsize=16, y=1.05)
legend_handles = [plt.Line2D([0], [0], color=colors[i], linewidth=2, marker='o', markersize=8, alpha=0.85) 
                  for i in range(len(file_names))]
fig.legend(legend_handles, file_names, title='Retrieval Methods', 
           loc='upper center', ncol=5, fontsize=10, title_fontsize=12, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.01, 'Rating: :smile: (≤0.3), :neutral: (0.3-0.5), :frown: (>0.5)', 
         ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('cosine_similarity_15_subplots_lines.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Generated plot with {min(len(question_ids), 15)} subplots for question IDs: {question_ids}")

# %%



