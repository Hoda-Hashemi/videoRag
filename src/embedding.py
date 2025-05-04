import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

def text_embeddings(texts, embedding_file):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        texts,convert_to_tensor=True
    )
    
    if embedding_file:
        torch.save(embeddings, embedding_file)
    return embeddings

def plot_embeddings(image_embeddings, text_embeddings):
    image_embeddings = image_embeddings.cpu().detach().numpy()
    text_embeddings = text_embeddings.cpu().detach().numpy()
    
    plt.style.use('dark_background')
    
    def reduce_and_plot(embeddings, title, color):
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        plt.figure(figsize=(6, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        plt.title(title, color='white')
        plt.xlabel("t-SNE Component 1", color='white')
        plt.ylabel("t-SNE Component 2", color='white')
        plt.grid(True, color='#444444')
        plt.tight_layout()
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
    
    reduce_and_plot(image_embeddings, "Image Embeddings (t-SNE)", "cyan")
    reduce_and_plot(text_embeddings, "Text Embeddings (t-SNE)", "lime")

