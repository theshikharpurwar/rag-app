#!/usr/bin/env python
# view_embeddings.py

import sys
import json
import argparse
from qdrant_client import QdrantClient
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def get_embeddings(client, collection_name, limit=100, offset=0, specific_id=None):
    """Retrieve embeddings from Qdrant collection"""
    if specific_id is not None:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[specific_id],
            with_vectors=True,
            with_payload=True
        )
    else:
        response = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        points = response[0]

    # Extract vectors and metadata
    vectors = [np.array(point.vector) for point in points]
    metadata = [point.payload for point in points]
    ids = [point.id for point in points]

    return vectors, metadata, ids

def visualize_embeddings(vectors, metadata, ids, method='pca', output_file=None):
    """Visualize embeddings using dimensionality reduction"""
    if len(vectors) < 2:
        print("Not enough vectors to visualize")
        return

    # Convert to numpy array
    vectors_array = np.array(vectors)

    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title = '2D PCA Visualization of Document Embeddings'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), n_iter=1000)
        title = '2D t-SNE Visualization of Document Embeddings'
    else:
        print(f"Unknown visualization method: {method}")
        return

    reduced_vectors = reducer.fit_transform(vectors_array)

    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # Add labels for some points
    for i, (x, y) in enumerate(reduced_vectors):
        if i % max(1, len(vectors) // 20) == 0:  # Label ~20 points to avoid clutter
            page_num = metadata[i].get('page_num', i)
            doc_name = metadata[i].get('document_name', '').split('_')[-1]
            label = f"P{page_num}"
            if doc_name:
                label += f" ({doc_name[:10]})"
            plt.annotate(label, (x, y), fontsize=8)

    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def analyze_embeddings(vectors, metadata, ids):
    """Perform basic analysis on embeddings"""
    # Convert to numpy array
    vectors_array = np.array(vectors)

    # Calculate statistics
    mean_vector = np.mean(vectors_array, axis=0)
    std_vector = np.std(vectors_array, axis=0)
    min_vector = np.min(vectors_array, axis=0)
    max_vector = np.max(vectors_array, axis=0)

    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(vectors_array, metric='cosine')
    distance_matrix = squareform(distances)

    # Find most similar and most different pairs
    np.fill_diagonal(distance_matrix, 1.0)  # Exclude self-comparisons
    most_similar_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    most_different_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

    # Print analysis results
    print("\n=== Embedding Analysis ===")
    print(f"Number of embeddings: {len(vectors)}")
    print(f"Embedding dimension: {vectors_array.shape[1]}")
    print(f"Mean vector (first 5 dimensions): {mean_vector[:5]}...")
    print(f"Standard deviation (first 5 dimensions): {std_vector[:5]}...")
    print(f"Min values (first 5 dimensions): {min_vector[:5]}...")
    print(f"Max values (first 5 dimensions): {max_vector[:5]}...")

    print("\nSimilarity Analysis:")
    print(f"Most similar pair: ID {ids[most_similar_idx[0]]} and ID {ids[most_similar_idx[1]]}")
    print(f"  - Cosine distance: {distance_matrix[most_similar_idx]:.4f}")
    print(f"  - Pages: {metadata[most_similar_idx[0]].get('page_num', 'N/A')} and {metadata[most_similar_idx[1]].get('page_num', 'N/A')}")

    print(f"Most different pair: ID {ids[most_different_idx[0]]} and ID {ids[most_different_idx[1]]}")
    print(f"  - Cosine distance: {distance_matrix[most_different_idx]:.4f}")
    print(f"  - Pages: {metadata[most_different_idx[0]].get('page_num', 'N/A')} and {metadata[most_different_idx[1]].get('page_num', 'N/A')}")

    return {
        "count": len(vectors),
        "dimension": vectors_array.shape[1],
        "statistics": {
            "mean": mean_vector.tolist(),
            "std": std_vector.tolist(),
            "min": min_vector.tolist(),
            "max": max_vector.tolist()
        },
        "similarity": {
            "most_similar": {
                "ids": [ids[most_similar_idx[0]], ids[most_similar_idx[1]]],
                "distance": float(distance_matrix[most_similar_idx]),
                "metadata": [metadata[most_similar_idx[0]], metadata[most_similar_idx[1]]]
            },
            "most_different": {
                "ids": [ids[most_different_idx[0]], ids[most_different_idx[1]]],
                "distance": float(distance_matrix[most_different_idx]),
                "metadata": [metadata[most_different_idx[0]], metadata[most_different_idx[1]]]
            }
        }
    }

def main():
    parser = argparse.ArgumentParser(description='View and analyze embeddings from Qdrant')
    parser.add_argument('--collection', type=str, default='documents', help='Collection name')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of embeddings to retrieve')
    parser.add_argument('--offset', type=int, default=0, help='Offset for pagination')
    parser.add_argument('--id', type=int, help='Retrieve a specific embedding by ID')
    parser.add_argument('--visualize', choices=['pca', 'tsne'], help='Visualize embeddings using the specified method')
    parser.add_argument('--analyze', action='store_true', help='Perform analysis on embeddings')
    parser.add_argument('--output', type=str, help='Output directory for results')

    args = parser.parse_args()

    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)

        # Get embeddings
        vectors, metadata, ids = get_embeddings(
            client,
            args.collection,
            args.limit,
            args.offset,
            args.id
        )

        if not vectors:
            print(f"No embeddings found in collection '{args.collection}'")
            return 1

        print(f"Retrieved {len(vectors)} embeddings from collection '{args.collection}'")

        # Create output directory if needed
        if args.output:
            os.makedirs(args.output, exist_ok=True)

        # Print first few embeddings
        for i in range(min(5, len(vectors))):
            print(f"\nEmbedding {i+1} (ID: {ids[i]}):")
            print(f"Payload: {json.dumps(metadata[i], indent=2)}")
            print(f"Vector (first 10 dimensions): {vectors[i][:10]}...")

        # Visualize if requested
        if args.visualize:
            output_file = None
            if args.output:
                output_file = os.path.join(args.output, f"{args.visualize}_visualization.png")
            visualize_embeddings(vectors, metadata, ids, args.visualize, output_file)

        # Analyze if requested
        analysis_result = None
        if args.analyze:
            analysis_result = analyze_embeddings(vectors, metadata, ids)

            # Save analysis to file if output is specified
            if args.output:
                analysis_file = os.path.join(args.output, "embedding_analysis.json")
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                print(f"Analysis saved to {analysis_file}")

        # Save embeddings to file if output is specified
        if args.output:
            embeddings_file = os.path.join(args.output, "embeddings.json")
            with open(embeddings_file, 'w') as f:
                json.dump({
                    "count": len(vectors),
                    "embeddings": [{
                        "id": ids[i],
                        "metadata": metadata[i],
                        "vector": vectors[i].tolist()
                    } for i in range(len(vectors))]
                }, f, indent=2)
            print(f"Embeddings saved to {embeddings_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())