import sys
import json
import argparse
from qdrant_client import QdrantClient
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='View embeddings from Qdrant')
    parser.add_argument('--collection', type=str, default='documents', help='Collection name')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of embeddings to retrieve')
    parser.add_argument('--id', type=int, help='Retrieve a specific embedding by ID')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')

    args = parser.parse_args()

    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)

        if args.id is not None:
            # Get a specific point by ID
            point = client.retrieve(
                collection_name=args.collection,
                ids=[args.id],
                with_vectors=True,
                with_payload=True
            )

            if not point:
                print(f"No embedding found with ID {args.id}")
                return 1

            print(f"Embedding ID: {args.id}")
            print(f"Payload: {json.dumps(point[0].payload, indent=2)}")
            print(f"Vector (first 10 dimensions): {point[0].vector[:10]}...")

            result = {
                "success": True,
                "id": args.id,
                "payload": point[0].payload,
                "vector": point[0].vector
            }
        else:
            # Get multiple points
            response = client.scroll(
                collection_name=args.collection,
                limit=args.limit,
                with_vectors=True,
                with_payload=True
            )

            points = response[0]

            print(f"Retrieved {len(points)} embeddings from collection '{args.collection}'")

            # Print first few embeddings
            for i, point in enumerate(points[:5]):
                print(f"\nEmbedding {i+1} (ID: {point.id}):")
                print(f"Payload: {json.dumps(point.payload, indent=2)}")
                print(f"Vector (first 10 dimensions): {point.vector[:10]}...")

            result = {
                "success": True,
                "count": len(points),
                "embeddings": [{
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                } for point in points]
            }

        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())