# D:\rag-app\python\local_llm.py

import os
import sys
import json
import logging
import argparse
from qdrant_client import QdrantClient
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def search_similar_content(query, collection_name, embedder, limit=5):
    """Search for similar content in Qdrant."""
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)

        # Get embedding for query
        query_embedding = embedder.get_embedding(query)

        # Search for similar vectors
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        # Log search results for debugging
        logging.info(f"Found {len(search_result)} search results for query: '{query}'")
        if search_result and len(search_result) > 0:
            logging.info(f"First result score: {search_result[0].score}")
            logging.info(f"First result payload keys: {search_result[0].payload.keys() if search_result[0].payload else 'No payload'}")

        # Extract payload from search results
        similar_content = []
        for result in search_result:
            if result.payload:
                # Extract text with better error handling
                text = "No text available"
                for key in ['text', 'content', 'page_content', 'document_content', 'context']:
                    if key in result.payload and result.payload[key]:
                        text = result.payload[key]
                        break

                # Extract page with better error handling
                page = "Unknown"
                for key in ['page', 'page_num', 'page_number', 'index']:
                    if key in result.payload and result.payload[key]:
                        page = result.payload[key]
                        break

                similar_content.append({
                    'text': text,
                    'page': page,
                    'score': result.score
                })

        return similar_content
    except Exception as e:
        logging.error(f"Error searching similar content: {str(e)}")
        raise

def get_all_content(collection_name, limit=100):
    """Get all content from the collection for summarization."""
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)

        # Get all points from the collection
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=limit
        )

        # Log the first point's payload for debugging
        if scroll_result[0] and len(scroll_result[0]) > 0:
            logging.info(f"First point payload keys: {scroll_result[0][0].payload.keys() if scroll_result[0][0].payload else 'No payload'}")
        else:
            logging.warning("No points found in collection")

        # Extract payload from results
        all_content = []
        for point in scroll_result[0]:
            if point.payload:
                # Extract text with better error handling
                text = "No text available"
                for key in ['text', 'content', 'page_content', 'document_content', 'context']:
                    if key in point.payload and point.payload[key]:
                        text = point.payload[key]
                        break

                # Extract page with better error handling
                page = "Unknown"
                for key in ['page', 'page_num', 'page_number', 'index']:
                    if key in point.payload and point.payload[key]:
                        page = point.payload[key]
                        break

                all_content.append({
                    'text': text,
                    'page': page,
                    'id': point.id
                })

        # Sort by page number if available
        try:
            all_content.sort(key=lambda x: int(x['page']) if x['page'] and str(x['page']).isdigit() else 0)
        except Exception as e:
            logging.warning(f"Could not sort by page number: {str(e)}")

        # Log summary of content found
        logging.info(f"Total content items found: {len(all_content)}")
        if all_content:
            logging.info(f"First item example - Page: {all_content[0]['page']}, Text available: {'Yes' if all_content[0]['text'] != 'No text available' else 'No'}")

        return all_content
    except Exception as e:
        logging.error(f"Error getting all content: {str(e)}")
        raise

def generate_simple_answer(query, similar_content):
    """Generate a simple answer based on similar content without using an LLM."""
    try:
        if not similar_content:
            return "I couldn't find any relevant information to answer your query."

        # Format the answer
        answer_parts = []
        answer_parts.append(f"Here's what I found related to your query: '{query}'")
        answer_parts.append("")

        for item in similar_content:
            answer_parts.append(f"From page {item['page']}:")
            answer_parts.append(item['text'])
            answer_parts.append("")

        return "\n".join(answer_parts)
    except Exception as e:
        logging.error(f"Error generating simple answer: {str(e)}")
        return f"I couldn't generate an answer due to an error: {str(e)}"

def generate_simple_summary(content):
    """Generate a simple text-based summary without using an LLM."""
    try:
        # Check if we have any content
        if not content:
            return "No content found in the document to summarize."

        # Create a simple text-based summary
        summary_parts = []
        summary_parts.append(f"Document Summary (Total Pages: {len(content)})")
        summary_parts.append("")

        # Add table of contents
        summary_parts.append("Table of Contents:")
        for i, item in enumerate(content):
            page_text = item['text']
            if page_text == "No text available":
                summary_parts.append(f"Page {item['page']}: [No text available]")
            else:
                # Get first line or first 50 characters as a section title
                first_line = page_text.split('\n')[0] if '\n' in page_text else page_text[:50]
                summary_parts.append(f"Page {item['page']}: {first_line}...")

        summary_parts.append("")
        summary_parts.append("Content Overview:")

        # Add first paragraph from each page
        for item in content:
            page_text = item['text']
            if page_text == "No text available":
                summary_parts.append(f"Page {item['page']}: [No text available]")
            else:
                first_para = page_text.split('\n\n')[0] if '\n\n' in page_text else page_text[:200]
                summary_parts.append(f"Page {item['page']}:")
                summary_parts.append(first_para + "...")
            summary_parts.append("")

        return "\n".join(summary_parts)
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return f"I couldn't generate a summary due to an error: {str(e)}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Local LLM for RAG')
    parser.add_argument('query', type=str, help='Query to process')
    parser.add_argument('--pdf_path', type=str, help='Path to PDF file')
    parser.add_argument('--collection_name', type=str, default='documents', help='Qdrant collection name')
    parser.add_argument('--model_name', type=str, default='simple', help='LLM model name (not used in simple mode)')
    parser.add_argument('--model_path', type=str, help='Path to embedding model (not used for LLM)')

    args = parser.parse_args()

    # Initialize the embedder
    from embeddings.embed_factory import get_embedder
    embedder = get_embedder("clip", "ViT-B/32")

    try:
        # Check if this is a summary request
        is_summary_request = args.query.lower().strip() in [
            "summary", "summarize", "give summary", "provide summary",
            "full summary", "give full summary", "document summary"
        ]

        if is_summary_request:
            # Get all content for summarization
            all_content = get_all_content(args.collection_name)

            # Generate summary
            summary = generate_simple_summary(all_content)

            # Output the result as JSON
            result = {
                "success": True,
                "answer": summary,
                "sources": [{"text": f"Full document ({len(all_content)} pages)", "page": "all"}]
            }
            print(json.dumps(result))
        else:
            # Search for similar content
            similar_content = search_similar_content(args.query, args.collection_name, embedder)

            # Generate answer
            answer = generate_simple_answer(args.query, similar_content)

            # Output the result as JSON
            result = {
                "success": True,
                "answer": answer,
                "sources": similar_content
            }
            print(json.dumps(result))

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        result = {
            "success": False,
            "message": str(e)
        }
        print(json.dumps(result))
        sys.exit(1)

if __name__ == "__main__":
    main()