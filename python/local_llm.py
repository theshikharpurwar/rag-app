# D:\rag-app\python\local_llm.py

import argparse
import json
import logging
import re
import sys
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_command_type(query):
    """Detect the type of command from the user query."""
    query = query.lower()

    if any(term in query for term in ["summarize", "summary", "summarize document", "give full summary"]):
        return "summary"
    elif re.search(r"define\s+([a-z\s]+)", query):
        term = re.search(r"define\s+([a-z\s]+)", query).group(1).strip()
        return "definition", term
    elif any(term in query for term in ["create questions", "sample questions", "generate questions"]):
        return "questions"
    elif "list topics" in query or "main topics" in query or "key topics" in query:
        return "topics"
    else:
        return "regular_query"

def generate_summary(client, collection_name):
    """Generate a summary of the entire document by retrieving all content."""
    try:
        # Get all points from the collection
        points = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Adjust based on your document size
            with_payload=True,
            with_vectors=False
        )[0]

        if not points:
            return {
                "answer": "I couldn't find any content to summarize.",
                "sources": []
            }

        # Extract text content from points
        text_points = [point for point in points if point.payload.get('content_type') == 'text']

        # Sort text points by page number
        text_points.sort(key=lambda p: p.payload.get('page', 0))

        # Create a document summary
        document_content = ""
        page_summary = {}

        # Group content by page
        for point in text_points:
            page = point.payload.get('page', 0)
            text = point.payload.get('text', '')

            if page not in page_summary:
                page_summary[page] = []

            page_summary[page].append(text)

        # Build the summary response
        summary_parts = []
        summary_parts.append("# Document Summary\n")

        # Add document metadata if available
        if text_points and 'document_name' in text_points[0].payload:
            summary_parts.append(f"## Document: {text_points[0].payload['document_name']}\n")

        summary_parts.append(f"## Overview\n")
        summary_parts.append(f"This document contains {len(page_summary)} pages of content.\n")

        # Create a consolidated summary
        summary_parts.append("## Key Content By Page\n")

        # Add content from each page (limit to first sentence or two for brevity)
        for page_num in sorted(page_summary.keys()):
            page_content = ' '.join(page_summary[page_num])

            # Extract first 200 characters for the page summary
            page_preview = page_content[:200].strip()
            if len(page_content) > 200:
                page_preview += "..."

            summary_parts.append(f"**Page {page_num}**: {page_preview}\n")

        # Include a note about how to get more specific information
        summary_parts.append("\n## How to Use This Document\n")
        summary_parts.append("You can ask more specific questions about any topic in the document for detailed information.\n")
        summary_parts.append("Try queries like:\n")
        summary_parts.append("- 'What does the document say about [specific topic]?'\n")
        summary_parts.append("- 'Define [term] from the document'\n")
        summary_parts.append("- 'Generate sample questions from the document'\n")

        # Format sources
        sources = [
            {
                "page": point.payload.get('page', 0),
                "text": point.payload.get('text', '')[:150] + "...",
                "document": point.payload.get('document_name', 'Document')
            }
            for point in text_points[:5]  # Include first 5 pages as sources
        ]

        return {
            "answer": '\n'.join(summary_parts),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error generating the summary: {str(e)}",
            "sources": []
        }

def generate_definition(client, collection_name, term):
    """Find and generate a definition for a specific term in the document."""
    try:
        # Create an embedding for the term
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(f"definition of {term}").tolist()

        # Search for content related to this term
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=10,
            with_payload=True
        )

        if not search_results:
            return {
                "answer": f"I couldn't find a definition for '{term}' in the document.",
                "sources": []
            }

        # Extract and format results
        definition_parts = []
        definition_parts.append(f"# Definition of '{term}'\n")

        # Look for sentences containing the term
        term_pattern = re.compile(r'(.*\b' + re.escape(term) + r'\b[^.!?]*[.!?])', re.IGNORECASE)

        definition_sentences = []
        sources = []

        for result in search_results:
            text = result.payload.get('text', '')
            matches = term_pattern.findall(text)

            if matches:
                definition_sentences.extend(matches)

            sources.append({
                "page": result.payload.get('page', 0),
                "text": text[:150] + "...",
                "document": result.payload.get('document_name', 'Document'),
                "score": result.score
            })

        if definition_sentences:
            definition_parts.append("Based on the document, here's the definition and related information:\n")
            definition_parts.append("- " + "\n- ".join(definition_sentences[:5]))
        else:
            # If no exact sentence matches, use the relevant sections
            definition_parts.append("I couldn't find an explicit definition, but here's relevant information about this term:\n")

            for i, result in enumerate(search_results[:3]):
                definition_parts.append(f"**Reference {i+1}**: {result.payload.get('text', '')[:200]}...\n")

        return {
            "answer": '\n'.join(definition_parts),
            "sources": sources[:5]  # Limit to top 5 sources
        }

    except Exception as e:
        logger.error(f"Error generating definition: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error finding the definition: {str(e)}",
            "sources": []
        }

def generate_questions(client, collection_name):
    """Generate sample questions based on the document content."""
    try:
        # Get a sample of points from the collection
        points = client.scroll(
            collection_name=collection_name,
            limit=20,  # Get a reasonable sample
            with_payload=True
        )[0]

        if not points:
            return {
                "answer": "I couldn't find any content to generate questions from.",
                "sources": []
            }

        # Extract text content
        text_points = [point for point in points if point.payload.get('content_type') == 'text']

        if not text_points:
            return {
                "answer": "I couldn't find any text content to generate questions from.",
                "sources": []
            }

        # Sort by page number
        text_points.sort(key=lambda p: p.payload.get('page', 0))

        # Generate questions based on content
        questions = []
        sources = []

        # Create generic question patterns
        question_patterns = [
            "What is {0}?",
            "Explain the concept of {0}.",
            "How does {0} relate to {1}?",
            "What are the main components of {0}?",
            "Why is {0} important?",
            "Describe the process of {0}.",
            "What are the advantages of {0}?",
            "Compare {0} and {1}.",
            "How is {0} implemented?",
            "What are the challenges in {0}?"
        ]

        # Extract potential topics
        topics = set()

        for point in text_points:
            text = point.payload.get('text', '')

            # Find capitalized terms or terms that might be important topics
            topic_matches = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', text)
            topics.update(topic_matches)

            # Add this as a source
            sources.append({
                "page": point.payload.get('page', 0),
                "text": text[:150] + "...",
                "document": point.payload.get('document_name', 'Document')
            })

        topics = list(topics)[:15]  # Limit to 15 topics

        # Generate questions about the topics
        used_topics = set()

        for pattern in question_patterns[:10]:  # Limit to 10 questions
            if len(topics) > 0:
                topic = topics.pop(0)

                if len(topics) > 0:
                    second_topic = topics[0]  # Use the next topic for patterns with two placeholders
                else:
                    second_topic = topic

                if "{1}" in pattern:
                    question = pattern.format(topic, second_topic)
                else:
                    question = pattern.format(topic)

                questions.append(question)
                used_topics.add(topic)
                if second_topic != topic:
                    used_topics.add(second_topic)

        # Format the response
        answer_parts = []
        answer_parts.append("# Sample Questions About This Document\n")

        if questions:
            answer_parts.append("Here are some questions you could ask about the content in this document:\n")

            for i, question in enumerate(questions, 1):
                answer_parts.append(f"{i}. {question}")

            answer_parts.append("\nThese questions are based on topics found in the document. You can ask any of these or form your own questions about the content.")
        else:
            answer_parts.append("I couldn't generate specific questions from the content. Try asking about specific topics in the document.")

        return {
            "answer": '\n'.join(answer_parts),
            "sources": sources[:5]  # Limit to 5 sources
        }

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error generating questions: {str(e)}",
            "sources": []
        }

def generate_topics(client, collection_name):
    """Extract and list the main topics covered in the document."""
    try:
        # Get content from the collection
        points = client.scroll(
            collection_name=collection_name,
            limit=100,  # Get a reasonable sample
            with_payload=True
        )[0]

        if not points:
            return {
                "answer": "I couldn't find any content to extract topics from.",
                "sources": []
            }

        # Extract text content
        text_points = [point for point in points if point.payload.get('content_type') == 'text']

        if not text_points:
            return {
                "answer": "I couldn't find any text content to extract topics from.",
                "sources": []
            }

        # Concatenate content
        all_text = " ".join([point.payload.get('text', '') for point in text_points])

        # Extract potential topics (capitalized phrases, phrases before colons, numeric sections)
        topic_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b',  # Capitalized terms
            r'([^.!?:]+):',  # Phrases before colons
            r'(\d+(?:\.\d+)*)\s+([^.!?]+)',  # Numbered sections
            r'•\s*([^•\n]+)'  # Bullet points
        ]

        topics = []

        for pattern in topic_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                if isinstance(match, tuple):
                    # For tuple matches, use the last group
                    topic = match[-1]
                else:
                    topic = match

                # Clean up the topic
                topic = topic.strip()
                if len(topic) > 5 and len(topic) < 50:  # Reasonable topic length
                    topics.append(topic)

        # Remove duplicates and sort by frequency
        topic_count = {}
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower in topic_count:
                topic_count[topic_lower]['count'] += 1
            else:
                topic_count[topic_lower] = {'text': topic, 'count': 1}

        # Sort by count (descending)
        sorted_topics = sorted(topic_count.values(), key=lambda x: x['count'], reverse=True)

        # Format the response
        answer_parts = []
        answer_parts.append("# Main Topics in This Document\n")

        if sorted_topics:
            answer_parts.append("Here are the key topics covered in this document:\n")

            # Group topics by pages they appear on
            topic_pages = {}
            for topic in sorted_topics[:20]:  # Top 20 topics
                topic_text = topic['text']
                topic_pages[topic_text] = set()

                for point in text_points:
                    if topic_text.lower() in point.payload.get('text', '').lower():
                        topic_pages[topic_text].add(point.payload.get('page', 0))

            # Format the topic list with page references
            for i, topic in enumerate(sorted_topics[:20], 1):
                topic_text = topic['text']
                pages = sorted(topic_pages[topic_text])

                if pages:
                    page_str = f"(Pages: {', '.join(map(str, pages))})"
                else:
                    page_str = ""

                answer_parts.append(f"{i}. **{topic_text}** {page_str}")

        else:
            answer_parts.append("I couldn't identify specific topics in this document.")

        # Create sources from the first few pages
        sources = []
        used_pages = set()

        for point in text_points:
            page = point.payload.get('page', 0)
            if page not in used_pages and len(sources) < 5:
                sources.append({
                    "page": page,
                    "text": point.payload.get('text', '')[:150] + "...",
                    "document": point.payload.get('document_name', 'Document')
                })
                used_pages.add(page)

        return {
            "answer": '\n'.join(answer_parts),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error extracting topics: {str(e)}",
            "sources": []
        }

def process_regular_query(query, client, collection_name):
    """Process a regular query by searching for relevant content."""
    try:
        # Generate embedding for the query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query).tolist()

        # Search for similar content
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5,
            with_payload=True
        )

        if not search_results:
            return {
                "answer": "I couldn't find any relevant information. Please try a different query.",
                "sources": []
            }

        # Format the results
        context = ""
        sources = []

        for i, result in enumerate(search_results):
            content_type = result.payload.get('content_type', 'text')
            page = result.payload.get('page', 0)
            document_name = result.payload.get('document_name', 'Document')

            if content_type == 'text':
                text = result.payload.get('text', '')
                context += f"[Content {i+1}] {text}\n\n"

                sources.append({
                    "page": page,
                    "text": text[:150] + "..." if len(text) > 150 else text,
                    "document": document_name,
                    "score": result.score
                })

        # Generate an answer using Ollama (if available) or text-based response
        try:
            # Try to use Ollama for better responses
            prompt = f"""
            Based on the following document excerpts, provide a comprehensive answer to the question.

            Question: {query}

            Document excerpts:
            {context}

            Instructions:
            1. Answer the question based ONLY on the information in the document excerpts
            2. If the answer isn't in the provided excerpts, say "I don't have enough information to answer this question based on the document"
            3. Maintain bullet point formatting when appropriate
            4. Do not include information that's not in the document excerpts

            Answer:
            """

            # Try to use Ollama for response generation
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "phi",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    answer = response.json().get("response", "")
                else:
                    # Fallback to manual response if Ollama fails
                    answer = f"Based on the document, here's what I found about '{query}':\n\n"
                    for i, result in enumerate(search_results, 1):
                        if result.payload.get('content_type') == 'text':
                            answer += f"**Reference {i}**: {result.payload.get('text', '')[:200]}...\n\n"

            except Exception as e:
                logger.error(f"Error using Ollama: {str(e)}")
                # Fallback to manual response
                answer = f"Based on the document, here's what I found about '{query}':\n\n"
                for i, result in enumerate(search_results, 1):
                    if result.payload.get('content_type') == 'text':
                        answer += f"**Reference {i}**: {result.payload.get('text', '')[:200]}...\n\n"

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Based on the document, here's what I found about '{query}':\n\n"
            for i, result in enumerate(search_results, 1):
                if result.payload.get('content_type') == 'text':
                    answer += f"**Reference {i}**: {result.payload.get('text', '')[:200]}...\n\n"

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "answer": f"Sorry, I encountered an error processing your query: {str(e)}",
            "sources": []
        }

def main():
    """Main function to process the query."""
    parser = argparse.ArgumentParser(description='Process a query for RAG')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--collection_name', type=str, default='documents', help='Qdrant collection name')

    args = parser.parse_args()
    query = args.query
    collection_name = args.collection_name

    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)

        # Generate embedding for the query
        logger.info(f"Generating embedding for query: {query}")

        # Detect command type
        command = detect_command_type(query)

        # Process based on command type
        if command == "summary":
            logger.info(f"Generating document summary")
            result = generate_summary(client, collection_name)
        elif isinstance(command, tuple) and command[0] == "definition":
            term = command[1]
            logger.info(f"Generating definition for term: {term}")
            result = generate_definition(client, collection_name, term)
        elif command == "questions":
            logger.info(f"Generating sample questions")
            result = generate_questions(client, collection_name)
        elif command == "topics":
            logger.info(f"Extracting main topics")
            result = generate_topics(client, collection_name)
        else:
            logger.info(f"Searching for relevant content in collection: {collection_name}")
            result = process_regular_query(query, client, collection_name)

        # Print the result as JSON
        print(json.dumps(result))

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        error_response = {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": []
        }
        print(json.dumps(error_response))
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())