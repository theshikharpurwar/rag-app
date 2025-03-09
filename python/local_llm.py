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
        logger.info(f"Starting document summary generation from collection {collection_name}")

        # Debug: Check if collection exists and its configuration
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        logger.info(f"Available collections: {collection_names}")

        if collection_name not in collection_names:
            logger.error(f"Collection {collection_name} not found")
            return {
                "answer": f"Error: Collection {collection_name} not found.",
                "sources": []
            }

        # Get all points - try a different approach with query_points
        logger.info("Retrieving all points from collection")

        # First attempt with scroll to get all documents
        all_points = []
        offset = None
        limit = 100

        while True:
            points_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True
            )

            logger.info(f"Retrieved {len(points_batch)} points")
            all_points.extend(points_batch)

            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Total points retrieved: {len(all_points)}")

        if not all_points:
            # Try a fallback approach with query_points to get everything
            logger.info("No points found with scroll, trying query_points fallback")
            try:
                query_results = client.query_points(
                    collection_name=collection_name,
                    query_filter=None,  # No filter to get everything
                    limit=1000,
                    with_payload=True
                )
                all_points = query_results
                logger.info(f"Retrieved {len(all_points)} points with query_points fallback")
            except Exception as e:
                logger.error(f"Error with query_points fallback: {str(e)}")

        if not all_points:
            return {
                "answer": "I couldn't find any content to summarize. The document appears to be empty or not properly processed.",
                "sources": []
            }

        # Extract text content from points
        logger.info(f"Extracting text content from {len(all_points)} points")
        text_points = []

        for point in all_points:
            payload = point.payload if hasattr(point, 'payload') else point
            content_type = payload.get('content_type', None)

            if content_type == 'text':
                text_points.append(point)

        logger.info(f"Found {len(text_points)} text points")

        if not text_points:
            # Debug information about the points we found
            sample_payload = all_points[0].payload if hasattr(all_points[0], 'payload') else all_points[0]
            logger.info(f"Sample point payload keys: {sample_payload.keys()}")
            logger.info(f"Sample point payload: {sample_payload}")

            return {
                "answer": "I found data in the document but couldn't identify any text content. The document may contain only images or non-text elements.",
                "sources": []
            }

        # Sort text points by page number
        text_points.sort(key=lambda p: p.payload.get('page', 0) if hasattr(p, 'payload') else p.get('page', 0))

        # Group content by page
        page_summary = {}
        document_name = None

        for point in text_points:
            payload = point.payload if hasattr(point, 'payload') else point
            page = payload.get('page', 0)
            text = payload.get('text', '')

            if document_name is None and 'document_name' in payload:
                document_name = payload['document_name']

            if page not in page_summary:
                page_summary[page] = []

            page_summary[page].append(text)

        logger.info(f"Grouped content into {len(page_summary)} pages")

        # Build the summary response
        summary_parts = []
        summary_parts.append("# Document Summary\n")

        # Add document metadata if available
        if document_name:
            summary_parts.append(f"## Document: {document_name}\n")

        summary_parts.append(f"## Overview\n")
        summary_parts.append(f"This document contains {len(page_summary)} pages of content.\n")

        # Create a consolidated summary
        summary_parts.append("## Key Content By Page\n")

        # Add content from each page (up to 500 chars for each page)
        for page_num in sorted(page_summary.keys()):
            page_content = ' '.join(page_summary[page_num])

            # Extract more content for the page summary - up to 500 chars
            page_preview = page_content[:500].strip()
            if len(page_content) > 500:
                page_preview += "..."

            summary_parts.append(f"### Page {page_num}\n")
            summary_parts.append(f"{page_preview}\n")

        # Include a note about how to get more specific information
        summary_parts.append("\n## How to Use This Document\n")
        summary_parts.append("You can ask more specific questions about any topic in the document for detailed information.\n")
        summary_parts.append("Try queries like:\n")
        summary_parts.append("- 'What does the document say about [specific topic]?'\n")
        summary_parts.append("- 'Define [term] from the document'\n")
        summary_parts.append("- 'Generate sample questions from the document'\n")
        summary_parts.append("- 'List the main topics in this document'\n")

        # Format sources
        sources = []
        for i, page_num in enumerate(sorted(page_summary.keys())[:5]):  # First 5 pages as sources
            if i >= len(text_points):
                break

            point = text_points[i]
            payload = point.payload if hasattr(point, 'payload') else point

            sources.append({
                "page": payload.get('page', 0),
                "text": payload.get('text', '')[:200] + "...",
                "document": payload.get('document_name', 'Document')
            })

        logger.info(f"Summary generation complete with {len(sources)} sources")
        return {
            "answer": '\n'.join(summary_parts),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        return {
            "answer": f"Sorry, I encountered an error generating the summary: {str(e)}",
            "sources": []
        }

def generate_definition(client, collection_name, term):
    """Find and generate a definition for a specific term in the document."""
    try:
        logger.info(f"Starting definition generation for term '{term}' from collection {collection_name}")

        # Create an embedding for the term
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(f"definition of {term}").tolist()

        # Search for content related to this term using query_points instead of search
        search_results = client.query_points(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=15,  # Get more results for better context
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

        # Also look for similar terms (e.g., COCOMO vs COCOMO model)
        term_words = term.lower().split()
        similar_patterns = []

        # Create patterns for partial matches with 2 or more words
        if len(term_words) >= 2:
            for i in range(len(term_words)):
                partial_term = ' '.join(term_words[:i] + term_words[i+1:])
                if len(partial_term.split()) >= 1:  # At least one word
                    similar_patterns.append(re.compile(r'(.*\b' + re.escape(partial_term) + r'\b[^.!?]*[.!?])', re.IGNORECASE))

        definition_sentences = []
        sources = []
        found_paragraphs = set()  # Track paragraphs to avoid duplication

        # First pass: look for exact matches
        for result in search_results:
            payload = result.payload if hasattr(result, 'payload') else result
            text = payload.get('text', '')
            matches = term_pattern.findall(text)

            if matches:
                for match in matches:
                    if match not in found_paragraphs:
                        definition_sentences.append(match)
                        found_paragraphs.add(match)

            sources.append({
                "page": payload.get('page', 0),
                "text": text[:200] + "..." if len(text) > 200 else text,
                "document": payload.get('document_name', 'Document'),
                "score": getattr(result, 'score', 1.0) if hasattr(result, 'score') else 1.0
            })

        # Second pass: look for similar terms if exact matches weren't found
        if len(definition_sentences) < 3:
            for pattern in similar_patterns:
                for result in search_results:
                    payload = result.payload if hasattr(result, 'payload') else result
                    text = payload.get('text', '')
                    matches = pattern.findall(text)

                    if matches:
                        for match in matches:
                            if match not in found_paragraphs:
                                definition_sentences.append(match)
                                found_paragraphs.add(match)

                # If we have enough definitions now, break
                if len(definition_sentences) >= 5:
                    break

        # Format the answer
        if definition_sentences:
            definition_parts.append("Based on the document, here's the definition and related information about this term:\n")

            # Include full paragraphs for better context - up to 10 sentences
            for i, sentence in enumerate(definition_sentences[:10]):
                definition_parts.append(f"**{i+1}.** {sentence.strip()}\n")
        else:
            # If no exact sentence matches, use the relevant sections
            definition_parts.append("I couldn't find an explicit definition, but here's relevant information about this term:\n")

            for i, result in enumerate(search_results[:5]):
                payload = result.payload if hasattr(result, 'payload') else result
                text = payload.get('text', '')

                # Include full text for better context
                definition_parts.append(f"**Reference {i+1} (Page {payload.get('page', 0)})**: {text}\n")

        # Add an overall summary if multiple pieces of information were found
        if len(definition_sentences) > 1 or (not definition_sentences and len(search_results) > 1):
            definition_parts.append("\n## Summary\n")
            definition_parts.append(f"The term '{term}' appears to be related to {', '.join([result.payload.get('text', '')[:50].strip() + '...' for result in search_results[:3]])}. You can ask follow-up questions for more specific aspects of this concept.\n")

        return {
            "answer": '\n'.join(definition_parts),
            "sources": sources[:5]  # Limit to top 5 sources
        }

    except Exception as e:
        logger.error(f"Error generating definition: {str(e)}", exc_info=True)
        return {
            "answer": f"Sorry, I encountered an error finding the definition: {str(e)}",
            "sources": []
        }

def generate_questions(client, collection_name):
    """Generate sample questions based on the document content."""
    try:
        # Get points from the collection
        all_points = []
        offset = None
        limit = 50

        while True:
            points_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True
            )

            all_points.extend(points_batch)

            if next_offset is None or len(all_points) >= 100:
                break

            offset = next_offset

        if not all_points:
            return {
                "answer": "I couldn't find any content to generate questions from.",
                "sources": []
            }

        # Extract text content
        text_points = []
        for point in all_points:
            payload = point.payload if hasattr(point, 'payload') else point
            if payload.get('content_type') == 'text':
                text_points.append(point)

        if not text_points:
            return {
                "answer": "I couldn't find any text content to generate questions from.",
                "sources": []
            }

        # Sort by page number
        text_points.sort(key=lambda p: p.payload.get('page', 0) if hasattr(p, 'payload') else p.get('page', 0))

        # Generate questions based on content
        questions = []
        sources = []

        # Create generic question patterns
        question_patterns = [
            "What is {0}?",
            "Explain the concept of {0} in detail.",
            "How does {0} relate to {1} in the context of this document?",
            "What are the main components or characteristics of {0}?",
            "Why is {0} important in this field?",
            "Describe the process of {0} as explained in the document.",
            "What are the advantages and disadvantages of {0}?",
            "Compare and contrast {0} and {1} based on the document.",
            "How is {0} implemented or applied in practice?",
            "What challenges or limitations are associated with {0}?",
            "What is the purpose of {0} in the context of {1}?",
            "How has {0} evolved over time according to the document?",
            "What are the key principles behind {0}?",
            "How would you calculate or measure {0}?",
            "What factors influence {0}?"
        ]

        # Extract potential topics
        topics = set()
        full_text = ""

        for point in text_points:
            payload = point.payload if hasattr(point, 'payload') else point
            text = payload.get('text', '')
            full_text += " " + text

            # Add this as a source if we don't have too many
            if len(sources) < 5:
                sources.append({
                    "page": payload.get('page', 0),
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "document": payload.get('document_name', 'Document')
                })

        # Find capitalized terms or terms that might be important topics
        topic_matches = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[a-zA-Z]+){0,3})\b', full_text)
        topics.update(topic_matches)

        # Add acronyms and all-caps terms
        acronym_matches = re.findall(r'\b([A-Z]{2,})\b', full_text)
        topics.update(acronym_matches)

        # Find terms before colons (often definitions)
        colon_matches = re.findall(r'([a-zA-Z\s]+):', full_text)
        topics.update([match.strip() for match in colon_matches if len(match.strip()) > 3])

        # Find terms with numbers (like "COCOMO-II", "Type 1", etc.)
        numeric_matches = re.findall(r'\b([a-zA-Z]+[\-\s][0-9IVX]+)\b', full_text)
        topics.update(numeric_matches)

        # Filter out too short or too long topics
        filtered_topics = [topic for topic in topics if 3 < len(topic) < 30]

        # Get the most important topics based on frequency
        topic_count = {}
        for topic in filtered_topics:
            if topic.lower() in topic_count:
                topic_count[topic.lower()] += 1
            else:
                topic_count[topic.lower()] = 1

        # Sort topics by frequency
        sorted_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)
        top_topics = [item[0] for item in sorted_topics[:20]]  # Top 20 topics

        # Generate questions about the topics
        for i, pattern in enumerate(question_patterns):
            if i >= 15 or i >= len(top_topics):  # Limit to 15 questions
                break

            topic = top_topics[i]

            # Find a second topic that's different from the first
            second_topic = None
            for t in top_topics:
                if t != topic:
                    second_topic = t
                    break

            if second_topic is None:
                second_topic = topic

            # Format the question
            if "{1}" in pattern:
                question = pattern.format(topic.title(), second_topic.title())
            else:
                question = pattern.format(topic.title())

            questions.append(question)

        # Format the response
        answer_parts = []
        answer_parts.append("# Sample Questions About This Document\n")

        if questions:
            answer_parts.append("Here are some questions you could ask about the content in this document:\n")

            for i, question in enumerate(questions, 1):
                answer_parts.append(f"{i}. {question}")

            answer_parts.append("\nThese questions are based on key topics found in the document. You can ask any of these questions or formulate your own specific questions about the content.")
        else:
            answer_parts.append("I couldn't generate specific questions from the content. Try asking about specific topics in the document.")

        return {
            "answer": '\n'.join(answer_parts),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        return {
            "answer": f"Sorry, I encountered an error generating questions: {str(e)}",
            "sources": []
        }

def generate_topics(client, collection_name):
    """Extract and list the main topics covered in the document."""
    try:
        # Get content from the collection
        all_points = []
        offset = None
        limit = 50

        while True:
            points_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True
            )

            all_points.extend(points_batch)

            if next_offset is None or len(all_points) >= 100:
                break

            offset = next_offset

        if not all_points:
            return {
                "answer": "I couldn't find any content to extract topics from.",
                "sources": []
            }

        # Extract text content
        text_points = []
        for point in all_points:
            payload = point.payload if hasattr(point, 'payload') else point
            if payload.get('content_type') == 'text':
                text_points.append(point)

        if not text_points:
            return {
                "answer": "I couldn't find any text content to extract topics from.",
                "sources": []
            }

        # Concatenate content
        all_text = " ".join([point.payload.get('text', '') if hasattr(point, 'payload') else point.get('text', '') for point in text_points])

        # Extract potential topics (capitalized phrases, phrases before colons, numeric sections)
        topic_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z]?[a-zA-Z]*){0,3})\b',  # Capitalized terms
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
            for topic in sorted_topics[:30]:  # Top 30 topics
                topic_text = topic['text']
                topic_pages[topic_text] = set()

                for point in text_points:
                    payload = point.payload if hasattr(point, 'payload') else point
                    if topic_text.lower() in payload.get('text', '').lower():
                        topic_pages[topic_text].add(payload.get('page', 0))

            # Format the topic list with page references
            for i, topic in enumerate(sorted_topics[:30], 1):
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
            payload = point.payload if hasattr(point, 'payload') else point
            page = payload.get('page', 0)
            if page not in used_pages and len(sources) < 5:
                sources.append({
                    "page": page,
                    "text": payload.get('text', '')[:200] + "..." if len(payload.get('text', '')) > 200 else payload.get('text', ''),
                    "document": payload.get('document_name', 'Document')
                })
                used_pages.add(page)

        return {
            "answer": '\n'.join(answer_parts),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}", exc_info=True)
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

        # Search for similar content using query_points instead of search
        search_results = client.query_points(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=10,  # Get more results for better context
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
            payload = result.payload if hasattr(result, 'payload') else result
            content_type = payload.get('content_type', 'text')
            page = payload.get('page', 0)
            document_name = payload.get('document_name', 'Document')

            if content_type == 'text':
                text = payload.get('text', '')
                context += f"[Content {i+1}] {text}\n\n"

                sources.append({
                    "page": page,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "document": document_name,
                    "score": getattr(result, 'score', 1.0) if hasattr(result, 'score') else 1.0
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
            4. Be detailed and thorough in your response, including all relevant information
            5. Do not include information that's not in the document excerpts

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
                    timeout=15
                )

                if response.status_code == 200:
                    answer = response.json().get("response", "")
                else:
                    # Fallback to manual response if Ollama fails
                    answer = f"Based on the document, here's what I found about '{query}':\n\n"
                    for i, result in enumerate(search_results, 1):
                        payload = result.payload if hasattr(result, 'payload') else result
                        if payload.get('content_type') == 'text':
                            answer += f"**Reference {i}**: {payload.get('text', '')}\n\n"

            except Exception as e:
                logger.error(f"Error using Ollama: {str(e)}")
                # Fallback to manual response
                answer = f"Based on the document, here's what I found about '{query}':\n\n"
                for i, result in enumerate(search_results, 1):
                    payload = result.payload if hasattr(result, 'payload') else result
                    if payload.get('content_type') == 'text':
                        answer += f"**Reference {i}**: {payload.get('text', '')}\n\n"

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Based on the document, here's what I found about '{query}':\n\n"
            for i, result in enumerate(search_results, 1):
                payload = result.payload if hasattr(result, 'payload') else result
                if payload.get('content_type') == 'text':
                    answer += f"**Reference {i}**: {payload.get('text', '')}\n\n"

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
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
        logger.error(f"Error: {str(e)}", exc_info=True)
        error_response = {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": []
        }
        print(json.dumps(error_response))
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())