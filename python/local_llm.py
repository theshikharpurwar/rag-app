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
    elif "explain each topic" in query or "explain all topics" in query:
        return "explain_topics"
    else:
        return "regular_query"

def generate_summary(client, collection_name):
    """Generate a comprehensive summary of the document with complete page content."""
    logger.info(f"Generating summary for collection: {collection_name}")

    try:
        # First, verify the collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

        # Get all points from the collection
        search_result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True
        )

        points = search_result[0]
        logger.info(f"Retrieved {len(points)} points from collection")

        if not points:
            return {
                "answer": "I couldn't find any content to summarize.",
                "sources": []
            }

        # Sort points by page number
        points.sort(key=lambda x: x.payload.get('page', 0))

        # Extract text content from each page
        page_contents = []
        for point in points:
            page_num = point.payload.get('page', 0)
            text = point.payload.get('text', '')

            # Skip empty text
            if not text or text.strip() == '':
                continue

            page_contents.append({
                'page': page_num,
                'text': text.strip(),
                'document': point.payload.get('document', 'Unknown')
            })

        # Create a structured summary
        total_pages = len(page_contents)

        summary = f"Document Summary: {page_contents[0]['document']}\n\n"
        summary += f"This document contains {total_pages} pages of content.\n\n\n"
        summary += f"Key Content By Page:\n\n"

        # Add content from each page without truncation
        for page in page_contents:
            page_num = page['page']
            text = page['text']

            summary += f"Page {page_num}:\n{text}\n\n"

        # Add usage instructions
        summary += "\nHow to use this document:\n\n"
        summary += "You can ask specific questions about any topic in the document for detailed information.\n\n"
        summary += "Try queries like:\n"
        summary += "• What does the document say about [specific topic]?\n"
        summary += "• Define [term] from the document\n"
        summary += "• Generate sample questions from the document\n"
        summary += "• List the main topics in this document\n"

        # Create sources for the first few pages
        sources = [{'page': page['page'], 'document': page['document']} for page in page_contents[:5]]

        return {
            "answer": summary,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            "answer": f"I encountered an error while generating the summary: {str(e)}",
            "sources": []
        }

def clean_format_text(text):
    """Clean and format text for better readability."""
    import re
    
    # Remove reference markers
    text = re.sub(r'\*\*Reference \d+( \(Page \d+\))?\*\*:', '', text)
    
    # Remove heading markers at beginning
    text = re.sub(r'^# [^\n]+\n', '', text)
    text = re.sub(r'^## [^\n]+\n', '', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def format_bullet_points(text):
    """Ensure bullet points are properly formatted."""
    import re
    
    # Replace inline bullet points with properly formatted ones
    text = re.sub(r'([.!?])\s+[●○•]\s+', r'\1\n\n• ', text)
    text = re.sub(r'\n[●○•]\s+', '\n• ', text)
    
    # Convert circular bullets to standard bullet points
    text = text.replace('●', '•')
    text = text.replace('○', '  •')
    
    return text

def generate_definition(client, collection_name, term):
    """Find and generate a definition for a specific term in the document."""
    try:
        logger.info(f"Starting definition generation for term '{term}' from collection {collection_name}")

        # Verify collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

        # Create an embedding for the term
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(f"definition of {term}").tolist()

        # Search for content related to this term using search instead of query_points
        search_results = client.search(
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
                "document": payload.get('source', 'Document'),
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
        # Verify collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

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
            if 'text' in payload and payload.get('text'):
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
                    "document": payload.get('source', 'Document')
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
        # Verify collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

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
            if 'text' in payload and payload.get('text'):
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
                    "document": payload.get('source', 'Document')
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

def explain_topics(client, collection_name):
    """Explain each topic in the document in detail."""
    try:
        # Verify collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

        # First get the topics
        topics_result = generate_topics(client, collection_name)
        
        if "I couldn't identify specific topics" in topics_result["answer"]:
            return {
                "answer": "I couldn't identify specific topics to explain. The document may not contain clearly defined topics.",
                "sources": []
            }
            
        # Extract topic names from the result
        topic_pattern = r'\*\*(.*?)\*\*'
        topics = re.findall(topic_pattern, topics_result["answer"])
        
        if not topics:
            return {
                "answer": "I found topics but couldn't extract them properly for detailed explanation.",
                "sources": []
            }
            
        # Limit to top 10 topics
        topics = topics[:10]
        
        # For each topic, find relevant content
        explanations = []
        all_sources = []
        
        for topic in topics:
            # Create an embedding for the topic
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(f"explain {topic}").tolist()
            
            # Search for content related to this topic
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=3,  # Get top 3 results per topic
                with_payload=True
            )
            
            if search_results:
                explanation = f"## {topic}\n\n"
                
                # Combine the content from the search results
                content = []
                for result in search_results:
                    payload = result.payload if hasattr(result, 'payload') else result
                    text = payload.get('text', '')
                    if text:
                        content.append(text)
                        
                        # Add to sources if not too many
                        if len(all_sources) < 15:
                            all_sources.append({
                                "page": payload.get('page', 0),
                                "text": text[:200] + "..." if len(text) > 200 else text,
                                "document": payload.get('source', 'Document'),
                                "topic": topic
                            })
                
                if content:
                    # Join the content and add to explanations
                    explanation += "\n".join(content)
                    explanations.append(explanation)
            
        if explanations:
            answer = "# Detailed Topic Explanations\n\n"
            answer += "Here are detailed explanations for the main topics in the document:\n\n"
            answer += "\n\n".join(explanations)
            
            return {
                "answer": answer,
                "sources": all_sources[:10]  # Limit to 10 sources
            }
        else:
            return {
                "answer": "I couldn't find detailed explanations for the topics in the document.",
                "sources": []
            }
            
    except Exception as e:
        logger.error(f"Error explaining topics: {str(e)}", exc_info=True)
        return {
            "answer": f"Sorry, I encountered an error explaining the topics: {str(e)}",
            "sources": []
        }

def process_regular_query(query, client, collection_name):
    """Process a regular query by searching for relevant content."""
    logger.info(f"Processing regular query: {query}")

    try:
        # Verify collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            return {
                "answer": f"Collection '{collection_name}' does not exist.",
                "sources": []
            }

        # Generate embedding for the query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(query).tolist()
        
        if not embedding:
            return {"answer": "I couldn't generate an embedding for your query. Please try again.", "sources": []}

        # Search for similar content in Qdrant
        search_result = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=10
        )

        if not search_result:
            logger.info("No relevant content found")
            return {"answer": "I couldn't find any relevant information.", "sources": []}

        # Extract content and relevance from the search results
        context = []
        sources = []

        for hit in search_result:
            payload = hit.payload

            # Extract text - handle different possible payload structures
            text = ""
            if "text" in payload:
                text = payload["text"]
            elif "content" in payload:
                text = payload["content"]

            # Add context and source information
            if text and text.strip():  # Only add non-empty text
                context.append({
                    "text": text,
                    "score": hit.score,
                    "page": payload.get("page", "Unknown"),
                    "document": payload.get("document_name", "Unknown")
                })

                sources.append({
                    "page": payload.get("page", "Unknown"),
                    "document": payload.get("document_name", "Unknown"),
                    "score": hit.score
                })

        if not context:
            logger.info("No text content found in results")
            return {"answer": "I found matches but couldn't extract useful text content.", "sources": []}

        # Create a formatted answer from the retrieved content
        answer = format_answer(query, context)

        return {
            "answer": answer,
            "sources": sources[:5]  # Limit to top 5 sources
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"answer": f"I encountered an error while processing your query: {str(e)}", "sources": []}

def format_answer(query, context):
    """Format the answer in a clean, readable way."""
    # Sort context by relevance score
    sorted_context = sorted(context, key=lambda x: x.get("score", 0), reverse=True)

    # Extract text from the top results
    content_texts = [item["text"] for item in sorted_context[:5]]
    combined_text = "\n\n".join(content_texts)

    # Clean up the text
    answer = clean_format_text(combined_text)

    # Format bullet points properly
    answer = format_bullet_points(answer)

    return answer

def main():
    """Main function to process the query."""
    parser = argparse.ArgumentParser(description='Process a query for RAG')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--collection_name', type=str, default='documents', help='Qdrant collection name')

    args = parser.parse_args()
    query = args.query
    collection_name = args.collection_name

    try:
        # Connect to Qdrant and verify connection
        client = QdrantClient(host="localhost", port=6333)
        try:
            client.get_collections()  # Test connection
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            error_response = {
                "answer": f"Failed to connect to Qdrant database: {str(e)}",
                "sources": []
            }
            print(json.dumps(error_response))
            sys.exit(1)

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
        elif command == "explain_topics":
            logger.info(f"Explaining topics in detail")
            result = explain_topics(client, collection_name)
        else:
            logger.info(f"Processing regular query for collection: {collection_name}")
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