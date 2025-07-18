# Multimodal RAG with Nomic Embeddings (Local Models)

This upgrade transforms your PDF RAG application to support **multimodal retrieval** using Nomic's state-of-the-art embedding models that work in a shared embedding space for both text and images. **Everything runs locally** - no API keys required!

## 🚀 Features

- **Text Embeddings**: Uses `nomic-embed-text-v1.5` with proper search prefixes
- **Image Embeddings**: Uses `nomic-embed-vision-v1.5` for visual content
- **Shared Embedding Space**: Text and image embeddings are comparable
- **Multimodal Retrieval**: Queries can retrieve both relevant text and images
- **Enhanced Context**: LLM responses include both textual and visual information
- **100% Local**: No external API calls - everything runs on your machine
- **Offline Capable**: Works without internet after initial model download

## 📋 Prerequisites

1. **Qdrant**: Vector database running (localhost:6333 by default)
2. **Ollama**: Local LLM server running (localhost:11434 by default)
3. **Python Dependencies**: Install with `pip install -r requirements.txt`
4. **Internet Connection**: Only needed for initial model download

## 🛠️ Setup

### 1. Environment Variables

Set the optional environment variables (all have defaults):

```bash
# Optional (defaults shown)
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export OLLAMA_HOST_URL="http://localhost:11434"
```

**Note**: No API keys needed! Nomic models run locally.

### 2. Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 3. Setup Collections

Run the setup script to create Qdrant collections:

```bash
python setup_multimodal.py
```

### 4. Test Setup

Verify everything works:

```bash
python test_multimodal.py
```

## 📖 Usage

### Processing PDFs (Multimodal)

Use the new multimodal embedding script instead of the original:

```bash
python compute_embeddings_multimodal.py /path/to/document.pdf --pdf_id "your_mongodb_id"
```

This will:
- Extract text and chunk it appropriately
- Extract images from the PDF (minimum 100px, maximum 2048px)
- Generate text embeddings with `search_document:` prefix
- Generate image embeddings
- Store both in separate Qdrant collections with linking metadata

### Querying (Multimodal)

Use the new multimodal retrieval script:

```bash
python local_llm_multimodal.py "What does the performance graph show?" --pdf_id "your_mongodb_id"
```

Optional parameters:
- `--text_limit 5`: Number of text chunks to retrieve (default: 5)
- `--image_limit 3`: Number of images to retrieve (default: 3)
- `--history '[...]'`: Chat history as JSON

## 🏗️ Architecture

### Collections

- **documents_text**: Text embeddings (768-dimensional)
- **documents_images**: Image embeddings (768-dimensional)

### Embedding Process

1. **Text Processing**:
   - PDFs are parsed and text is extracted
   - Text is chunked with overlap for context
   - Each chunk is embedded with `search_document:` prefix
   - Stored with metadata: pdf_id, source, page, content_type, etc.

2. **Image Processing**:
   - Images are extracted from PDFs using PyMuPDF
   - Images are filtered by size (min 100px, max 2048px)
   - Images are resized if too large
   - Each image is embedded using the vision model
   - Stored with metadata: pdf_id, source, page, dimensions, filename, etc.

### Retrieval Process

1. **Query Embedding**: User query is embedded with `search_query:` prefix
2. **Multimodal Search**: Same query embedding searches both collections
3. **Context Formatting**: Results from both modalities are formatted for LLM
4. **Response Generation**: LLM generates response considering both text and image context

## 🔧 Files Added/Modified

### New Files
- `embeddings/nomic_embed.py`: Nomic embedder implementation
- `compute_embeddings_multimodal.py`: Multimodal PDF processing
- `local_llm_multimodal.py`: Multimodal RAG queries
- `setup_multimodal.py`: Environment setup script
- `test_multimodal.py`: Test script for verification

### Modified Files
- `embeddings/embed_factory.py`: Added Nomic embedder support
- `requirements.txt`: Added requests and Pillow dependencies

## 🎯 Example Usage Flow

1. **Setup**: Run `python setup_multimodal.py`
2. **Process PDF**: `python compute_embeddings_multimodal.py report.pdf --pdf_id "abc123"`
3. **Query**: `python local_llm_multimodal.py "What trends are shown in the charts?" --pdf_id "abc123"`

The system will:
- Find relevant text mentioning charts/trends
- Find actual chart images from the document
- Provide a comprehensive answer referencing both

## 🔍 Response Format

Responses include sources with type information:

```json
{
  "answer": "Based on the analysis...",
  "sources": [
    {
      "id": 1,
      "type": "text",
      "page": 5,
      "document": "report.pdf",
      "score": 0.85
    },
    {
      "id": 2,
      "type": "image",
      "page": 5,
      "document": "report.pdf",
      "score": 0.78,
      "image_filename": "abc123_page_5_img_1.jpg",
      "image_dimensions": "800x600"
    }
  ]
}
```

## 🚨 Troubleshooting

### Common Issues

1. **No NOMIC_API_KEY**: Set the environment variable
2. **Qdrant Connection Failed**: Ensure Qdrant is running on the specified host/port
3. **Ollama Not Found**: Start Ollama and pull the required model (`ollama pull gemma3:1b`)
4. **No Images Found**: Check if PDF contains extractable images (some PDFs have text-based charts)

### API Limits

- Nomic API has rate limits
- Process large PDFs in smaller batches if needed
- The system processes images and text in batches to respect limits

## 🎉 Benefits

1. **Richer Context**: Answers can reference both textual descriptions and visual elements
2. **Better Understanding**: Charts, diagrams, and images are now searchable
3. **Unified Queries**: Single query can find related text and images
4. **Enhanced Responses**: LLM has access to multimodal context for more comprehensive answers

Enjoy your enhanced multimodal RAG system! 🚀
