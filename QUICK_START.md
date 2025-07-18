# Quick Start Guide - RAG Application

## 🚀 Getting Started

### **Option 1: Quick Start (Original RAG)**
```powershell
# Navigate to project
cd d:\rag-app

# Start application
.\start.bat

# Access at http://localhost:3000
```

### **Option 2: Multimodal RAG (Local Models)**
```powershell
# Navigate to project
cd d:\rag-app

# Start with multimodal setup (no API key needed!)
.\start_multimodal.bat

# Access at http://localhost:3000
```

## 📋 Prerequisites

1. **Docker Desktop** - Must be running
2. **Ollama** - Must be running
3. **LLM Model**: Run `ollama pull gemma3:1b`
4. **For Multimodal**: No API key needed - models run locally!

## 🔧 Application URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000  
- **Qdrant Vector DB**: http://localhost:6333
- **MongoDB**: localhost:27018

## 📖 How to Use

### **Web Interface (Both Modes)**
1. Open http://localhost:3000
2. Upload a PDF using the upload interface
3. Ask questions about your document
4. Get AI-powered answers

### **Multimodal Features (Command Line)**

#### Process PDFs with multimodal embeddings:
```powershell
cd d:\rag-app\python
python compute_embeddings_multimodal.py "path\to\document.pdf" --pdf_id "mongodb_document_id"
```

#### Query with multimodal retrieval:
```powershell
python local_llm_multimodal.py "What do the charts show?" --pdf_id "mongodb_document_id"
```

## 🆚 Comparison

| Feature | Original RAG | Multimodal RAG |
|---------|-------------|----------------|
| Text Processing | ✅ | ✅ |
| Image Processing | ❌ | ✅ |
| Embedding Model | all-MiniLM-L6-v2 | Nomic text + vision |
| Shared Embedding Space | ❌ | ✅ |
| Image-Text Queries | ❌ | ✅ |
| API Required | No | Yes (Nomic) |

## 🛠 Troubleshooting

### Common Issues:
1. **Docker not running**: Start Docker Desktop
2. **Ollama not found**: Start Ollama and run `ollama pull gemma3:1b`
3. **Nomic API errors**: Check your API key and internet connection
4. **Port conflicts**: Make sure ports 3000, 5000, 6333, 27018 are available

### Check Services:
```powershell
# Check Docker
docker ps

# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/collections
```

### Stop Application:
```powershell
cd d:\rag-app
docker-compose down
```

## 🎯 Next Steps

1. **Start with Original RAG** to test basic functionality
2. **Get Nomic API key** for multimodal features
3. **Try multimodal setup** for enhanced image+text processing
4. **Upload PDFs with charts/images** to see the difference!

Happy querying! 🚀
