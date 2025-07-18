# D:\rag-app\python\config\validate_config.py
"""
Configuration validation script to ensure all components use consistent models.
Run this script to verify that the centralized configuration is properly applied.
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add the parent directory to the path to import from config
sys.path.append(str(Path(__file__).parent.parent))

from config.models import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, OLLAMA_HOST_URL, QDRANT_HOST, QDRANT_PORT, DEFAULT_VECTOR_SIZE

# Construct URLs from the config
OLLAMA_API_URL = f"{OLLAMA_HOST_URL}/api"
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

def check_ollama_models():
    """Check if required models are available in Ollama."""
    print("🔍 Checking Ollama models...")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check LLM model
            llm_available = any(LLM_MODEL_NAME in name for name in model_names)
            print(f"  LLM Model ({LLM_MODEL_NAME}): {'✅ Available' if llm_available else '❌ Not found'}")
            
            # Check embedding model
            embed_available = any(EMBEDDING_MODEL_NAME in name for name in model_names)
            print(f"  Embedding Model ({EMBEDDING_MODEL_NAME}): {'✅ Available' if embed_available else '❌ Not found'}")
            
            if not llm_available:
                print(f"  → Run: ollama pull {LLM_MODEL_NAME}")
            if not embed_available:
                print(f"  → Run: ollama pull {EMBEDDING_MODEL_NAME}")
                
            return llm_available and embed_available
        else:
            print(f"  ❌ Failed to connect to Ollama: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Failed to connect to Ollama: {e}")
        return False

def check_qdrant_connection():
    """Check Qdrant connection and collection configuration."""
    print("\n🔍 Checking Qdrant connection...")
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        if response.status_code == 200:
            print("  ✅ Qdrant is accessible")
            
            # Check if documents collection exists and has correct vector size
            collections = response.json().get('result', {}).get('collections', [])
            doc_collection = next((c for c in collections if c['name'] == 'documents'), None)
            
            if doc_collection:
                # Get collection info
                info_response = requests.get(f"{QDRANT_URL}/collections/documents", timeout=5)
                if info_response.status_code == 200:
                    config = info_response.json().get('result', {}).get('config', {})
                    vector_size = config.get('params', {}).get('vectors', {}).get('size', 0)
                    print(f"  Collection 'documents': ✅ Exists (vector size: {vector_size})")
                    
                    if vector_size != DEFAULT_VECTOR_SIZE:
                        print(f"  ⚠️  Warning: Vector size mismatch! Expected {DEFAULT_VECTOR_SIZE}, found {vector_size}")
                        print(f"     Consider running cleanup_qdrant.py to reset the collection")
                else:
                    print("  ❌ Could not get collection info")
            else:
                print("  ℹ️  Collection 'documents' does not exist (will be created on first use)")
            return True
        else:
            print(f"  ❌ Failed to connect to Qdrant: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Failed to connect to Qdrant: {e}")
        return False

def check_file_consistency():
    """Check that configuration is consistently applied across files."""
    print("\n🔍 Checking file consistency...")
    
    # Check docker-compose.yml
    docker_compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
    if docker_compose_path.exists():
        content = docker_compose_path.read_text()
        if LLM_MODEL_NAME in content:
            print(f"  ✅ docker-compose.yml contains LLM model: {LLM_MODEL_NAME}")
        else:
            print(f"  ⚠️  docker-compose.yml might not contain LLM model: {LLM_MODEL_NAME}")
    
    # Check frontend config.js
    frontend_config_path = Path(__file__).parent.parent.parent / "frontend" / "src" / "config.js"
    if frontend_config_path.exists():
        content = frontend_config_path.read_text()
        if LLM_MODEL_NAME in content and EMBEDDING_MODEL_NAME in content:
            print(f"  ✅ frontend/src/config.js contains both model names")
        else:
            print(f"  ⚠️  frontend/src/config.js might not contain all model names")

def main():
    """Main validation function."""
    print("🔧 RAG App Configuration Validation")
    print("=" * 50)
    print(f"LLM Model: {LLM_MODEL_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector Size: {DEFAULT_VECTOR_SIZE}")
    print(f"Ollama URL: {OLLAMA_HOST_URL}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print("=" * 50)
    
    # Run checks
    ollama_ok = check_ollama_models()
    qdrant_ok = check_qdrant_connection()
    check_file_consistency()
    
    print("\n📋 Summary:")
    if ollama_ok and qdrant_ok:
        print("  ✅ All services are ready!")
        print("  🚀 You can start the application with: docker-compose up")
    else:
        print("  ❌ Some issues found. Please address them before starting the application.")
        
    return ollama_ok and qdrant_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
