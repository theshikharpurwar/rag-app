// D:\rag-app\frontend\src\config.js
// Frontend configuration - models should match python/config/models.py

// API configuration
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// LLM model configuration (should match python/config/models.py)
// Change models in python/config/models.py and update these to match
export const LLM_MODEL_NAME = 'qwen3.5:0.8b';
export const EMBEDDING_MODEL_NAME = 'nomic-embed-text:v1.5';

// TODO: Consider fetching these from backend API to ensure consistency 