// D:\rag-app\frontend\src\api.js

const API_URL = 'http://localhost:5000/api';

// Fetch all PDFs
export const fetchPDFs = async () => {
  const response = await fetch(`${API_URL}/pdfs`);
  if (!response.ok) {
    throw new Error('Failed to fetch PDFs');
  }
  return response.json();
};

// Upload a PDF
export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('pdf', file);

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  return response.json();
};

// Process a PDF
export const processPDF = async (pdfId, modelName = 'all-MiniLM-L6-v2', collectionName = 'documents') => {
  const response = await fetch(`${API_URL}/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pdfId,
      modelName,
      collectionName,
    }),
  });

  return response.json();
};

// Query the RAG system
export const queryRAG = async (query, modelName = 'phi', collectionName = 'documents') => {
  const response = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      modelName,
      collectionName,
    }),
  });

  return response.json();
};

// Reset the system
export const resetSystem = async () => {
  const response = await fetch(`${API_URL}/reset`, {
    method: 'POST',
  });

  return response.json();
};