// D:\rag-app\frontend\src\api.js

const API_URL = 'http://localhost:5000/api';

// Fetch all PDFs
export const fetchPDFs = async () => {
  try {
    const response = await fetch(`${API_URL}/pdfs`);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch PDFs');
    }
    
    return data.pdfs;
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    throw error;
  }
};

// Upload a PDF
export const uploadPDF = async (file, apiKey, modelName = 'mistral') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('apiKey', apiKey);
    formData.append('modelName', modelName);
    
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to upload PDF');
    }
    
    return data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    throw error;
  }
};

// Query the RAG system
export const queryRAG = async (query, apiKey, pdfId = null, modelName = 'mistral', modelPath = null) => {
  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        pdfId,
        modelName,
        modelPath,
        apiKey
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to process query');
    }
    
    return data;
  } catch (error) {
    console.error('Error querying RAG system:', error);
    throw error;
  }
};

// Reset the application (clear databases and files)
export const resetApplication = async () => {
  try {
    const response = await fetch(`${API_URL}/reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to reset application');
    }
    
    return data;
  } catch (error) {
    console.error('Error resetting application:', error);
    throw error;
  }
};

// Delete a specific PDF
export const deletePDF = async (id) => {
  try {
    const response = await fetch(`${API_URL}/pdfs/${id}`, {
      method: 'DELETE'
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to delete PDF');
    }
    
    return data;
  } catch (error) {
    console.error('Error deleting PDF:', error);
    throw error;
  }
};