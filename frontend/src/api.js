// D:\rag-app\frontend\src\api.js

const API_URL = 'http://localhost:5000/api';

export const uploadPDF = async (file) => {
  try {
    const formData = new FormData();
    formData.append('pdf', file);
    
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return { success: false, message: 'Error uploading PDF' };
  }
};

export const fetchPDFs = async () => {
  try {
    const response = await fetch(`${API_URL}/pdfs`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    throw error;
  }
};

export const queryRAG = async (query) => {
  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query
      }),
    });
    
    return response.json();
  } catch (error) {
    console.error('Error querying RAG:', error);
    return { 
      success: false, 
      answer: 'Error querying the system', 
      error: error.message 
    };
  }
};

export const resetSystem = async () => {
  try {
    const response = await fetch(`${API_URL}/reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    return response.json();
  } catch (error) {
    console.error('Error resetting system:', error);
    return { success: false, message: 'Error resetting system' };
  }
};