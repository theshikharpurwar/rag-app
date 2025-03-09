// D:\rag-app\frontend\src\api.js

const API_URL = 'http://localhost:5000/api';

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('pdf', file);
  
  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return { success: false, message: 'Failed to upload PDF' };
  }
};

export const fetchPDFs = async () => {
  try {
    const response = await fetch(`${API_URL}/pdfs`);
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    return { success: false, message: 'Failed to fetch PDFs' };
  }
};

export const queryRAG = async (pdfId, query) => {
  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ pdfId, query }),
    });
    
    return await response.json();
  } catch (error) {
    console.error('Error querying RAG:', error);
    return { success: false, message: 'Failed to query RAG' };
  }
};

export const resetSystem = async () => {
  try {
    const response = await fetch(`${API_URL}/reset`, {
      method: 'POST',
    });
    
    return await response.json();
  } catch (error) {
    console.error('Error resetting system:', error);
    return { success: false, message: 'Failed to reset system' };
  }
};