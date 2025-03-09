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
    
    const data = await response.json();
    console.log('Upload response:', data);
    return data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return { success: false, message: 'Failed to upload PDF' };
  }
};

export const fetchPDFs = async () => {
  try {
    console.log('Fetching PDFs...');
    // Add timestamp parameter to prevent caching
    const cacheParam = `t=${new Date().getTime()}`;
    const response = await fetch(`${API_URL}/pdfs?${cacheParam}`);
    
    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Fetch PDFs response:', data);
    
    // Make sure we're returning the array of PDFs, not just the response object
    if (Array.isArray(data)) {
      return data;
    } else if (data.success && Array.isArray(data.pdfs)) {
      return data.pdfs;
    } else {
      console.warn('Unexpected response format:', data);
      return []; // Return empty array as fallback
    }
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    throw error; // Re-throw to allow component to handle the error
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