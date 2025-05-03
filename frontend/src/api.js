// D:\rag-app\frontend\src\api.js

// Allow configuring the API URL via environment variable at build time
// For containerized environment it will work with relative URL
const API_URL = process.env.REACT_APP_API_URL || '/api';

// Upload a PDF file
export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return { success: false, message: 'Error uploading file' };
  }
};

// Fetch all PDFs
export const fetchPDFs = async () => {
  try {
    const response = await fetch(`${API_URL}/pdfs`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    return { success: false, message: 'Error fetching PDFs' };
  }
};

// Query the RAG model
export const queryRAG = async (pdfId, query, modelPath = null) => {
  try {
    console.log(`Sending query to backend: ${query} for PDF: ${pdfId}`);
    
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdfId,
        query,
        modelPath
      }),
    });

    const data = await response.json();
    console.log('Received response from backend:', data);
    
    if (data.success) {
      return {
        answer: data.answer,
        sources: data.sources,
      };
    } else {
      console.error('API error:', data.message);
      return {
        answer: `Error: ${data.message}`,
        sources: [],
      };
    }
  } catch (error) {
    console.error('Error querying RAG:', error);
    return {
      answer: 'Error connecting to the server',
      sources: [],
    };
  }
};

// Reset the system
export const resetSystem = async () => {
  try {
    const response = await fetch(`${API_URL}/reset`, {
      method: 'POST',
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error resetting system:', error);
    return { success: false, message: 'Error resetting system' };
  }
};