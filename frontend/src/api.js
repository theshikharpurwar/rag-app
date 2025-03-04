// D:\rag-app\frontend\src\api.js

import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('pdf', file);
  
  try {
    const response = await axios.post(`${API_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    throw error;
  }
};

export const getPDFs = async () => {
  try {
    const response = await axios.get(`${API_URL}/pdfs`);
    return response.data;
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    throw error;
  }
};

export const getPDFById = async (id) => {
  try {
    const response = await axios.get(`${API_URL}/pdfs/${id}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching PDF:', error);
    throw error;
  }
};

export const queryRAG = async (pdfId, query, modelName, modelPath) => {
  try {
    const response = await axios.post(`${API_URL}/query`, {
      pdfId,
      query,
      modelName,
      modelPath
    });
    return response.data;
  } catch (error) {
    console.error('Error querying PDF:', error);
    throw error;
  }
};