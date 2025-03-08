// D:\rag-app\frontend\src\components\PDFUploader.js

import React, { useState } from 'react';
import { uploadPDF } from '../api';
import './PDFUploader.css';

const PDFUploader = ({ onUpload, apiKey, selectedModel }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file to upload');
      return;
    }

    if (!apiKey) {
      setError('Please enter a Mistral API key');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await uploadPDF(file, apiKey, selectedModel);
      if (response.success) {
        setFile(null);
        onUpload();
      } else {
        setError('Upload failed: ' + response.message);
      }
    } catch (err) {
      setError('Error uploading PDF: ' + (err.message || 'Unknown error'));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="pdf-uploader">
      <h3>Upload Document</h3>
      <div className="upload-controls">
        <input 
          type="file" 
          accept="application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation" 
          onChange={handleFileChange}
          disabled={uploading}
        />
        <button 
          onClick={handleUpload} 
          disabled={!file || uploading || !apiKey}
          className={!apiKey ? 'disabled' : ''}
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      {error && <p className="error-message">{error}</p>}
      {!apiKey && <p className="warning-message">API key required for upload</p>}
      {file && <p className="file-name">Selected: {file.name}</p>}
    </div>
  );
};

export default PDFUploader;