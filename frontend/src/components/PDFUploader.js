// D:\rag-app\frontend\src\components\PDFUploader.js

import React, { useState } from 'react';
import { uploadPDF } from '../api';
import './PDFUploader.css';

const PDFUploader = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file to upload');
      return;
    }
    try {
      setUploading(true);
      const response = await uploadPDF(file);
      if (response.success) {
        onUpload();
        setFile(null);
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
      <div className="file-input-container">
        <input 
          type="file" 
          accept="application/pdf" 
          onChange={handleFileChange} 
          className="file-input" 
          id="pdf-file-input"
          disabled={uploading}
        />
        <label htmlFor="pdf-file-input" className="file-input-label">
          {file ? file.name : 'Choose PDF file...'}
        </label>
      </div>
      
      <button 
        onClick={handleUpload} 
        className="upload-button"
        disabled={!file || uploading}
      >
        {uploading ? 'Uploading...' : 'Upload PDF'}
      </button>
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default PDFUploader;