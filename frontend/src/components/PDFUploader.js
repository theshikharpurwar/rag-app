// D:\rag-app\frontend\src\components\PDFUploader.js

import React, { useState } from 'react';
import { uploadPDF } from '../api';
import './PDFUploader.css';

const PDFUploader = ({ onUpload }) => {
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

    try {
      setUploading(true);
      const response = await uploadPDF(file);
      if (response.success) {
        setFile(null);
        // Reset file input
        document.getElementById('pdf-upload').value = '';
        onUpload();
      } else {
        setError('Upload failed: ' + response.message);
      }
    } catch (err) {
      console.error('Error in handleUpload:', err);
      setError('Error uploading PDF');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="pdf-uploader">
      <input 
        type="file" 
        id="pdf-upload"
        accept="application/pdf" 
        onChange={handleFileChange} 
        disabled={uploading}
      />
      <button 
        onClick={handleUpload} 
        disabled={uploading || !file}
      >
        {uploading ? 'Uploading...' : 'Upload'}
      </button>
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default PDFUploader;