// D:\rag-app\frontend\src\components\PDFUploader.js

import React, { useState } from 'react';
import { uploadPDF } from '../api';
import './PDFUploader.css';

const PDFUploader = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file to upload');
      return;
    }

    setLoading(true);
    try {
      const response = await uploadPDF(file);
      if (response.success) {
        onUpload();
        setFile(null);
      } else {
        setError('Upload failed: ' + response.message);
      }
    } catch (err) {
      setError('Error uploading PDF');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pdf-uploader">
      <input type="file" accept="application/pdf" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload'}
      </button>
      {file && <p>Selected: {file.name}</p>}
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default PDFUploader;