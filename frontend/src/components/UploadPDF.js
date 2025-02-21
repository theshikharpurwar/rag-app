import React, { useState } from 'react';
import axios from 'axios';

function UploadPDF({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
    setError('');
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file');
      return;
    }

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      // Log upload attempt to terminal via backend
      const logResponse = await axios.post('http://localhost:5000/api/log', {
        action: `Uploading PDF: ${file.name}`,
      });

      const response = await axios.post('http://localhost:5000/api/upload/pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setMessage(response.data.message || 'PDF uploaded successfully');
      onUploadSuccess(file.name); // Use filename as PDF ID for simplicity
      setError('');
    } catch (error) {
      setError(`Error: ${error.response?.data?.error || error.message || 'Upload failed'}`);
      setMessage('');
      console.error('Upload Error:', error.response?.data || error.message);
    }
  };

  return (
    <div>
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload PDF</button>
      {message && <p style={{ color: 'green' }}>{message}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default UploadPDF;