// D:\rag-app\frontend\src\components\PDFUploader.js

import React, { useState, useRef } from 'react';
import { uploadPDF } from '../api';
import './PDFUploader.css';

const PDFUploader = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    setUploadProgress(0);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file to upload');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(10); // Start progress

      // Simulate progress during upload
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + Math.random() * 20;
          return newProgress > 90 ? 90 : newProgress; // Cap at 90% until complete
        });
      }, 500);

      const response = await uploadPDF(file);
      
      clearInterval(progressInterval);
      
      if (response.success) {
        setUploadProgress(100);
        setFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        setTimeout(() => {
          onUpload();
          setUploadProgress(0);
        }, 1000);
      } else {
        setError('Upload failed: ' + response.message);
        setUploadProgress(0);
      }
    } catch (err) {
      console.error('Error in handleUpload:', err);
      setError('Error uploading PDF');
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="pdf-uploader">
      <div className="file-input-container">
        <input 
          type="file" 
          id="pdf-upload"
          ref={fileInputRef}
          accept="application/pdf" 
          onChange={handleFileChange} 
          disabled={uploading}
          className="file-input"
        />
        <label htmlFor="pdf-upload" className="file-label">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M13,15V9H11V15H8L12,19L16,15H13M13,7V3.5L18.5,9H13V7Z" />
          </svg>
          <span>{file ? file.name : 'Choose PDF file'}</span>
        </label>
      </div>
      
      {uploadProgress > 0 && (
        <div className="progress-container">
          <div 
            className="progress-bar" 
            style={{ width: `${uploadProgress}%` }}
          ></div>
        </div>
      )}
      
      <button 
        className="upload-button btn btn-primary"
        onClick={handleUpload} 
        disabled={uploading || !file}
      >
        {uploading ? 'Uploading...' : 'Upload Document'}
      </button>
      
      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default PDFUploader;