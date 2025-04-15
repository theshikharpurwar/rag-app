// FILE: frontend/src/components/PDFUploader.js (Full Code)

import React, { useState, useRef } from 'react';
import { uploadPDF } from '../api'; // Assumes api.js handles the fetch
import './PDFUploader.css';

const PDFUploader = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  // Remove uploadProgress state as backend now waits - use simple 'uploading' state
  // const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
        setFile(selectedFile);
        setError(null); // Clear error on new file selection
    } else {
        setFile(null);
        setError("Please select a valid PDF file.");
        if (fileInputRef.current) {
            fileInputRef.current.value = ''; // Clear the input field
        }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file to upload');
      return;
    }

    setUploading(true);
    setError(null);
    // Remove progress simulation - backend handles the wait now

    try {
      // uploadPDF now waits for backend processing
      const response = await uploadPDF(file);

      if (response.success && response.pdf) {
        logger.info(`Upload and processing successful for ${response.pdf.originalName}`);
        setFile(null); // Clear the selected file state
        if (fileInputRef.current) {
          fileInputRef.current.value = ''; // Clear the actual file input element
        }
        // Call the callback passed from parent (RAGInterface) to refresh the list
        if (onUpload) {
          onUpload(); // This should trigger loadPDFs in RAGInterface
        }
      } else {
        // Handle failure response from backend
        const errorMessage = `Upload failed: ${response.message || 'Unknown error from server'}`;
        logger.error(errorMessage);
        setError(errorMessage);
      }
    } catch (err) {
      // Handle network errors or other exceptions during the fetch
      logger.error('Error during handleUpload:', err);
      const networkError = `Error uploading file: ${err.message || 'Network error or server unreachable'}`;
      setError(networkError);
    } finally {
      setUploading(false); // Set uploading false whether success or failure
    }
  };

  // Helper logger (can be removed if console is sufficient)
  const logger = {
      info: console.log,
      warn: console.warn,
      error: console.error
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
        <label htmlFor="pdf-upload" className={`file-label ${uploading ? 'disabled' : ''}`}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M13,15V9H11V15H8L12,19L16,15H13M13,7V3.5L18.5,9H13V7Z" />
          </svg>
          <span>{file ? file.name : 'Choose PDF file'}</span>
        </label>
      </div>

      {/* Remove determinate progress bar, show generic loading state */}
      {/* {uploadProgress > 0 && (...) } */}

      <button
        className="upload-button btn btn-primary"
        onClick={handleUpload}
        disabled={uploading || !file} // Disable if uploading or no file
      >
        {uploading ? 'Processing...' : 'Upload & Process'} {/* Update button text */}
      </button>

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default PDFUploader;