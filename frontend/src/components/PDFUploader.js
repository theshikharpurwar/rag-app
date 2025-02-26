import React, { useState } from 'react';


function PDFUploader({ onUpload, disabled }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length && files[0].type === 'application/pdf') {
      setFile(files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (file) {
      onUpload(file);
      setFile(null);
    }
  };

  return (
    <div className="pdf-uploader">
      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          id="file-input"
          disabled={disabled}
        />
        <label htmlFor="file-input">
          {file ? file.name : 'Drop PDF here or click to browse'}
        </label>
      </div>
      
      {file && (
        <button 
          className="upload-button" 
          onClick={handleUpload}
          disabled={disabled}
        >
          {disabled ? 'Uploading...' : 'Upload PDF'}
        </button>
      )}
    </div>
  );
}

export default PDFUploader;