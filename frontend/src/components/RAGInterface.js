// D:\rag-app\frontend\src\components\RAGInterface.js

import React, { useState, useEffect } from 'react';
import { fetchPDFs, uploadPDF } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load PDFs on component mount
  useEffect(() => {
    loadPDFs();
  }, []);

  // Function to load PDFs from the backend
  const loadPDFs = async () => {
    setLoading(true);
    setError(null);
    try {
      const fetchedPdfs = await fetchPDFs();
      setPdfs(fetchedPdfs);
      
      // If we had a selected PDF, find its updated version
      if (selectedPdf) {
        const updatedPdf = fetchedPdfs.find(pdf => pdf._id === selectedPdf._id);
        setSelectedPdf(updatedPdf || null);
      }
    } catch (err) {
      console.error('Error loading PDFs:', err);
      setError('Failed to load PDFs');
    } finally {
      setLoading(false);
    }
  };

  // Handle PDF upload completion
  const handleUploadComplete = () => {
    // Add a slight delay to ensure backend processing has started
    setTimeout(loadPDFs, 500);
  };

  // Handle reset completion
  const handleResetComplete = () => {
    setSelectedPdf(null);
    loadPDFs();
  };

  // Check for processing status every 5 seconds for selected PDF
  useEffect(() => {
    if (!selectedPdf) return;
    
    // Only poll if page count is 0 (still processing)
    if (selectedPdf.pageCount === 0) {
      const intervalId = setInterval(() => {
        loadPDFs();
      }, 5000);
      
      return () => clearInterval(intervalId);
    }
  }, [selectedPdf]);

  return (
    <div className="rag-interface">
      <h3>Upload and Query PDFs</h3>
      <div className="controls-container">
        <PDFUploader onUpload={handleUploadComplete} />
        <ResetButton onReset={handleResetComplete} />
      </div>
      
      {loading && <p className="loading-message">Loading PDFs...</p>}
      {error && <p className="error-message">{error}</p>}
      
      <div className="pdf-list">
        <h4>Uploaded PDFs</h4>
        {pdfs.length === 0 ? (
          <p>No PDFs uploaded yet.</p>
        ) : (
          <ul>
            {pdfs.map(pdf => (
              <li 
                key={pdf._id} 
                onClick={() => setSelectedPdf(pdf)}
                className={selectedPdf && pdf._id === selectedPdf._id ? 'selected' : ''}
              >
                {pdf.originalName} - {pdf.pageCount || 0} pages
                {pdf.pageCount === 0 && <span className="processing-tag">Processing...</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
      
      {selectedPdf && (
        <ChatInterface pdf={selectedPdf} />
      )}
    </div>
  );
};

export default RAGInterface;