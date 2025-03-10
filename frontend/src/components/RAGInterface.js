// D:\rag-app\frontend\src\components\RAGInterface.js

import React, { useState, useEffect } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load PDFs when component mounts
  useEffect(() => {
    loadPDFs();
  }, []);

  // Function to load PDFs from the server
  const loadPDFs = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      console.log('Fetched PDFs response:', response);
      
      if (response.success) {
        // Make sure we're setting an array of PDFs
        setPdfs(response.pdfs || []);
        // Clear selected PDF if it's not in the list anymore
        setSelectedPdf(prev => {
          if (!prev) return null;
          const stillExists = response.pdfs.some(pdf => pdf._id === prev._id);
          return stillExists ? prev : null;
        });
      } else {
        console.error('Failed to fetch PDFs:', response.message);
        setError('Failed to fetch PDFs: ' + response.message);
        setPdfs([]); // Reset to empty array on error
      }
    } catch (err) {
      console.error('Error fetching PDFs:', err);
      setError('Error fetching PDFs: ' + (err.message || 'Unknown error'));
      setPdfs([]); // Reset to empty array on error
    } finally {
      setLoading(false);
    }
  };

  // Handler for when a PDF is uploaded
  const handlePdfUpload = () => {
    loadPDFs();
  };

  // Handler for when system is reset
  const handleReset = () => {
    setPdfs([]);
    setSelectedPdf(null);
    loadPDFs();
  };

  return (
    <div className="rag-interface">
      <h2>Multimodal RAG Application</h2>
      
      <div className="top-controls">
        <PDFUploader onUpload={handlePdfUpload} />
        <ResetButton onReset={handleReset} />
      </div>
      
      {loading && <p className="status-message loading">Loading PDFs...</p>}
      {error && <p className="status-message error">{error}</p>}
      
      <div className="main-content">
        <div className="pdf-list">
          <h3>Uploaded PDFs</h3>
          {Array.isArray(pdfs) && pdfs.length > 0 ? (
            <ul>
              {pdfs.map(pdf => (
                <li 
                  key={pdf._id} 
                  className={selectedPdf && selectedPdf._id === pdf._id ? 'selected' : ''}
                  onClick={() => setSelectedPdf(pdf)}
                >
                  <span className="pdf-name">{pdf.originalName}</span>
                  <span className="pdf-pages">{pdf.pageCount || 0} pages</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-pdfs">No PDFs uploaded yet</p>
          )}
        </div>
        
        <div className="chat-container">
          {selectedPdf ? (
            <ChatInterface pdf={selectedPdf} />
          ) : (
            <div className="select-pdf-prompt">
              <p>Select a PDF from the list to start chatting</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;