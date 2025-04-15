// FILE: frontend/src/components/RAGInterface.js

import React, { useState, useEffect } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';
// ModelSelector import is removed

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // selectedModel state is removed

  useEffect(() => {
    loadPDFs();
  }, []);

  const loadPDFs = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        const fetchedPdfs = response.pdfs || [];
        setPdfs(fetchedPdfs);
        // Update selectedPdf state robustly
        setSelectedPdf(prev => {
           const currentId = prev?._id;
           // Check if previous selection still exists in the new list
           const stillExists = currentId ? fetchedPdfs.some(pdf => pdf._id === currentId) : false;
           if (stillExists) {
               return prev; // Keep current selection
           } else {
               // If previous selection gone, or no selection, select first PDF if available
               return fetchedPdfs.length > 0 ? fetchedPdfs[0] : null;
           }
        });
      } else {
        setError('Failed to fetch PDFs: ' + (response.message || 'Unknown error'));
        setPdfs([]); // Clear PDFs on error
        setSelectedPdf(null); // Clear selection on error
      }
    } catch (err) {
      setError('Error fetching PDFs: ' + (err.message || 'Unknown error'));
      setPdfs([]);
      setSelectedPdf(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePdfUpload = () => {
    loadPDFs(); // Reload PDF list after upload
  };

  const handleReset = () => {
    // Clear local state immediately for better UX
    setPdfs([]);
    setSelectedPdf(null);
    // The actual reset happens via API, confirmation could be added
    // Optionally trigger loadPDFs again after reset API call returns success
  };

  return (
    <div className="rag-app">
      <header className="app-header">
        <div className="container">
          <h1>Local RAG Application</h1>
        </div>
      </header>

      <main className="container">
        <div className="controls-section">
          <div className="card upload-card">
            <h2>Upload Document</h2>
            <PDFUploader onUpload={handlePdfUpload} />
          </div>
          {/* ModelSelector card removed */}
          <div className="reset-section">
             <ResetButton onReset={handleReset} />
          </div>
        </div>

        {loading && <div className="status-message loading">Loading documents...</div>}
        {error && <div className="status-message error">{error}</div>}

        <div className="content-section">
          <div className="card documents-card">
            <h2>Uploaded Documents</h2>
            {/* PDF List Logic */}
            {(pdfs && pdfs.length > 0) ? (
              <div className="documents-list">
                {pdfs.map(pdf => (
                  <div
                    key={pdf._id}
                    className={`document-item ${selectedPdf && selectedPdf._id === pdf._id ? 'selected' : ''}`}
                    onClick={() => setSelectedPdf(pdf)}
                    title={`Select ${pdf.originalName}`} // Add title for accessibility
                  >
                     <div className="document-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" /></svg>
                     </div>
                     <div className="document-info">
                        <div className="document-name">{pdf.originalName}</div>
                        <div className="document-pages">
                           {(pdf.pageCount !== undefined && pdf.pageCount !== null) ? `${pdf.pageCount} pages` : 'Processing...'}
                        </div>
                     </div>
                  </div>
                ))}
              </div>
            ) : !loading && ( // Only show empty state if not loading
              <div className="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48"><path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" /></svg>
                <p>No documents uploaded yet</p>
              </div>
            )}
          </div>

          <div className="chat-section">
            {selectedPdf ? (
              // *** Pass only the pdf prop ***
              <ChatInterface pdf={selectedPdf} />
            ) : (
              <div className="card select-prompt-card">
                <div className="empty-state">
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48"><path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" /></svg>
                  <p>Select a document to start chatting</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

       <footer className="app-footer">
         <div className="container">
           <p>Powered by Ollama ({LLM_MODEL_NAME}) & Sentence Transformers ({EMBEDDING_MODEL_NAME})</p>
         </div>
       </footer>
    </div>
  );
};

// Define constants used in the footer or pass them down if needed
const LLM_MODEL_NAME = 'tinyllama';
const EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2';


export default RAGInterface;