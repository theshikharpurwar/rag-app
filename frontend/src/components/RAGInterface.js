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

  useEffect(() => {
    loadPDFs();
  }, []);

  const loadPDFs = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        setPdfs(response.pdfs || []);
        setSelectedPdf(prev => {
          if (!prev) return null;
          const stillExists = response.pdfs.some(pdf => pdf._id === prev._id);
          return stillExists ? prev : null;
        });
      } else {
        setError('Failed to fetch PDFs: ' + response.message);
        setPdfs([]);
      }
    } catch (err) {
      setError('Error fetching PDFs: ' + (err.message || 'Unknown error'));
      setPdfs([]);
    } finally {
      setLoading(false);
    }
  };

  const handlePdfUpload = () => {
    loadPDFs();
  };

  const handleReset = () => {
    setPdfs([]);
    setSelectedPdf(null);
    loadPDFs();
  };

  return (
    <div className="rag-app">
      <header className="app-header">
        <div className="container">
          <h1>Multimodal RAG Application</h1>
        </div>
      </header>
      
      <main className="container">
        <div className="controls-section">
          <div className="card upload-card">
            <h2>Upload Document</h2>
            <PDFUploader onUpload={handlePdfUpload} />
          </div>
          
          <div className="reset-section">
            <ResetButton onReset={handleReset} />
          </div>
        </div>
        
        {loading && <div className="status-message loading">Loading documents...</div>}
        {error && <div className="status-message error">{error}</div>}
        
        <div className="content-section">
          <div className="card documents-card">
            <h2>Uploaded Documents</h2>
            {Array.isArray(pdfs) && pdfs.length > 0 ? (
              <div className="documents-list">
                {pdfs.map(pdf => (
                  <div 
                    key={pdf._id} 
                    className={`document-item ${selectedPdf && selectedPdf._id === pdf._id ? 'selected' : ''}`}
                    onClick={() => setSelectedPdf(pdf)}
                  >
                    <div className="document-icon">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                      </svg>
                    </div>
                    <div className="document-info">
                      <div className="document-name">{pdf.originalName}</div>
                      <div className="document-pages">{pdf.pageCount || 0} pages</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48">
                  <path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
                </svg>
                <p>No documents uploaded yet</p>
              </div>
            )}
          </div>
          
          <div className="chat-section">
            {selectedPdf ? (
              <ChatInterface pdf={selectedPdf} model="phi-2" />
            ) : (
              <div className="card select-prompt-card">
                <div className="empty-state">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48">
                    <path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" />
                  </svg>
                  <p>Select a document to start chatting</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      
      <footer className="app-footer">
        <div className="container">
          <p>Powered by Ollama & Sentence Transformers</p>
        </div>
      </footer>
    </div>
  );
};

export default RAGInterface;