// D:\rag-app\frontend\src\components\RAGInterface.js

import React, { useState, useEffect } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ModelSelector from './ModelSelector';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [selectedModel, setSelectedModel] = useState('phi');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPDFs()
      .then(data => {
        setPdfs(data);
      })
      .catch(err => {
        setError('Error fetching PDFs');
      });
  }, []);

  const handlePdfUpload = () => {
    setLoading(true);
    setError(null);
    
    fetchPDFs()
      .then(data => {
        setPdfs(data);
        setLoading(false);
      })
      .catch(err => {
        setError('Error fetching PDFs');
        setLoading(false);
      });
  };

  const handleReset = () => {
    setSelectedPdf(null);
    handlePdfUpload();
  };

  return (
    <div className="rag-interface">
      <h2>Local RAG System</h2>
      <div className="controls">
        <PDFUploader onUpload={handlePdfUpload} />
        <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
        <ResetButton onReset={handleReset} />
      </div>
      
      {loading && <p className="loading">Loading PDFs...</p>}
      {error && <p className="error">{error}</p>}
      
      <div className="pdf-list">
        <h3>Uploaded PDFs</h3>
        {pdfs.length === 0 ? (
          <p>No PDFs uploaded yet</p>
        ) : (
          <ul>
            {pdfs.map(pdf => (
              <li 
                key={pdf._id} 
                onClick={() => setSelectedPdf(pdf)}
                className={selectedPdf && selectedPdf._id === pdf._id ? 'selected' : ''}
              >
                {pdf.originalName} 
                {pdf.processed && <span className="processed-badge">Processed</span>}
                {pdf.pageCount > 0 && <span className="page-count">{pdf.pageCount} pages</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
      
      {selectedPdf && (
        <ChatInterface pdf={selectedPdf} model={selectedModel} />
      )}
    </div>
  );
};

export default RAGInterface;