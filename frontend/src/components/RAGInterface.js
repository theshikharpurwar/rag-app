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
  const [selectedModel, setSelectedModel] = useState('mistral');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAllPDFs();
  }, []);

  const fetchAllPDFs = async () => {
    setLoading(true);
    setError(null);
    try {
      const fetchedPdfs = await fetchPDFs();
      setPdfs(fetchedPdfs);
      if (fetchedPdfs.length > 0 && !selectedPdf) {
        setSelectedPdf(fetchedPdfs[0]);
      }
    } catch (err) {
      console.error('Error fetching PDFs:', err);
      setError('Failed to fetch PDFs');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPdfs([]);
    setSelectedPdf(null);
    fetchAllPDFs();
  };

  return (
    <div className="rag-interface">
      <h2>Mistral AI RAG System</h2>
      
      <div className="controls-container">
        <ModelSelector 
          selectedModel={selectedModel} 
          setSelectedModel={setSelectedModel} 
          apiKey={apiKey}
          setApiKey={setApiKey}
        />
        
        <PDFUploader 
          onUpload={fetchAllPDFs} 
          apiKey={apiKey}
          selectedModel={selectedModel}
        />
        
        <ResetButton onReset={handleReset} />
      </div>
      
      {loading && <p className="loading-indicator">Loading PDFs...</p>}
      {error && <p className="error-message">{error}</p>}
      
      <div className="content-container">
        <div className="pdf-list">
          <h3>Uploaded Documents</h3>
          {pdfs.length === 0 ? (
            <p>No documents uploaded yet</p>
          ) : (
            <ul>
              {pdfs.map(pdf => (
                <li 
                  key={pdf._id} 
                  onClick={() => setSelectedPdf(pdf)}
                  className={selectedPdf && selectedPdf._id === pdf._id ? 'selected' : ''}
                >
                  {pdf.originalName} 
                  <span className="page-count">({pdf.pageCount || '?'} pages)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        
        <div className="chat-container">
          {selectedPdf ? (
            <ChatInterface 
              pdf={selectedPdf} 
              model={selectedModel} 
              apiKey={apiKey}
            />
          ) : (
            <div className="no-pdf-selected">
              <p>Upload or select a document to start chatting</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;