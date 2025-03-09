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
      setError('Error fetching PDFs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllPDFs();
  }, []);

  return (
    <div className="rag-interface">
      <h2>Local RAG System</h2>
      
      <div className="rag-container">
        <div className="left-panel">
          <h3>Upload Document</h3>
          <PDFUploader onUpload={fetchAllPDFs} />
          
          <h3>Uploaded PDFs</h3>
          {loading && <p>Loading PDFs...</p>}
          {error && <p className="error">{error}</p>}
          {pdfs.length === 0 && !loading ? (
            <p>No PDFs uploaded yet</p>
          ) : (
            <ul className="pdf-list">
              {pdfs.map(pdf => (
                <li 
                  key={pdf._id} 
                  onClick={() => setSelectedPdf(pdf)}
                  className={selectedPdf && pdf._id === selectedPdf._id ? 'selected' : ''}
                >
                  {pdf.originalName} ({pdf.pageCount} pages)
                </li>
              ))}
            </ul>
          )}
          
          <div className="reset-container">
            <ResetButton onReset={fetchAllPDFs} />
          </div>
        </div>
        
        <div className="right-panel">
          {selectedPdf ? (
            <ChatInterface pdf={selectedPdf} />
          ) : (
            <div className="no-pdf-selected">
              <p>Select a PDF or upload a new one to start chatting</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;