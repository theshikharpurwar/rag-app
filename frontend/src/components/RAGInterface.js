import React, { useState, useEffect } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadPDFs = async () => {
    setLoading(true);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        setPdfs(response.data || []);
        
        // Clear selected PDF if it's no longer in the list
        if (selectedPdf && !response.data.find(pdf => pdf._id === selectedPdf._id)) {
          setSelectedPdf(null);
        }
      } else {
        setError('Failed to fetch PDFs');
      }
    } catch (err) {
      setError('Error fetching PDFs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPDFs();
  }, []);

  const handleReset = async () => {
    setSelectedPdf(null);
    await loadPDFs();
  };

  const handleUpload = async () => {
    await loadPDFs();
  };

  return (
    <div className="rag-interface">
      <h2>Local RAG System</h2>
      <PDFUploader onUpload={handleUpload} />
      <div className="pdf-list">
        <h3>Uploaded PDFs</h3>
        {loading && <p>Loading PDFs...</p>}
        {error && <p className="error">{error}</p>}
        {!loading && pdfs.length === 0 && <p>No PDFs uploaded yet</p>}
        <ul>
          {pdfs.map(pdf => (
            <li key={pdf._id} onClick={() => setSelectedPdf(pdf)}>
              {pdf.originalName} - {pdf.pageCount} pages
            </li>
          ))}
        </ul>
      </div>
      
      <div className="reset-container">
        <ResetButton onReset={handleReset} />
      </div>
      
      {selectedPdf && (
        <ChatInterface pdf={selectedPdf} />
      )}
      
      <div className="footer">
        <p>Powered by Ollama & Sentence Transformers</p>
      </div>
    </div>
  );
};

export default RAGInterface;