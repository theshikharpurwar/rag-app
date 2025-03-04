// D:\rag-app\frontend\src\components\RAGInterface.js

import React, { useState, useEffect } from 'react';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ModelSelector from './ModelSelector';
import { getPDFs } from '../api';
import './RAGInterface.css';

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [selectedModel, setSelectedModel] = useState({
    name: 'clip',
    path: 'openai/clip-vit-base-patch32'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch PDFs on component mount
  useEffect(() => {
    fetchPDFs();
  }, []);

  // Fetch PDFs from API
  const fetchPDFs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await getPDFs();
      
      if (response.success) {
        setPdfs(response.pdfs);
      } else {
        setError('Failed to fetch PDFs');
      }
    } catch (err) {
      setError('Error fetching PDFs');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rag-interface">
      <h2>Upload and Query PDFs</h2>
      <PDFUploader onUpload={fetchPDFs} />
      <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      {loading && <p>Loading PDFs...</p>}
      {error && <p className="error">{error}</p>}
      <div className="pdf-list">
        <h3>Uploaded PDFs</h3>
        <ul>
          {pdfs.map(pdf => (
            <li key={pdf._id} onClick={() => setSelectedPdf(pdf)}>
              {pdf.originalName} - {pdf.processingStatus}
              {pdf.pageCount && ` - ${pdf.pageCount} pages`}
            </li>
          ))}
        </ul>
      </div>
      {selectedPdf && (
        <ChatInterface pdf={selectedPdf} model={selectedModel} />
      )}
    </div>
  );
};

export default RAGInterface;