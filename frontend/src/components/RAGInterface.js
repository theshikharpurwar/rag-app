import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ModelSelector from './ModelSelector';
import './RAGInterface.css';

function RAGInterface() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [models, setModels] = useState({
    embedding: { name: 'colpali', path: 'vidore/colpali-v1.2' },
    llm: { name: 'qwen', path: 'Qwen/Qwen2.5-VL-3B-Instruct' }
  });

  // Fetch documents on component mount
  useEffect(() => {
    fetchDocuments();
    fetchModels();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/api/documents');
      setDocuments(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch documents');
      console.error('Error fetching documents:', err);
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/model-config');
      
      // Process model configurations
      const embeddingModels = response.data.filter(m => m.type === 'embedding');
      const llmModels = response.data.filter(m => m.type === 'llm');
      
      const activeEmbedding = embeddingModels.find(m => m.isActive) || models.embedding;
      const activeLLM = llmModels.find(m => m.isActive) || models.llm;
      
      setModels({
        embedding: activeEmbedding,
        llm: activeLLM
      });
    } catch (err) {
      console.error('Error fetching model configurations:', err);
    }
  };

  const handleDocumentUpload = async (file) => {
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('pdf', file);
      
      const response = await axios.post('http://localhost:5000/api/upload/pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Add the newly uploaded document to the list
      setDocuments(prevDocs => [
        {
          _id: response.data.documentId,
          originalName: file.name,
          storedName: response.data.storedName,
          status: 'processing',
          uploadedAt: new Date().toISOString()
        },
        ...prevDocs
      ]);
      
      setLoading(false);
      
      // Poll for document processing status
      const pollInterval = setInterval(async () => {
        try {
          const updatedDocs = await axios.get('http://localhost:5000/api/documents');
          setDocuments(updatedDocs.data);
          
          // Find the uploaded document and check its status
          const uploadedDoc = updatedDocs.data.find(doc => doc.storedName === response.data.storedName);
          
          if (uploadedDoc && (uploadedDoc.status === 'indexed' || uploadedDoc.status === 'failed')) {
            clearInterval(pollInterval);
          }
        } catch (err) {
          console.error('Error polling document status:', err);
          clearInterval(pollInterval);
        }
      }, 5000); // Poll every 5 seconds
      
    } catch (err) {
      setError('Failed to upload document');
      console.error('Error uploading document:', err);
      setLoading(false);
    }
  };

  const handleModelChange = async (type, modelId) => {
    try {
      await axios.put(`http://localhost:5000/api/model-config/${modelId}`, {
        isActive: true
      });
      
      // Refresh models
      fetchModels();
    } catch (err) {
      setError(`Failed to change ${type} model`);
      console.error('Error changing model:', err);
    }
  };

  const handleDocumentSelect = (docId) => {
    const doc = documents.find(d => d._id === docId);
    setSelectedDoc(doc);
  };

  return (
    <div className="rag-container">
      <header className="rag-header">
        <h3>MultiModal RAG Application</h3>
        <p>Upload PDFs and ask questions about their content</p>
      </header>
      
      <div className="rag-content">
        <div className="rag-sidebar">
          <ModelSelector
            embeddingModel={models.embedding}
            llmModel={models.llm}
            onModelChange={handleModelChange}
          />
          
          <div className="doc-selector">
            <h4>Documents</h4>
            {loading && <p>Loading...</p>}
            
            <ul className="doc-list">
              {documents.map(doc => (
                <li
                  key={doc._id}
                  className={`doc-item ${selectedDoc && selectedDoc._id === doc._id ? 'selected' : ''} ${doc.status}`}
                  onClick={() => doc.status === 'indexed' && handleDocumentSelect(doc._id)}
                >
                  <span className="doc-name">{doc.originalName}</span>
                  <span className="doc-status">{doc.status}</span>
                </li>
              ))}
            </ul>
            
            <PDFUploader onUpload={handleDocumentUpload} disabled={loading} />
          </div>
        </div>
        
        <div className="rag-main">
          {selectedDoc ? (
            <ChatInterface
              document={selectedDoc}
              llmModel={models.llm}
            />
          ) : (
            <div className="placeholder">
              <h4>Select a document or upload a new one</h4>
              <p>Once a document is indexed, you can ask questions about it</p>
            </div>
          )}
        </div>
      </div>
      
      {error && (
        <div className="error-notification">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
    </div>
  );
}

export default RAGInterface;