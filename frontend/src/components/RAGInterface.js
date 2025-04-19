// FILE: frontend/src/components/RAGInterface.js (Full Code with Fixes)

import React, { useState, useEffect, useCallback } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css'; // Ensure this is imported

// Helper logger (optional)
const logger = { info: console.log, warn: console.warn, error: console.error };

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Note: processingPdfIds state was part of an older approach, can be removed if not used.
  // const [processingPdfIds, setProcessingPdfIds] = useState(new Set()); // Leftover state?

  const loadPDFs = useCallback(async (selectFirst = false) => {
    logger.info("Loading PDFs...");
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        const fetchedPdfs = response.pdfs || [];
        setPdfs(fetchedPdfs);
        // *** FIX: Corrected log message typo ***
        logger.info(`Fetched ${fetchedPdfs.length} PDFs.`);

        // Update selection logic
        setSelectedPdf(prev => {
           const currentId = prev?._id;
           const stillExists = currentId ? fetchedPdfs.some(pdf => pdf._id === currentId) : false;
           if (stillExists) {
               const updatedCurrentPdf = fetchedPdfs.find(pdf => pdf._id === currentId);
               return updatedCurrentPdf || null;
           } else if (selectFirst && fetchedPdfs.length > 0) {
               return fetchedPdfs[0];
           } else {
               return null;
           }
        });
      } else {
        const errorMsg = 'Failed to fetch PDFs: ' + (response.message || 'Unknown server error');
        logger.error(errorMsg); setError(errorMsg); setPdfs([]); setSelectedPdf(null);
      }
    } catch (err) {
      const errorMsg = 'Error fetching PDFs: ' + (err.message || 'Network error');
      logger.error(errorMsg, err); setError(errorMsg); setPdfs([]); setSelectedPdf(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadPDFs(true); }, [loadPDFs]);

  const handlePdfUploadSuccess = () => {
    logger.info("Upload successful, reloading PDF list.");
    loadPDFs(); // Reload list to get new PDF with updated status/page count
  };

  const handleResetSuccess = () => {
    logger.info("System reset successful, clearing local state.");
    setPdfs([]);
    setSelectedPdf(null);
  };

  const handleSelectPdf = (pdf) => {
      if (pdf.processed === false) {
           logger.warn(`PDF ${pdf.originalName} is still processing. Selection prevented.`);
           setError(`Document "${pdf.originalName}" is still processing. Please wait.`);
           setTimeout(() => setError(null), 3000);
           return;
      }
      setSelectedPdf(pdf);
      setError(null);
  };

  return (
    <div className="rag-app ghibli-theme"> {/* Ensure theme class from App.css */}
      <header className="app-header">
        <div className="container">
          <h1>Local RAG Application</h1>
        </div>
      </header>

      <main className="container">
        {/* Section for Upload - takes full width */}
        <div className="upload-section">
          <div className="card upload-card">
            <h2>Upload Document</h2>
            <PDFUploader onUpload={handlePdfUploadSuccess} />
          </div>
        </div>

        {/* Section for Reset Button - moved below upload */}
        <div className="reset-section">
          <ResetButton onReset={handleResetSuccess} />
        </div>

        {/* Loading/Error Messages */}
        {loading && !pdfs.length && <div className="status-message loading">Loading documents...</div>}
        {error && <div className="status-message error">{error}</div>}

        {/* Main Content Grid */}
        <div className="content-section">
          {/* Documents List Card */}
          <div className="card documents-card">
            <h2>Uploaded Documents</h2>
            {(pdfs && pdfs.length > 0) ? (
              <div className="documents-list">
                {pdfs.map(pdf => {
                  const isProcessing = pdf.processed === false;
                  const isSelected = selectedPdf && selectedPdf._id === pdf._id;
                  return (
                    <div
                      key={pdf._id}
                      className={`document-item ${isSelected ? 'selected' : ''} ${isProcessing ? 'processing' : ''}`}
                      onClick={() => handleSelectPdf(pdf)}
                      title={isProcessing ? `${pdf.originalName} (Processing...)` : `Select ${pdf.originalName}`}
                      style={{ cursor: isProcessing ? 'not-allowed' : 'pointer', opacity: isProcessing ? 0.6 : 1 }}
                    >
                      <div className="document-icon">
                        {isProcessing ? <SpinnerIcon /> : <PdfIcon /> }
                      </div>
                      <div className="document-info">
                        <div className="document-name">{pdf.originalName}</div>
                        <div className="document-pages">
                          {isProcessing ? 'Processing...' : `${pdf.pageCount ?? 0} pages`}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : !loading && (
              <div className="empty-state"> <PdfIcon /> <p>No documents uploaded yet</p> </div>
            )}
          </div>

          {/* Chat Section */}
          <div className="chat-section">
            {(selectedPdf && selectedPdf.processed !== false) ? (
              <ChatInterface pdf={selectedPdf} />
            ) : (
              <div className="card select-prompt-card">
                <div className="empty-state"> <ChatIcon /> <p>{selectedPdf && selectedPdf.processed === false ? `"${selectedPdf.originalName}" is processing...` : "Select a processed document to start chatting"}</p> </div>
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

// Constants
const LLM_MODEL_NAME = 'tinyllama';
const EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2';

// Simple SVG Icon Components (Defined only ONCE)
const SpinnerIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M12,4a8,8,0,0,1,7.89,6.7A1.53,1.53,0,0,0,21.38,12h0a1.5,1.5,0,0,0,1.48-1.75,11,11,0,0,0-21.72,0A1.5,1.5,0,0,0,2.62,12h0a1.53,1.53,0,0,0,1.49-1.3A8,8,0,0,1,12,4Z"><animateTransform attributeName="transform" type="rotate" dur="0.75s" values="0 12 12;360 12 12" repeatCount="indefinite"/></path></svg>;
const PdfIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" /></svg>;
const ChatIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48"><path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" /></svg>;

// *** REMOVED DUPLICATE SVG DEFINITIONS FROM HERE ***

export default RAGInterface;