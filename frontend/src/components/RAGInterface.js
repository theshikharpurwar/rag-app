// FILE: frontend/src/components/RAGInterface.js (Full Code)

import React, { useState, useEffect, useCallback } from 'react';
import { fetchPDFs } from '../api'; // Assuming api.js handles calls
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

// Helper logger
const logger = { info: console.log, warn: console.warn, error: console.error };

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(true); // Start loading on initial mount
  const [error, setError] = useState(null);
  // *** ADDED STATE to track processing PDFs ***
  const [processingPdfIds, setProcessingPdfIds] = useState(new Set());

  // Function to load PDFs, memoized
  const loadPDFs = useCallback(async (selectFirst = false) => {
    logger.info("Loading PDFs...");
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        const fetchedPdfs = response.pdfs || [];
        setPdfs(fetchedPdfs);
        logger.info(`Workspaceed ${fetchedPdfs.length} PDFs.`);

        // Update selection logic
        setSelectedPdf(prev => {
           const currentId = prev?._id;
           const stillExists = currentId ? fetchedPdfs.some(pdf => pdf._id === currentId) : false;

           if (stillExists) {
               // If current selection still exists, find its updated data (e.g., page count, processed status)
               const updatedCurrentPdf = fetchedPdfs.find(pdf => pdf._id === currentId);
               return updatedCurrentPdf ? updatedCurrentPdf : null; // Return updated data or null if somehow missing
           } else if (selectFirst && fetchedPdfs.length > 0) {
               // If told to select first (e.g., after initial load or reset)
               return fetchedPdfs[0];
           } else {
               // Otherwise, keep null or previous valid selection if it wasn't found
               return null;
           }
        });

      } else {
        const errorMsg = 'Failed to fetch PDFs: ' + (response.message || 'Unknown server error');
        logger.error(errorMsg);
        setError(errorMsg);
        setPdfs([]);
        setSelectedPdf(null);
      }
    } catch (err) {
      const errorMsg = 'Error fetching PDFs: ' + (err.message || 'Network error or server down');
      logger.error(errorMsg, err);
      setError(errorMsg);
      setPdfs([]);
      setSelectedPdf(null);
    } finally {
      setLoading(false);
    }
  }, []); // useCallback with empty dependency array

  // Initial load
  useEffect(() => {
    loadPDFs(true); // Select first PDF on initial load
  }, [loadPDFs]);


  const handlePdfUploadSuccess = () => {
    logger.info("Upload successful, reloading PDF list.");
    // No need to manage processingPdfIds here anymore, as the upload API waits.
    // Simply reload the list to get the latest data including the new PDF.
    loadPDFs();
  };

  const handleResetSuccess = () => {
    logger.info("System reset successful, clearing local state.");
    setPdfs([]);
    setSelectedPdf(null);
    // Optionally reload PDFs after reset if needed, or leave list empty
    // loadPDFs(true);
  };

  const handleSelectPdf = (pdf) => {
      // *** PREVENT SELECTION if PDF is not processed ***
      if (pdf.processed === false) {
           logger.warn(`PDF ${pdf.originalName} is still processing. Selection prevented.`);
           // Optionally show a temporary message to the user
           setError(`Document "${pdf.originalName}" is still processing. Please wait.`);
           setTimeout(() => setError(null), 3000); // Clear message after 3s
           return; // Do not select
      }
      setSelectedPdf(pdf);
      setError(null); // Clear any previous errors on successful selection
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
            {/* Pass the success handler */}
            <PDFUploader onUpload={handlePdfUploadSuccess} />
          </div>
          <div className="reset-section">
            {/* Pass the success handler */}
            <ResetButton onReset={handleResetSuccess} />
          </div>
        </div>

        {/* Display general loading/error states */}
        {loading && !pdfs.length && <div className="status-message loading">Loading documents...</div>}
        {error && <div className="status-message error">{error}</div>}

        <div className="content-section">
          <div className="card documents-card">
            <h2>Uploaded Documents</h2>
            {/* PDF List Logic */}
            {(pdfs && pdfs.length > 0) ? (
              <div className="documents-list">
                {pdfs.map(pdf => {
                  // Check if this specific PDF is processing (using the 'processed' flag from backend)
                  const isProcessing = pdf.processed === false; // Check the flag from the fetched data
                  const isSelected = selectedPdf && selectedPdf._id === pdf._id;

                  return (
                    <div
                      key={pdf._id}
                      // Add 'processing' class if needed for styling
                      className={`document-item ${isSelected ? 'selected' : ''} ${isProcessing ? 'processing' : ''}`}
                      onClick={() => handleSelectPdf(pdf)} // Use handler to prevent selection if processing
                      title={isProcessing ? `${pdf.originalName} (Processing...)` : `Select ${pdf.originalName}`}
                      style={{ cursor: isProcessing ? 'not-allowed' : 'pointer', opacity: isProcessing ? 0.6 : 1 }} // Visual cue
                    >
                      <div className="document-icon">
                        {/* Use different icon or add spinner if processing */}
                        {isProcessing ? (
                           <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M12,4a8,8,0,0,1,7.89,6.7A1.53,1.53,0,0,0,21.38,12h0a1.5,1.5,0,0,0,1.48-1.75,11,11,0,0,0-21.72,0A1.5,1.5,0,0,0,2.62,12h0a1.53,1.53,0,0,0,1.49-1.3A8,8,0,0,1,12,4Z"><animateTransform attributeName="transform" type="rotate" dur="0.75s" values="0 12 12;360 12 12" repeatCount="indefinite"/></path></svg>
                        ) : (
                           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" /></svg>
                        )}
                      </div>
                      <div className="document-info">
                        <div className="document-name">{pdf.originalName}</div>
                        <div className="document-pages">
                          {/* Show actual page count or processing status */}
                          {isProcessing ? 'Processing...' : `${pdf.pageCount ?? 0} pages`}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : !loading && ( // Only show empty state if not loading and no PDFs
              <div className="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48"><path fill="currentColor" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" /></svg>
                <p>No documents uploaded yet</p>
              </div>
            )}
          </div>

          <div className="chat-section">
            {/* Render chat only if a PDF is selected AND processed */}
            {(selectedPdf && selectedPdf.processed !== false) ? (
              <ChatInterface pdf={selectedPdf} />
            ) : (
              <div className="card select-prompt-card">
                <div className="empty-state">
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48"><path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" /></svg>
                  {/* Show appropriate message */}
                  <p>{selectedPdf && selectedPdf.processed === false ? `"${selectedPdf.originalName}" is processing...` : "Select a processed document to start chatting"}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

       <footer className="app-footer">
         <div className="container">
           {/* Footer remains the same */}
           <p>Powered by Ollama ({LLM_MODEL_NAME}) & Sentence Transformers ({EMBEDDING_MODEL_NAME})</p>
         </div>
       </footer>
    </div>
  );
};

// Constants used in footer
const LLM_MODEL_NAME = 'tinyllama';
const EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2';

export default RAGInterface;