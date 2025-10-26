import React, { useState, useEffect, useCallback } from 'react';
import { fetchPDFs } from '../api';
import PDFUploader from './PDFUploader';
import ChatInterface from './ChatInterface';
import ResetButton from './ResetButton';
import './RAGInterface.css';

const logger = { info: console.log, warn: console.warn, error: console.error };

const RAGInterface = () => {
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Unused state removed: const [processingPdfIds, setProcessingPdfIds] = useState(new Set());

  const loadPDFs = useCallback(async (selectFirst = false) => {
    logger.info("Loading PDFs...");
    setLoading(true);
    setError(null);
    try {
      const response = await fetchPDFs();
      if (response.success) {
        const fetchedPdfs = response.pdfs || [];
        setPdfs(fetchedPdfs);
        logger.info(`Fetched ${fetchedPdfs.length} PDFs.`); // Corrected typo
        setSelectedPdf(prev => {
           const currentId = prev?._id;
           const stillExists = currentId ? fetchedPdfs.some(pdf => pdf._id === currentId) : false;
           if (stillExists) {
               const updatedCurrentPdf = fetchedPdfs.find(pdf => pdf._id === currentId);
               return updatedCurrentPdf || null;
           } else if (selectFirst && fetchedPdfs.length > 0) {
               return fetchedPdfs[0];
           } else { return null; }
        });
      } else {
        const errorMsg = 'Failed to fetch PDFs: ' + (response.message || 'Server error');
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

  const handlePdfUploadSuccess = () => { logger.info("Upload complete, reloading list."); loadPDFs(); };
  const handleResetSuccess = () => { logger.info("System reset complete."); setPdfs([]); setSelectedPdf(null); };

  const handleSelectPdf = (pdf) => {
      if (pdf.processed === false) {
           logger.warn(`PDF ${pdf.originalName} processing.`); setError(`"${pdf.originalName}" processing...`);
           setTimeout(() => setError(null), 3000); return;
      }
      setSelectedPdf(pdf); setError(null);
  };

  return (
    <div className="docuverse-app">
      {/* Header with Brand */}
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="brand">
              <div className="brand-icon">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 2L5 6V20C5 20.5304 5.21071 21.0391 5.58579 21.4142C5.96086 21.7893 6.46957 22 7 22H17C17.5304 22 18.0391 21.7893 18.4142 21.4142C18.7893 21.0391 19 20.5304 19 20V6L15 2H9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M9 2V6H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 11V17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M9 14L12 11L15 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="brand-text">
                <h1>DocuVerse</h1>
                <span className="brand-tagline">Ask anything, get answers</span>
              </div>
            </div>
            <ResetButton onReset={handleResetSuccess} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          {loading && !pdfs.length && (
            <div className="status-message loading">
              <LoadingSpinner />
              <span>Loading your documents...</span>
            </div>
          )}
          
          {error && (
            <div className="status-message error">
              <ErrorIcon />
              <span>{error}</span>
            </div>
          )}

          <div className="content-grid">
            {/* Left Sidebar - Upload & Documents */}
            <aside className="sidebar">
              {/* Upload Section */}
              <div className="card upload-card">
                <div className="card-header">
                  <UploadIcon />
                  <h3>Upload Document</h3>
                </div>
                <PDFUploader onUpload={handlePdfUploadSuccess} />
              </div>

              {/* Documents List */}
              <div className="card documents-card">
                <div className="card-header">
                  <DocumentsIcon />
                  <h3>Your Library</h3>
                  {pdfs.length > 0 && <span className="badge">{pdfs.length}</span>}
                </div>
                
                <div className="documents-list">
                  {pdfs && pdfs.length > 0 ? (
                    pdfs.map(pdf => {
                      const isProcessing = pdf.processed === false;
                      const isSelected = selectedPdf?._id === pdf._id;
                      
                      return (
                        <div
                          key={pdf._id}
                          className={`document-item ${isSelected ? 'selected' : ''} ${isProcessing ? 'processing' : ''}`}
                          onClick={() => handleSelectPdf(pdf)}
                          title={isProcessing ? `${pdf.originalName} (Processing...)` : `Select ${pdf.originalName}`}
                        >
                          <div className="document-icon">
                            {isProcessing ? <LoadingSpinner /> : <PdfIcon />}
                          </div>
                          <div className="document-info">
                            <div className="document-name">{pdf.originalName}</div>
                            <div className="document-meta">
                              {isProcessing ? (
                                <span className="processing-label">Processing...</span>
                              ) : (
                                <span className="pages-count">{pdf.pageCount ?? 0} pages</span>
                              )}
                            </div>
                          </div>
                          {isSelected && !isProcessing && (
                            <div className="selected-indicator">
                              <CheckIcon />
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : !loading ? (
                    <div className="empty-state">
                      <EmptyDocumentsIcon />
                      <p>No documents yet</p>
                      <span>Upload a PDF to get started</span>
                    </div>
                  ) : null}
                </div>
              </div>
            </aside>

            {/* Right Side - Chat Interface */}
            <div className="chat-container">
              {selectedPdf && selectedPdf.processed !== false ? (
                <ChatInterface pdf={selectedPdf} />
              ) : (
                <div className="card empty-chat-card">
                  <div className="empty-chat-state">
                    <ChatEmptyIcon />
                    <h3>
                      {selectedPdf?.processed === false
                        ? `Processing "${selectedPdf.originalName}"...`
                        : 'Select a document to start chatting'}
                    </h3>
                    <p>
                      {selectedPdf?.processed === false
                        ? 'Your document is being analyzed. This usually takes a few moments.'
                        : 'Choose any document from your library to ask questions and get instant answers.'}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};


// ============================================
// ICON COMPONENTS
// ============================================

const LoadingSpinner = () => (
  <svg className="spinner-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
    <circle className="spinner-circle" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
  </svg>
);

const PdfIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
  </svg>
);

const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const DocumentsIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round"/>
    <polyline points="14 2 14 8 20 8" strokeLinecap="round" strokeLinejoin="round"/>
    <line x1="16" y1="13" x2="8" y2="13" strokeLinecap="round" strokeLinejoin="round"/>
    <line x1="16" y1="17" x2="8" y2="17" strokeLinecap="round" strokeLinejoin="round"/>
    <polyline points="10 9 9 9 8 9" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ChatEmptyIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const EmptyDocumentsIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" strokeLinecap="round" strokeLinejoin="round"/>
    <polyline points="13 2 13 9 20 9" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const CheckIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="3">
    <polyline points="20 6 9 17 4 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ErrorIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" strokeLinecap="round" strokeLinejoin="round"/>
    <line x1="12" y1="8" x2="12" y2="12" strokeLinecap="round" strokeLinejoin="round"/>
    <line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

export default RAGInterface;