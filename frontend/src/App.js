import React, { useState } from 'react';
import UploadPDF from './components/UploadPDF';
import QuestionForm from './components/QuestionForm';
import './App.css';

function App() {
  const [pdfId, setPdfId] = useState(null);

  const handleUploadSuccess = (pdfId) => {
    setPdfId(pdfId);
  };

  return (
    <div className="App">
      <h1>Local RAG Application</h1>
      <UploadPDF onUploadSuccess={handleUploadSuccess} />
      {pdfId && <QuestionForm pdfId={pdfId} />}
    </div>
  );
}

export default App;