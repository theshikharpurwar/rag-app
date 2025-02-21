import React, { useState } from 'react';
import axios from 'axios';

function RAGInterface() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [pdfId, setPdfId] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
    setError('');
    setPdfId(null);
    setAnswer('');
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a PDF file');
      return;
    }

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/api/upload/pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setMessage(response.data.message);
      setPdfId(response.data.pdfId);
      setError('');
    } catch (error) {
      setError(`Error: ${error.response?.data?.error || error.message}`);
      setMessage('');
      console.error('Upload Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!question || !pdfId) {
      setError('Please upload a PDF and enter a question');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:5000/api/query/question', {
        question,
        pdfId,
      });
      setAnswer(response.data.answer);
    } catch (error) {
      setError(`Error: ${error.response?.data?.error || error.message}`);
      setAnswer('');
      console.error('Query Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Local RAG Application</h1>
      <div>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Processing...' : 'Upload PDF'}
        </button>
        {message && <p style={{ color: 'green' }}>{message}</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </div>
      {pdfId && (
        <div style={{ marginTop: '20px' }}>
          <h2>Ask a Question About {pdfId}</h2>
          <form onSubmit={handleQuestionSubmit}>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question here..."
              rows="4"
              style={{ width: '100%', marginBottom: '10px' }}
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Loading...' : 'Ask'}
            </button>
          </form>
          {answer && <p style={{ marginTop: '10px' }}>Answer: {answer}</p>}
          {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
      )}
    </div>
  );
}

export default RAGInterface;