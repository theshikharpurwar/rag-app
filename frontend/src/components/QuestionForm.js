import React, { useState } from 'react';
import axios from 'axios';

function QuestionForm({ pdfId }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Log question attempt to terminal via backend
      const logResponse = await axios.post('http://localhost:5000/api/log', {
        action: `Asking question: ${question} for PDF: ${pdfId}`,
      });

      const response = await axios.post('http://localhost:5000/api/query/question', {
        question,
        pdfId,
      });
      setAnswer(response.data.answer || 'No answer available');
    } catch (error) {
      setError(`Error: ${error.response?.data?.error || error.message || 'Query failed'}`);
      setAnswer('');
      console.error('Query Error:', error.response?.data || error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Ask a Question</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Ask'}
        </button>
      </form>
      {answer && <p>Answer: {answer}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default QuestionForm;