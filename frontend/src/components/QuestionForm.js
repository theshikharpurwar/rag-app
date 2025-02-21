import React, { useState } from 'react';
import axios from 'axios';

function QuestionForm({ pdfId }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/api/ask-question', {
        question,
        pdfId,
      });
      setAnswer(response.data.answer);
    } catch (error) {
      setAnswer(`Error: ${error.response.data.error}`);
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
    </div>
  );
}

export default QuestionForm;