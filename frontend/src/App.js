// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // State for the selected file to upload.
  const [file, setFile] = useState(null);
  // State for messages related to the upload process.
  const [uploadMessage, setUploadMessage] = useState('');
  // State for storing the user's query.
  const [query, setQuery] = useState('');
  // State for storing the answer received from the backend.
  const [answer, setAnswer] = useState('');
  // State for optionally viewing the prompt sent to the LLM (for debugging).
  const [prompt, setPrompt] = useState('');

  // Handler for file selection.
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Handler for uploading the selected file.
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }

    // Create a FormData object to send the file.
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send a POST request to the /api/upload endpoint.
      const res = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setUploadMessage(res.data.message);
    } catch (error) {
      console.error("Upload Error:", error);
      setUploadMessage("Upload failed.");
    }
  };

  // Handler for sending a query.
  const handleQuery = async () => {
    if (!query) {
      alert("Please enter a query.");
      return;
    }

    try {
      // Send a POST request to the /api/query endpoint with the query.
      const res = await axios.post('http://localhost:5000/api/query', { query });
      setAnswer(res.data.answer);
      setPrompt(res.data.prompt);
    } catch (error) {
      console.error("Query Error:", error);
      setAnswer("Error processing query.");
    }
  };

  return (
    <div className="App" style={{ padding: '2rem' }}>
      <h1>Retrieval-Augmented Generator (RAG) App</h1>

      {/* Section for document upload */}
      <section style={{ marginBottom: '2rem' }}>
        <h2>Upload Document</h2>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload} style={{ marginLeft: '1rem' }}>Upload</button>
        <p>{uploadMessage}</p>
      </section>

      {/* Section for query submission */}
      <section style={{ marginBottom: '2rem' }}>
        <h2>Ask a Question</h2>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query here..."
          rows="4"
          cols="50"
        />
        <br />
        <button onClick={handleQuery} style={{ marginTop: '1rem' }}>Submit Query</button>
      </section>

      {/* Section to display the answer */}
      <section style={{ marginBottom: '2rem' }}>
        <h2>Answer</h2>
        <p style={{ whiteSpace: 'pre-wrap', background: '#f4f4f4', padding: '1rem' }}>{answer}</p>
      </section>

      {/* Optional section to display the prompt sent to the LLM */}
      <section>
        <h2>Prompt (for debugging)</h2>
        <pre style={{ background: '#eef', padding: '1rem' }}>{prompt}</pre>
      </section>
    </div>
  );
}

export default App;
