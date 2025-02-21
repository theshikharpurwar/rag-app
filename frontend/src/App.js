import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [imagePath, setImagePath] = useState('');
  const [response, setResponse] = useState(null);

  const handleQuery = async () => {
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, image: imagePath }),
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Multimodal RAG App</h1>
        <input
          type="text"
          placeholder="Enter your query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <input
          type="text"
          placeholder="Enter image path"
          value={imagePath}
          onChange={(e) => setImagePath(e.target.value)}
        />
        <button onClick={handleQuery}>Submit</button>
      </header>
      {response && <p>{response}</p>}
    </div>
  );
}

export default App;