import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSearch = async () => {
    const formData = new FormData();
    formData.append('file', file);
    await axios.post('http://localhost:5000/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    const response = await axios.get(`http://localhost:5000/search?query=${query}`);
    setResults(response.data);
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSearch}>Search</button>
      <ul>
        {results.map((result, index) => (
          <li key={index}>{result.metadata}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;