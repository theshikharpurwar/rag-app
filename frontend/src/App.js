// D:\rag-app\frontend\src\App.js

import React from 'react';
import RAGInterface from './components/RAGInterface';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Local RAG Application</h1>
      </header>
      <main>
        <RAGInterface />
      </main>
      <footer>
        <p>Powered by Ollama & Sentence Transformers</p>
      </footer>
    </div>
  );
}

export default App;