// D:\rag-app\frontend\src\App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RAGInterface from './components/RAGInterface';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Local RAG Application</h1>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<RAGInterface />} />
          </Routes>
        </main>
        <footer className="App-footer">
          <p>© 2023 Local RAG Application</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;