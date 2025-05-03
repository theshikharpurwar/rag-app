// D:\rag-app\frontend\src\components\Layout.js

import React from 'react';
import './Layout.css';

const Layout = ({ children }) => {
  return (
    <div className="layout">
      <header className="header">
        <h1>Ollama RAG Application</h1>
      </header>
      <main className="main-content">
        {children}
      </main>
      <footer className="footer">
        <p>&copy; {new Date().getFullYear()} - Multimodal RAG System</p>
      </footer>
    </div>
  );
};

export default Layout;