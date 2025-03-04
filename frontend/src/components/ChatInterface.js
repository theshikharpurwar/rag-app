// D:\rag-app\frontend\src\components\ChatInterface.js

import React, { useState } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf, model }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);
    
    try {
      // Only pass the model name, not the path
      const response = await queryRAG(pdf._id, input, model.name);
      
      if (response.success) {
        const botMessage = {
          text: response.answer,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString(),
          sources: response.sources
        };
        
        setMessages(prev => [...prev, botMessage]);
      } else {
        setError(`Error: ${response.message || 'Failed to get response'}`);
      }
    } catch (err) {
      setError(`Error: ${err.message || 'Something went wrong'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>Chat with {pdf.originalName}</h3>
      </div>
      
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="message-content">
              <p>{msg.text}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <p>Sources:</p>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        {source.pdf_name} (Page {source.page_num})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="timestamp">{msg.timestamp}</div>
          </div>
        ))}
        
        {loading && (
          <div className="message bot">
            <div className="message-content">
              <p>Thinking...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}
      </div>
      
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;