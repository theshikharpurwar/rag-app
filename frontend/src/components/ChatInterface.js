// D:\rag-app\frontend\src\components\ChatInterface.js

import React, { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf, model, apiKey }) => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: `Hi! I'm ready to answer questions about "${pdf.originalName}". What would you like to know?`, timestamp: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Reset chat when PDF changes
    setMessages([
      { role: 'assistant', content: `Hi! I'm ready to answer questions about "${pdf.originalName}". What would you like to know?`, timestamp: new Date() }
    ]);
  }, [pdf]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    if (!apiKey) {
      setMessages([
        ...messages,
        { role: 'user', content: input, timestamp: new Date() },
        { role: 'assistant', content: 'Please provide a Mistral API key in the model settings to use this feature.', timestamp: new Date() }
      ]);
      setInput('');
      return;
    }
    
    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsTyping(true);
    
    try {
      const response = await queryRAG(input, apiKey, pdf._id, model);
      
      let assistantMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date()
      };
      
      // Add sources if available
      if (response.sources && response.sources.length > 0) {
        assistantMessage.sources = response.sources;
      }
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error querying RAG:', err);
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: `Error: ${err.message || 'Failed to get response'}`, 
          timestamp: new Date() 
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>Chat with "{pdf.originalName}"</h3>
        <div className="header-actions">
          <button 
            className="icon-button" 
            onClick={() => setMessages([{ role: 'assistant', content: `Hi! I'm ready to answer questions about "${pdf.originalName}". What would you like to know?`, timestamp: new Date() }])}
            title="Clear chat"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2V22M2 12H22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
      
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">
              <p>{msg.content}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <p className="sources-title">Sources:</p>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        {source.file && <span className="source-file">{source.file}</span>}
                        {source.page && <span className="source-page">Page {source.page}</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <span className="timestamp">{formatTime(msg.timestamp)}</span>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="message assistant typing">
            <p>Typing...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input" onSubmit={handleSubmit}>
        <input 
          type="text" 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder={apiKey ? "Ask a question..." : "Enter API key to enable chat"} 
          disabled={!apiKey}
          required 
        />
        <button type="submit" disabled={!apiKey}>Send</button>
      </form>
      
      {!apiKey && (
        <div className="api-key-missing">
          <p>Please enter your Mistral API key in the Model Settings section to enable chat functionality.</p>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;