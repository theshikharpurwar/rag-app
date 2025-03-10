// D:\rag-app\frontend\src\components\ChatInterface.js

import React, { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf, model }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Clear messages when PDF changes
    setMessages([]);
  }, [pdf]);

  useEffect(() => {
    // Scroll to bottom of messages
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      // Call API with the query
      const response = await queryRAG(pdf._id, input, model);
      
      console.log('Response from API:', response);
      
      // Add assistant message with sources
      const assistantMessage = {
        role: 'assistant',
        content: response.answer || "I couldn't generate an answer for that query.",
        timestamp: Date.now(),
        sources: response.sources || []
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error processing query:', error);
      // Add error message
      const errorMessage = {
        role: 'assistant',
        content: "Sorry, I encountered an error processing your query.",
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>Chat with {pdf.originalName}</h3>
        <div className="header-actions">
          <button className="icon-button" onClick={() => setMessages([])}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
              <span className="timestamp">{formatTime(msg.timestamp)}</span>
              
              {/* Display sources if available */}
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <h4>Sources:</h4>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        Page {source.page} - {source.document || pdf.originalName}
                        {/* Only display score if it exists */}
                        {source.score !== undefined && (
                          <span className="score"> (Relevance: {Number(source.score).toFixed(2)})</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
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
          placeholder="Ask a question..." 
          required 
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatInterface;