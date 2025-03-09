// D:\rag-app\frontend\src\components\ChatInterface.js

import React, { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Add welcome message when PDF is selected
    if (pdf) {
      setMessages([
        {
          role: 'assistant',
          content: `I'm ready to answer questions about "${pdf.originalName}". What would you like to know?`,
          timestamp: new Date()
        }
      ]);
    }
  }, [pdf]);

  useEffect(() => {
    // Scroll to bottom whenever messages change
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    
    try {
      // Send query to backend
      const response = await queryRAG(input);
      
      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request.',
        timestamp: new Date()
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
                        {source.source} (Page {source.page})
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
          placeholder="Ask a question..." 
          required 
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatInterface;