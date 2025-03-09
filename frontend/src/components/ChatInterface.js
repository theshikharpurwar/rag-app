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
    // When pdf changes, reset messages
    setMessages([]);
  }, [pdf]);

  useEffect(() => {
    // Scroll to bottom whenever messages change
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsTyping(true);
    
    try {
      const response = await queryRAG(pdf._id, input);
      
      if (response.success) {
        const { answer, sources } = response.data;
        
        const assistantMessage = {
          role: 'assistant',
          content: answer,
          sources: sources,
          timestamp: new Date()
        };
        
        setMessages(prevMessages => [...prevMessages, assistantMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: `Error: ${response.message || 'Failed to get a response'}`,
          timestamp: new Date()
        };
        
        setMessages(prevMessages => [...prevMessages, errorMessage]);
      }
    } catch (error) {
      console.error('Error in handleSubmit:', error);
      
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request',
        timestamp: new Date()
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
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
        {messages.length === 0 && (
          <div className="welcome-message">
            <p>Ask questions about this document</p>
            <p>Try: "What is this document about?" or "Give me a summary"</p>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">
              <p>{msg.content}</p>
              
              {msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <h4>Sources:</h4>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        Page {source.page} - Score: {source.score.toFixed(2)}
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
        <button type="submit" disabled={isTyping}>Send</button>
      </form>
    </div>
  );
};

export default ChatInterface;