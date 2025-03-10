// D:\rag-app\frontend\src\components\ChatInterface.js

import React, { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf, model }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const [showCommands, setShowCommands] = useState(false);

  const sampleCommands = [
    { text: "Summarize this document", description: "Get a complete summary" },
    { text: "What are the main topics?", description: "Extract key topics" },
    { text: "Generate questions about this document", description: "Create sample questions" },
  ];

  useEffect(() => {
    setMessages([]);
  }, [pdf]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setShowCommands(false);

    try {
      const response = await queryRAG(pdf._id, input, model);
      
      const assistantMessage = {
        role: 'assistant',
        content: response.answer || "I couldn't generate an answer for that query.",
        timestamp: Date.now(),
        sources: response.sources || []
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error processing query:', error);
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

  const handleCommandClick = (command) => {
    setInput(command);
    setShowCommands(false);
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="chat-interface card">
      <div className="chat-header">
        <div className="chat-title">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" />
          </svg>
          <h3>Chat with {pdf.originalName}</h3>
        </div>
        <div className="chat-actions">
          <button className="icon-button" onClick={clearChat} title="Clear chat">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z" />
            </svg>
          </button>
        </div>
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h4>Ask questions about this document</h4>
            <p>You can ask about specific content, request summaries, or explore topics covered in the document.</p>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                  <path fill="currentColor" d="M12,4A4,4 0 0,1 16,8A4,4 0 0,1 12,12A4,4 0 0,1 8,8A4,4 0 0,1 12,4M12,14C16.42,14 20,15.79 20,18V20H4V18C4,15.79 7.58,14 12,14Z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                  <path fill="currentColor" d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7H14A7,7 0 0,1 21,14H22A1,1 0 0,1 23,15V18A1,1 0 0,1 22,19H21V20A2,2 0 0,1 19,22H5A2,2 0 0,1 3,20V19H2A1,1 0 0,1 1,18V15A1,1 0 0,1 2,14H3A7,7 0 0,1 10,7H11V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2M7.5,13A2.5,2.5 0 0,0 5,15.5A2.5,2.5 0 0,0 7.5,18A2.5,2.5 0 0,0 10,15.5A2.5,2.5 0 0,0 7.5,13M16.5,13A2.5,2.5 0 0,0 14,15.5A2.5,2.5 0 0,0 16.5,18A2.5,2.5 0 0,0 19,15.5A2.5,2.5 0 0,0 16.5,13Z" />
                </svg>
              )}
            </div>
            <div className="message-content">
              <div className="message-text">{msg.content}</div>
              <div className="message-meta">
                <span className="timestamp">{formatTime(msg.timestamp)}</span>
              </div>
              
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <div className="sources-header">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
                      <path fill="currentColor" d="M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z" />
                    </svg>
                    <span>Sources:</span>
                  </div>
                  <ul className="sources-list">
                    {msg.sources.map((source, idx) => (
                      <li key={idx} className="source-item">
                        <span className="source-page">Page {source.page}</span>
                        <span className="source-document">{source.document || pdf.originalName}</span>
                        {source.score !== undefined && (
                          <span className="source-score">
                            <div className="score-bar" style={{ width: `${Math.min(source.score * 100, 100)}%` }}></div>
                          </span>
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
            <div className="message-avatar">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                <path fill="currentColor" d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7H14A7,7 0 0,1 21,14H22A1,1 0 0,1 23,15V18A1,1 0 0,1 22,19H21V20A2,2 0 0,1 19,22H5A2,2 0 0,1 3,20V19H2A1,1 0 0,1 1,18V15A1,1 0 0,1 2,14H3A7,7 0 0,1 10,7H11V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2M7.5,13A2.5,2.5 0 0,0 5,15.5A2.5,2.5 0 0,0 7.5,18A2.5,2.5 0 0,0 10,15.5A2.5,2.5 0 0,0 7.5,13M16.5,13A2.5,2.5 0 0,0 14,15.5A2.5,2.5 0 0,0 16.5,18A2.5,2.5 0 0,0 19,15.5A2.5,2.5 0 0,0 16.5,13Z" />
              </svg>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input" onSubmit={handleSubmit}>
        <div className="input-container">
          <input 
            type="text" 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Ask a question about the document..." 
            onFocus={() => setShowCommands(true)}
            required 
          />
          <button 
            type="button" 
            className="command-button"
            onClick={() => setShowCommands(!showCommands)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M7,10L12,15L17,10H7Z" />
            </svg>
          </button>
        </div>
        
        {showCommands && (
          <div className="commands-dropdown">
            {sampleCommands.map((cmd, index) => (
              <div 
                key={index} 
                className="command-item"
                onClick={() => handleCommandClick(cmd.text)}
              >
                <div className="command-text">{cmd.text}</div>
                <div className="command-description">{cmd.description}</div>
              </div>
            ))}
          </div>
        )}
        
        <button type="submit" className="send-button" disabled={!input.trim() || isTyping}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
            <path fill="currentColor" d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;