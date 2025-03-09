import React, { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatInterface.css';

const ChatInterface = ({ pdf }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Add welcome message on component mount
    setMessages([
      {
        role: 'assistant',
        content: `Ask me anything about "${pdf.originalName}"`,
        timestamp: new Date()
      }
    ]);
  }, [pdf]);

  useEffect(() => {
    // Scroll to bottom of messages when messages change
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(messages => [...messages, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await queryRAG(pdf._id, input);
      
      if (response.success) {
        const botMessage = {
          role: 'assistant',
          content: response.data.answer,
          timestamp: new Date(),
          sources: response.data.sources || []
        };
        setMessages(messages => [...messages, botMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: `Error: ${response.message || 'Failed to get a response'}`,
          timestamp: new Date()
        };
        setMessages(messages => [...messages, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message || 'Something went wrong'}`,
        timestamp: new Date()
      };
      setMessages(messages => [...messages, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>Chat with {pdf.originalName}</h3>
        <div className="header-actions">
          <button 
            className="icon-button"
            onClick={() => {
              setMessages([
                {
                  role: 'assistant',
                  content: `Ask me anything about "${pdf.originalName}"`,
                  timestamp: new Date()
                }
              ]);
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
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
            </div>
            {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
              <div className="message-sources">
                <p className="sources-heading">Sources:</p>
                <ul>
                  {msg.sources.map((source, idx) => (
                    <li key={idx}>
                      Page {source.page} from {source.document} (Relevance: {source.score})
                    </li>
                  ))}
                </ul>
              </div>
            )}
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