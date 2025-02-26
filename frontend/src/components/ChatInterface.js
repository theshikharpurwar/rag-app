import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './ChatInterface.css';

function ChatInterface({ document, llmModel }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    
    const question = input.trim();
    setInput('');
    
    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    
    // Add temporary assistant message
    setMessages(prev => [...prev, { role: 'assistant', content: '...', temporary: true }]);
    
    setLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/api/query', {
        question,
        documentId: document._id
      });
      
      // Replace temporary message with actual response
      setMessages(prev => [
        ...prev.filter(msg => !msg.temporary),
        { role: 'assistant', content: response.data.answer }
      ]);
    } catch (error) {
      // Replace temporary message with error
      setMessages(prev => [
        ...prev.filter(msg => !msg.temporary),
        { 
          role: 'assistant', 
          content: `Error: ${error.response?.data?.error || 'Failed to get response'}`,
          isError: true
        }
      ]);
      console.error('Query error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h4>{document.originalName}</h4>
        <div className="model-info">
          Using: {llmModel.name}
        </div>
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <p>Ask a question about this document</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div 
              key={index} 
              className={`message ${msg.role} ${msg.isError ? 'error' : ''} ${msg.temporary ? 'temporary' : ''}`}
            >
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={!input.trim() || loading}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;