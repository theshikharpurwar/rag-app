// FILE: frontend/src/components/ChatInterface.js

import React, { useState, useRef, useEffect } from 'react';
import { queryRAGStream } from '../api';
import './ChatInterface.css';

const PHASE_LABELS = {
  routing: 'Routing…',
  retrieving: 'Retrieving…',
  generating: 'Generating…',
  grading: 'Checking answer…',
  refining: 'Refining answer…',
};

// *** REMOVED model prop ***
const ChatInterface = ({ pdf, onPipelineEvent }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamPhase, setStreamPhase] = useState(null);
  const messagesEndRef = useRef(null);

  const [showCommands, setShowCommands] = useState(false);
  const sampleCommands = [
    { text: "Summarize this document", description: "Get a complete summary" },
    { text: "What are the main topics?", description: "Extract key topics" },
    { text: "Generate questions about this document", description: "Create sample questions" },
  ];

  // Clear messages when PDF changes
  useEffect(() => {
    setMessages([]);
    setInput(''); // Also clear input when PDF changes
  }, [pdf]);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !pdf || isStreaming) return;

    const userMessage = { role: 'user', content: input, timestamp: Date.now() };
    const currentInput = input;
    setInput('');
    setIsStreaming(true);
    setStreamPhase(null);
    setShowCommands(false);

    const history = [];
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role !== 'user') continue;
      const asst = messages[i + 1];
      if (!asst || asst.role !== 'assistant') continue;
      if (asst.streaming) continue;
      history.push({
        user: messages[i].content,
        assistant: asst.content || '',
      });
      i++;
    }

    setMessages((prev) => [
      ...prev,
      userMessage,
      {
        role: 'assistant',
        content: '',
        streaming: true,
        timestamp: Date.now(),
        sources: [],
      },
    ]);

    if (onPipelineEvent) {
      onPipelineEvent({ type: 'query', phase: 'question', query: currentInput });
    }

    await queryRAGStream(pdf._id, currentInput, history, {
      onPhase: (phase) => {
        if (onPipelineEvent) {
          onPipelineEvent({ type: 'query', phase, query: currentInput });
        }
        setStreamPhase(PHASE_LABELS[phase] || phase);
      },
      onStatus: (msg) => {
        if (onPipelineEvent) {
          onPipelineEvent({ type: 'query', phase: 'refining', detail: msg, query: currentInput });
        }
        setStreamPhase(msg || PHASE_LABELS.refining);
      },
      onToken: (token) => {
        if (onPipelineEvent) {
          onPipelineEvent({ type: 'query', phase: 'generating', token, query: currentInput });
        }
        setMessages((prev) => {
          const next = [...prev];
          const last = next.length - 1;
          if (last < 0 || next[last].role !== 'assistant') return next;
          next[last] = {
            ...next[last],
            content: (next[last].content || '') + token,
          };
          return next;
        });
      },
      onDone: ({ sources, answer }) => {
        if (onPipelineEvent) {
          onPipelineEvent({ type: 'query', phase: 'answer', done: true, answer, sources, query: currentInput });
        }
        setIsStreaming(false);
        setStreamPhase(null);
        setMessages((prev) => {
          const next = [...prev];
          const last = next.length - 1;
          if (last < 0 || next[last].role !== 'assistant') return next;
          const text =
            answer != null && answer !== ''
              ? answer
              : next[last].content || '';
          next[last] = {
            ...next[last],
            content: text,
            streaming: false,
            sources: sources || [],
          };
          return next;
        });
      },
      onError: (msg) => {
        if (onPipelineEvent) {
          onPipelineEvent({ type: 'query', phase: 'error', detail: msg, query: currentInput });
        }
        setIsStreaming(false);
        setStreamPhase(null);
        setMessages((prev) => {
          const next = [...prev];
          const last = next.length - 1;
          if (last >= 0 && next[last].role === 'assistant') {
            next[last] = {
              ...next[last],
              content:
                next[last].content ||
                `Error: ${msg || 'Something went wrong'}`,
              streaming: false,
              sources: [],
            };
            return next;
          }
          return [
            ...next,
            {
              role: 'assistant',
              content: `Error: ${msg || 'Something went wrong'}`,
              timestamp: Date.now(),
              sources: [],
            },
          ];
        });
      },
    });
  };

  const handleCommandClick = (command) => {
    if (!pdf) return;
    setInput(command);
    setShowCommands(false);
    // Consider focusing the input after selecting a command
    // document.getElementById('chat-input-field')?.focus();
  };

  const clearChat = () => {
    setMessages([]);
  };

  // Should be handled by parent, but good fallback
  if (!pdf) return null;

  const pdfName = pdf ? pdf.originalName : 'Document';

  return (
    <div className="chat-interface card">
      <div className="chat-header">
        <div className="chat-title">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20" /></svg>
          <h3>Chat with {pdfName}</h3>
        </div>
        <div className="chat-actions">
          <button className="icon-button" onClick={clearChat} title="Clear chat">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z" /></svg>
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h4>Ask questions about {pdfName}</h4>
            <p>You can ask about specific content, request summaries, or explore topics covered in the document.</p>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ?
                (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M12,4A4,4 0 0,1 16,8A4,4 0 0,1 12,12A4,4 0 0,1 8,8A4,4 0 0,1 12,4M12,14C16.42,14 20,15.79 20,18V20H4V18C4,15.79 7.58,14 12,14Z" /></svg>)
                : (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7H14A7,7 0 0,1 21,14H22A1,1 0 0,1 23,15V18A1,1 0 0,1 22,19H21V20A2,2 0 0,1 19,22H5A2,2 0 0,1 3,20V19H2A1,1 0 0,1 1,18V15A1,1 0 0,1 2,14H3A7,7 0 0,1 10,7H11V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2M7.5,13A2.5,2.5 0 0,0 5,15.5A2.5,2.5 0 0,0 7.5,18A2.5,2.5 0 0,0 10,15.5A2.5,2.5 0 0,0 7.5,13M16.5,13A2.5,2.5 0 0,0 14,15.5A2.5,2.5 0 0,0 16.5,18A2.5,2.5 0 0,0 19,15.5A2.5,2.5 0 0,0 16.5,13Z" /></svg>)
              }
            </div>
            <div className="message-content">
              {/* Use dangerouslySetInnerHTML ONLY if you trust the LLM output or sanitize it first.
                  Otherwise, render as plain text to prevent XSS.
                  For simplicity, rendering as plain text here. Replace if markdown needed. */}
              <div
                className={`message-text${msg.streaming ? ' streaming' : ''}`}
                style={{ whiteSpace: 'pre-wrap' }}
              >
                {msg.content}
              </div>
              {msg.role === 'assistant' &&
                msg.streaming &&
                streamPhase &&
                index === messages.length - 1 && (
                  <div className="stream-phase">{streamPhase}</div>
                )}
              <div className="message-meta">
                <span className="timestamp">{formatTime(msg.timestamp)}</span>
              </div>
              {/* Source Display */}
              {msg.role === 'assistant' &&
                !msg.streaming &&
                msg.sources &&
                msg.sources.length > 0 && (
                  <div className="message-sources">
                    <div className="sources-header">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z" /></svg>
                      <span>Sources:</span>
                    </div>
                    <ul className="sources-list">
                      {msg.sources.map((source, idx) => (
                        <li key={idx} className="source-item" title={`Score: ${source.score?.toFixed(3) ?? 'N/A'}`}>
                          <span className="source-page">Page {source.page}</span>
                          <span className="source-document">{source.document || pdfName}</span>
                          {/* Optional: Visual score indicator */}
                          {source.score !== undefined && source.score !== null && (
                            <span className="source-score">
                              <div className="score-bar" style={{ width: `${Math.max(0, Math.min(source.score * 100, 100))}%` }}></div>
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
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <div className="input-container">
          <input
            id="chat-input-field" // Added ID for potential focus targeting
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the document..."
            onFocus={() => setShowCommands(true)}
            onBlur={() => setTimeout(() => setShowCommands(false), 150)} // Hide dropdown on blur
            required
            disabled={!pdf || isStreaming}
          />
          <button
            type="button"
            className="command-button"
            onClick={() => setShowCommands(!showCommands)}
            title="Show sample commands"
            disabled={!pdf}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M7,10L12,15L17,10H7Z" /></svg>
          </button>
        </div>
        {showCommands && pdf && (
          <div className="commands-dropdown">
            {sampleCommands.map((cmd, index) => (
              <div
                key={index}
                className="command-item"
                onMouseDown={() => handleCommandClick(cmd.text)} // Use onMouseDown
              >
                <div className="command-text">{cmd.text}</div>
                <div className="command-description">{cmd.description}</div>
              </div>
            ))}
          </div>
        )}
        <button type="submit" className="send-button" disabled={!input.trim() || isStreaming || !pdf}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M2,21L23,12L2,3V10L17,12L2,14V21Z" /></svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
