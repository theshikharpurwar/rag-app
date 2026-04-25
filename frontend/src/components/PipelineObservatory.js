import React, { useEffect, useMemo, useState, useRef } from 'react';
import { queryRAGStream } from '../api';
import './PipelineObservatory.css';

const NODES = [
  { id: 'upload_pdf', label: 'PDF Upload', icon: 'PDF', x: 56, y: 62, kind: 'ingest', detail: 'User drops a source document into the system.' },
  { id: 'extracting', label: 'Text Extraction', icon: 'EXT', x: 214, y: 62, kind: 'ingest', detail: 'PyMuPDF4LLM or Docling extracts structured document text.' },
  { id: 'chunking', label: 'Semantic Chunking', icon: 'CHK', x: 380, y: 62, kind: 'ingest', detail: 'Document text is split into retrievable overlapping chunks.' },
  { id: 'embedding', label: 'Embedding', icon: 'EMB', x: 548, y: 62, kind: 'ingest', detail: 'Chunks are embedded with the local Ollama embedding model.' },

  { id: 'qdrant_store', label: 'Qdrant Vectors', icon: 'VEC', x: 728, y: 22, kind: 'store', detail: 'Dense vectors are stored with page and PDF metadata.' },
  { id: 'bm25_index', label: 'BM25 Index', icon: 'BM25', x: 728, y: 132, kind: 'store', detail: 'Sparse keyword index supports exact phrase and numeric matches.' },
  { id: 'kg_extract', label: 'KG Extraction', icon: 'KG', x: 900, y: 76, kind: 'store', detail: 'Entities and relationships are extracted into triples.' },
  { id: 'kg_store', label: 'Graph Store', icon: 'KG+', x: 1072, y: 76, kind: 'store', detail: 'NetworkX graph, communities, and optional summaries are persisted.' },

  { id: 'question', label: 'Question', icon: 'Q', x: 56, y: 328, kind: 'query', detail: 'The user asks a document-grounded question.' },
  { id: 'routing', label: 'Adaptive Router', icon: 'RTE', x: 226, y: 328, kind: 'agent', detail: 'The agent classifies the query as specific, broad, or multi-hop.' },
  { id: 'decomposing', label: 'Decomposer', icon: 'DEC', x: 396, y: 242, kind: 'agent', detail: 'Complex or multi-hop questions are split into focused sub-queries.' },
  { id: 'vector_search', label: 'Vector Search', icon: 'VEC', x: 532, y: 312, kind: 'retrieve', detail: 'Semantic nearest-neighbor search retrieves candidate chunks.' },
  { id: 'bm25_search', label: 'BM25 Search', icon: 'BM25', x: 532, y: 430, kind: 'retrieve', detail: 'Keyword retrieval catches exact terms, names, dates, and numbers.' },
  { id: 'graph_search', label: 'Graph Traversal', icon: 'GPH', x: 704, y: 312, kind: 'retrieve', detail: 'Relevant entities seed BFS over graph neighborhoods.' },
  { id: 'community_search', label: 'Community Search', icon: 'SUM', x: 704, y: 430, kind: 'retrieve', detail: 'GraphRAG summaries supply global context for broad questions.' },
  { id: 'fusing', label: 'RRF / DBSF Fusion', icon: 'FUS', x: 888, y: 370, kind: 'fusion', detail: 'Dense, sparse, graph, and summary rankings are merged.' },
  { id: 'reranking', label: 'Reranker', icon: 'RNK', x: 1058, y: 370, kind: 'fusion', detail: 'Cross-encoder or ColBERT reranks fused candidates.' },
  { id: 'generating', label: 'LLM Generation', icon: 'LLM', x: 1058, y: 548, kind: 'answer', detail: 'The local LLM generates grounded prose from retrieved extracts.' },
  { id: 'grading', label: 'Grader', icon: 'GRD', x: 888, y: 548, kind: 'agent', detail: 'The answer is checked for faithfulness and relevance.' },
  { id: 'answer', label: 'Answer', icon: 'ANS', x: 704, y: 548, kind: 'answer', detail: 'The final answer streams back to the UI with sources.' },
];

const EDGES = [
  ['upload_pdf', 'extracting'],
  ['extracting', 'chunking'],
  ['chunking', 'embedding'],
  ['embedding', 'qdrant_store'],
  ['embedding', 'bm25_index'],
  ['chunking', 'kg_extract'],
  ['kg_extract', 'kg_store'],
  ['question', 'routing'],
  ['routing', 'decomposing'],
  ['routing', 'vector_search'],
  ['routing', 'bm25_search'],
  ['decomposing', 'vector_search'],
  ['decomposing', 'bm25_search'],
  ['qdrant_store', 'vector_search'],
  ['bm25_index', 'bm25_search'],
  ['kg_store', 'graph_search'],
  ['kg_store', 'community_search'],
  ['vector_search', 'fusing'],
  ['bm25_search', 'fusing'],
  ['graph_search', 'fusing'],
  ['community_search', 'fusing'],
  ['fusing', 'reranking'],
  ['reranking', 'generating'],
  ['generating', 'grading'],
  ['grading', 'answer'],
  ['grading', 'routing', 'retry'],
];

const PHASE_TO_NODE = {
  retrieving: 'vector_search',
  refining: 'grading',
  ingest_done: 'kg_store',
  error: 'error',
};

const NODE_BY_ID = Object.fromEntries(NODES.map((node) => [node.id, node]));
const ALL_NODE_IDS = NODES.map((node) => node.id);

function latestQueryView(events) {
  const start = events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => event.type === 'query' && event.phase === 'question')
    .pop();

  if (!start) return { text: '', running: false, error: null };

  const queryEvents = events.slice(start.index).filter((event) => event.type === 'query');
  const done = [...queryEvents].reverse().find((event) => event.done || event.answer);
  const error = [...queryEvents].reverse().find((event) => event.phase === 'error');
  if (error) return { text: '', running: false, error: error.detail || 'Query failed' };
  if (done?.answer) return { text: done.answer, running: false, error: null };

  const text = queryEvents.map((event) => event.token || '').join('');
  const running = queryEvents.some((event) => event.phase === 'generating') && !done;
  return { text, running, error: null };
}

function buildStates(events) {
  const states = Object.fromEntries(ALL_NODE_IDS.map((id) => [id, 'idle']));
  let lastNode = null;
  events.forEach((item) => {
    const node = PHASE_TO_NODE[item.phase] || item.phase;
    if (!node || node === 'error' || !states[node]) return;
    if (lastNode && states[lastNode] === 'active') states[lastNode] = 'done';
    if (states[node] !== 'done') states[node] = item.done ? 'done' : 'active';
    lastNode = node;
  });
  return states;
}

function edgeState(from, to, states) {
  if (states[to] === 'active') return 'active';
  if ((states[from] === 'done' || states[from] === 'active') && states[to] === 'done') return 'done';
  return 'idle';
}

function connectorPath(fromId, toId, type) {
  const from = NODE_BY_ID[fromId];
  const to = NODE_BY_ID[toId];
  const startX = from.x + 68;
  const startY = from.y + 38;
  const endX = to.x;
  const endY = to.y + 38;

  if (type === 'retry') {
    return `M ${from.x + 18} ${from.y + 38} C ${from.x - 120} ${from.y + 68}, ${to.x - 76} ${to.y - 74}, ${to.x + 16} ${to.y + 28}`;
  }

  const dx = Math.max(56, Math.abs(endX - startX) * 0.42);
  return `M ${startX} ${startY} C ${startX + dx} ${startY}, ${endX - dx} ${endY}, ${endX} ${endY}`;
}

const PipelineObservatory = ({ pdf, events = [], onPipelineEvent }) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [running, setRunning] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState('routing');

  const [isFullscreen, setIsFullscreen] = useState(false);
  const shellRef = useRef(null);

  const states = useMemo(() => buildStates(events), [events]);
  const queryView = useMemo(() => latestQueryView(events), [events]);
  const recentEvents = events.slice(-10).reverse();
  const terminalText = queryView.text || answer || queryView.error || localError;
  const isLive = running || queryView.running;
  const selectedNode = NODE_BY_ID[selectedNodeId] || NODE_BY_ID.routing;

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      shellRef.current?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  useEffect(() => {
    const latestAnswer = [...events].reverse().find((e) => e.answer);
    if (latestAnswer && latestAnswer.answer !== answer) {
      setAnswer(latestAnswer.answer);
    }
  }, [events, answer]);

  const emit = (payload) => {
    if (onPipelineEvent) onPipelineEvent(payload);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!pdf || !question.trim() || running) return;
    const currentQuestion = question.trim();
    setQuestion('');
    setAnswer('');
    setLocalError(null);
    setRunning(true);
    emit({ type: 'query', phase: 'question', query: currentQuestion });
    let streamedAnswer = '';

    await queryRAGStream(pdf._id, currentQuestion, [], {
      onPhase: (phase, detail) => {
        emit({ type: 'query', phase, detail, query: currentQuestion });
      },
      onStatus: (msg) => {
        emit({ type: 'query', phase: 'refining', detail: msg, query: currentQuestion });
      },
      onToken: (token) => {
        streamedAnswer += token;
        setAnswer((prev) => prev + token);
        emit({ type: 'query', phase: 'generating', token, query: currentQuestion });
      },
      onDone: ({ answer: finalAnswer, sources, trace }) => {
        const resolved = finalAnswer || streamedAnswer;
        setAnswer(resolved);
        setRunning(false);
        emit({
          type: 'query',
          phase: 'answer',
          done: true,
          answer: resolved,
          sources,
          trace,
          query: currentQuestion,
        });
      },
      onError: (message) => {
        setRunning(false);
        setLocalError(message || 'Query failed');
        emit({ type: 'query', phase: 'error', detail: message, query: currentQuestion });
      },
    });
  };

  return (
    <section className="pipeline-observatory">
      <div className="pipeline-toolbar">
        <div>
          <h2>Pipeline Observatory</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <p style={{ margin: 0 }}>{pdf ? pdf.originalName : 'Upload or select a PDF to animate the architecture.'}</p>
            <button type="button" className="btn btn-primary" onClick={toggleFullscreen} style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </button>
          </div>
        </div>
        <form className="pipeline-query" onSubmit={handleSubmit}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={pdf ? 'Ask through the live pipeline...' : 'Select a processed PDF first'}
            disabled={!pdf || running}
          />
          <button className="btn btn-primary" disabled={!pdf || running || !question.trim()}>
            {running ? 'Running' : 'Run'}
          </button>
        </form>
      </div>

      <div className="architecture-shell" ref={shellRef}>
        <div 
          className={`architecture-map ${isFullscreen ? 'fullscreen-active' : ''}`} 
          role="img" 
          aria-label="Interactive RAG architecture diagram"
        >
          <div className="arch-band ingest-band">Ingest</div>
          <div className="arch-band storage-band">Hybrid Store</div>
          <div className="arch-band query-band">Agentic Query</div>
          <div className="arch-band answer-band">Answer Loop</div>

          <svg className="arch-links" viewBox="0 0 1220 680" preserveAspectRatio="none">
            <defs>
              <marker id="arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" />
              </marker>
            </defs>
            {EDGES.map(([from, to, type]) => {
              const state = edgeState(from, to, states);
              return (
                <path
                  key={`${from}-${to}`}
                  className={`arch-link ${state} ${type || ''}`}
                  d={connectorPath(from, to, type)}
                  markerEnd="url(#arrow-head)"
                />
              );
            })}
          </svg>

          {NODES.map((node) => (
            <button
              key={node.id}
              type="button"
              className={`arch-node ${node.kind} ${states[node.id] || 'idle'} ${selectedNodeId === node.id ? 'selected' : ''}`}
              style={{ left: `${node.x}px`, top: `${node.y}px` }}
              onClick={() => setSelectedNodeId(node.id)}
              title={node.detail}
            >
              <span className="arch-node-icon">{node.icon}</span>
              <span className="arch-node-label">{node.label}</span>
              <span className="arch-node-state">{states[node.id] === 'done' ? 'done' : states[node.id]}</span>
            </button>
          ))}

          <aside className="arch-inspector">
            <span className={`inspector-pill ${selectedNode.kind}`}>{selectedNode.kind}</span>
            <h3>{selectedNode.label}</h3>
            <p>{selectedNode.detail}</p>
            <div className={`inspector-state ${states[selectedNode.id] || 'idle'}`}>
              {states[selectedNode.id] || 'idle'}
            </div>
          </aside>
        </div>
      </div>

      <div className="pipeline-bottom">
        <div className="pipeline-terminal">
          <div className="terminal-header">
            <span>Streaming Answer</span>
            {isLive && <span className="terminal-live">live</span>}
          </div>
          <pre>{terminalText || 'The answer stream appears here while the LLM node is active.'}</pre>
        </div>
        <div className="pipeline-events">
          <div className="events-header">Recent Events</div>
          {recentEvents.length ? recentEvents.map((event, idx) => (
            <div className="event-row" key={`${event.phase}-${idx}`}>
              <span>{event.type || 'system'}</span>
              <strong>{event.phase || event.status}</strong>
              {event.detail && <em>{event.detail}</em>}
            </div>
          )) : (
            <div className="event-empty">Upload a PDF or run a query to wake the diagram.</div>
          )}
        </div>
      </div>
    </section>
  );
};

export default PipelineObservatory;
