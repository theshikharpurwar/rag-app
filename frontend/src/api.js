// D:\rag-app\frontend\src\api.js

// Import API_URL from config
import { API_URL } from './config';

// Upload a PDF file
export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return { success: false, message: 'Error uploading file' };
  }
};

export const uploadPDFStream = async (file, callbacks = {}) => {
  const {
    onPhase = () => { },
    onStatus = () => { },
    onDone = () => { },
    onError = () => { },
  } = callbacks;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_URL}/upload/stream`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let msg = `Upload failed (${response.status})`;
      try {
        const j = await response.json();
        if (j.message) msg = j.message;
      } catch (_e) {
        /* keep msg */
      }
      onError(msg);
      return { success: false, message: msg };
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (value) {
        buffer += decoder.decode(value, { stream: true });
      }
      const { events, rest } = parseSseBlocks(buffer);
      buffer = rest;
      for (const data of events) {
        if (data.error) {
          onError(data.error);
          return { success: false, message: data.error };
        }
        if (data.phase) {
          onPhase(data.phase, data.detail, data);
        }
        if (data.status) {
          onStatus(data.status, data);
        }
        if (data.done) {
          onDone(data);
          return { success: true, pdf: data.pdf, result: data.result };
        }
      }
      if (done) break;
    }
    const msg = 'Upload stream ended without completion';
    onError(msg);
    return { success: false, message: msg };
  } catch (error) {
    console.error('Error streaming PDF upload:', error);
    const msg = error.message || 'Error uploading file';
    onError(msg);
    return { success: false, message: msg };
  }
};

// Fetch all PDFs
export const fetchPDFs = async () => {
  try {
    const response = await fetch(`${API_URL}/pdfs`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    return { success: false, message: 'Error fetching PDFs' };
  }
};

function parseSseBlocks(buffer) {
  const events = [];
  let rest = buffer;
  let idx;
  while ((idx = rest.indexOf('\n\n')) !== -1) {
    const block = rest.slice(0, idx).trim();
    rest = rest.slice(idx + 2);
    if (!block) continue;
    const lines = block.split('\n');
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith('data:')) continue;
      const jsonStr = t.slice(5).trim();
      if (!jsonStr) continue;
      try {
        events.push(JSON.parse(jsonStr));
      } catch (_e) {
        /* skip malformed chunk */
      }
    }
  }
  return { events, rest };
}

/**
 * POST /api/query/stream — streams SSE events: phase, token, done, error.
 */
export const queryRAGStream = async (pdfId, query, history = [], callbacks = {}) => {
  const {
    onToken = () => { },
    onPhase = () => { },
    onStatus = () => { },
    onDone = () => { },
    onError = () => { },
  } = callbacks;

  let streamCompleted = false;
  const markDone = (payload) => {
    streamCompleted = true;
    onDone(payload);
  };

  try {
    const response = await fetch(`${API_URL}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdfId, query, history }),
    });

    if (!response.ok) {
      let msg = `Request failed (${response.status})`;
      try {
        const j = await response.json();
        if (j.message) msg = j.message;
      } catch (_e) {
        /* keep msg */
      }
      onError(msg);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (value) {
        buffer += decoder.decode(value, { stream: true });
      }
      const { events, rest } = parseSseBlocks(buffer);
      buffer = rest;
      for (const data of events) {
        if (data.error) {
          onError(data.error);
          return;
        }
        if (data.token != null && data.token !== '') {
          onToken(data.token);
        }
        if (data.phase) {
          onPhase(data.phase);
        }
        if (data.status) {
          onStatus(data.message || '', data);
        }
        if (data.done) {
          markDone({
            sources: data.sources || [],
            trace: data.trace || {},
            answer: data.answer,
          });
          return;
        }
      }
      if (done) {
        break;
      }
    }
    if (buffer.trim()) {
      for (const line of buffer.split('\n')) {
        const t = line.trim();
        if (!t.startsWith('data:')) continue;
        const jsonStr = t.slice(5).trim();
        if (!jsonStr) continue;
        try {
          const data = JSON.parse(jsonStr);
          if (data.error) {
            onError(data.error);
            return;
          }
          if (data.done) {
            markDone({
              sources: data.sources || [],
              trace: data.trace || {},
              answer: data.answer,
            });
            return;
          }
        } catch (_e) {
          /* ignore */
        }
      }
    }
    if (!streamCompleted) {
      onError('Stream ended without a complete response');
    }
  } catch (error) {
    console.error('queryRAGStream:', error);
    onError(error.message || 'Network error');
  }
};

// Query the RAG model
export const queryRAG = async (pdfId, query, history = []) => {
  try {
    console.log(`Sending query to backend: ${query} for PDF: ${pdfId}`);

    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdfId,
        query,
        history
      }),
    });

    const data = await response.json();
    console.log('Received response from backend:', data);

    if (data.success) {
      return {
        answer: data.answer,
        sources: data.sources,
      };
    } else {
      console.error('API error:', data.message);
      return {
        answer: `Error: ${data.message}`,
        sources: [],
      };
    }
  } catch (error) {
    console.error('Error querying RAG:', error);
    return {
      answer: 'Error connecting to the server',
      sources: [],
    };
  }
};

// Reset the system
export const resetSystem = async () => {
  try {
    const response = await fetch(`${API_URL}/reset`, {
      method: 'POST',
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error resetting system:', error);
    return { success: false, message: 'Error resetting system' };
  }
};
