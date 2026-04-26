// FILE: backend/routes/api.js

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDFModel = require('../models/pdf');
const logger = console;

// Persistent Python query server (replaces subprocess-per-query)
const PYTHON_QUERY_URL = process.env.PYTHON_QUERY_URL || 'http://127.0.0.1:5001';

// Multer setup with file validation
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
  logger.info(`Created uploads directory: ${uploadsDir}`);
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => { cb(null, uploadsDir); },
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${Date.now()}-${safe}`);
  }
});

// File filter for PDF validation
const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'application/pdf' || file.originalname.toLowerCase().endsWith('.pdf')) {
    cb(null, true);
  } else {
    cb(new Error('Only PDF files are allowed'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024  // 50MB limit
  }
});

const sendSse = (res, event) => {
  if (!res.writableEnded) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isTransientMongoError = (err) => {
  if (!err) return false;
  const labels = err.errorLabelSet || err.errorLabels;
  const hasResetPool = labels && typeof labels.has === 'function' && labels.has('ResetPool');
  return (
    hasResetPool ||
    err.code === 'ECONNRESET' ||
    err.cause?.code === 'ECONNRESET' ||
    /ECONNRESET|ResetPool|HandshakeError/i.test(err.message || '')
  );
};

const withMongoRetry = async (operation, label, attempts = 4) => {
  let lastErr;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation();
    } catch (err) {
      lastErr = err;
      if (!isTransientMongoError(err) || attempt === attempts) {
        throw err;
      }
      const delayMs = 250 * attempt;
      logger.warn(`${label} hit transient Mongo error; retrying in ${delayMs}ms (${attempt}/${attempts})`);
      await sleep(delayMs);
    }
  }
  throw lastErr;
};

const inferIngestPhase = (line) => {
  const text = String(line || '');
  if (text.includes('[Docling] Converting') || text.includes('[PyMuPDF4LLM] Extracting')) {
    return { phase: 'extracting', detail: 'Reading PDF structure' };
  }
  if (text.includes('[Docling] Chunking') || text.includes('Got ') || text.includes('Produced')) {
    return { phase: 'chunking', detail: 'Building text chunks' };
  }
  if (text.includes('Generated ') && text.includes('embeddings')) {
    return { phase: 'embedding', detail: 'Embedding chunks' };
  }
  if (text.includes('Upserting') || text.includes('Upserted batch') || text.includes('Upsert successful')) {
    return { phase: 'qdrant_store', detail: 'Writing vectors' };
  }
  if (text.includes('[BM25]')) {
    return { phase: 'bm25_index', detail: 'Building sparse index' };
  }
  if (text.includes('[KG] Starting entity extraction')) {
    return { phase: 'kg_extract', detail: 'Extracting triples' };
  }
  if (text.includes('[KG] Graph saved') || text.includes('[KG] Generated')) {
    return { phase: 'kg_store', detail: 'Saving graph' };
  }
  return null;
};

// --- MODIFIED /upload route: Uses spawn, waits for completion ---
router.post('/upload', (req, res) => {
  upload.single('file')(req, res, async (err) => {
    // Handle multer errors
    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ success: false, message: 'File too large. Maximum size is 50MB.' });
      }
      logger.error('Multer error:', err);
      return res.status(400).json({ success: false, message: `Upload error: ${err.message}` });
    } else if (err) {
      logger.error('Upload error:', err);
      return res.status(400).json({ success: false, message: err.message || 'File upload failed' });
    }

    logger.info('Received file upload request.');
    if (!req.file) {
      logger.warn('No file uploaded.');
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }
    logger.info(`File received: ${req.file.originalname}, stored as ${req.file.filename}`);

    let savedPdf;
    try {
      const pdf = new PDFModel({ /* ... pdf details ... */
        filename: req.file.filename, originalName: req.file.originalname, path: req.file.path,
        size: req.file.size, mimeType: req.file.mimetype || 'application/pdf', pageCount: 0, processed: false
      });
      savedPdf = await pdf.save();
      const pdfId = savedPdf._id.toString();
      logger.info(`PDF metadata saved (ID: ${pdfId}), starting processing...`);

      // --- Use spawn and wrap in Promise to wait ---
      const runEmbeddingScript = () => new Promise((resolve, reject) => {
        const pythonExecutable = 'python'; // Or 'python3' depending on container setup
        const pythonScript = path.resolve(__dirname, '../../python/compute_embeddings.py');
        const pdfFilePath = req.file.path;
        // Arguments for the script, including the required pdf_id
        const pythonArgs = [pythonScript, pdfFilePath, '--pdf_id', pdfId];

        // Use absolute path for script if running from different context
        logger.info(`Spawning: ${pythonExecutable} ${pythonArgs.join(' ')}`);

        // Pass environment variables explicitly to ensure they're available in the spawned process
        const env = {
          ...process.env,
          QDRANT_HOST: process.env.QDRANT_HOST || 'qdrant',
          QDRANT_PORT: process.env.QDRANT_PORT || '6333'
        };
        logger.info(`Using QDRANT_HOST=${env.QDRANT_HOST}, QDRANT_PORT=${env.QDRANT_PORT}`);

        const pythonProcess = spawn(pythonExecutable, pythonArgs, { env });

        let scriptOutput = "";
        let scriptError = "";
        pythonProcess.stdout.on('data', (data) => { scriptOutput += data.toString(); });
        pythonProcess.stderr.on('data', (data) => { logger.error(`Embedding stderr: ${data}`); scriptError += data.toString(); });

        pythonProcess.on('close', async (code) => {
          logger.info(`Embedding script exited code ${code} for PDF ID: ${pdfId}`);
          if (code === 0 && scriptOutput) {
            try {
              const result = JSON.parse(scriptOutput);
              if (result.success) {
                logger.info(`Embedding successful for ${pdfId}. Embeddings: ${result.embeddings_count}, Pages: ${result.page_count}`);
                const updatedPdf = await withMongoRetry(
                  () => PDFModel.findByIdAndUpdate(pdfId, { pageCount: result.page_count, processed: true }, { new: true }),
                  `Mark PDF ${pdfId} processed`,
                );
                if (updatedPdf) { resolve({ success: true, pdf: updatedPdf }); }
                else { reject(new Error(`Failed to find PDF ${pdfId} after processing.`)); }
              } else { reject(new Error(`Embedding script failed: ${result.error || 'Unknown script error'}`)); }
            } catch (parseError) { logger.error(`Parse Error: ${parseError}\nOutput: ${scriptOutput}`); reject(new Error('Failed to parse embedding script result.')); }
          } else { reject(new Error(`Embedding script failed (code ${code}). ${scriptError || 'Check logs.'}`)); }
        });
        pythonProcess.on('error', (spawnError) => { logger.error(`Spawn Error:`, spawnError); reject(new Error(`Failed to start embedding process: ${spawnError.message}`)); });
      }); // End Promise

      // Await the script completion
      const processingResult = await runEmbeddingScript();

      // Clear Python query server pipeline cache so re-uploaded PDFs are fresh
      try {
        const fetch = (await import('node-fetch')).default;
        await fetch(`${PYTHON_QUERY_URL}/cache/clear`, { method: 'POST', timeout: 2000 });
        logger.info('Python pipeline cache cleared after upload.');
      } catch (_cacheErr) {
        logger.warn('Could not clear Python pipeline cache (server may not be running yet).');
      }

      // Send final success response
      res.status(200).json({
        success: true, message: 'File uploaded and processed.', pdf: processingResult.pdf
      });

    } catch (err) {
      logger.error('Error during upload/processing:', err);
      if (savedPdf) {
        try {
          await withMongoRetry(
            () => PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }),
            `Mark PDF ${savedPdf._id} failed`,
          );
        } catch (dbErr) {
          logger.error("DB Update failed on error:", dbErr);
        }
      }
      res.status(500).json({ success: false, message: err.message || 'Upload/Processing Error' });
    }
  });  // Close the upload.single wrapper
});
// --- END MODIFIED /upload route ---

router.post('/upload/stream', (req, res) => {
  upload.single('file')(req, res, async (err) => {
    if (err instanceof multer.MulterError) {
      const message = err.code === 'LIMIT_FILE_SIZE'
        ? 'File too large. Maximum size is 50MB.'
        : `Upload error: ${err.message}`;
      return res.status(400).json({ success: false, message });
    } else if (err) {
      return res.status(400).json({ success: false, message: err.message || 'File upload failed' });
    }
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    if (typeof res.flushHeaders === 'function') {
      res.flushHeaders();
    }

    let savedPdf;
    const seenPhases = new Set();
    const emitPhase = (phase, detail) => {
      if (!seenPhases.has(phase)) {
        seenPhases.add(phase);
        sendSse(res, { phase, detail });
      }
    };

    try {
      emitPhase('upload_pdf', req.file.originalname);
      const pdf = new PDFModel({
        filename: req.file.filename,
        originalName: req.file.originalname,
        path: req.file.path,
        size: req.file.size,
        mimeType: req.file.mimetype || 'application/pdf',
        pageCount: 0,
        processed: false,
      });
      savedPdf = await pdf.save();
      const pdfId = savedPdf._id.toString();
      sendSse(res, { status: 'received', pdfId, filename: req.file.originalname });

      const pythonScript = path.resolve(__dirname, '../../python/compute_embeddings.py');
      const pythonArgs = [pythonScript, req.file.path, '--pdf_id', pdfId, '--reset'];
      const env = {
        ...process.env,
        QDRANT_HOST: process.env.QDRANT_HOST || 'qdrant',
        QDRANT_PORT: process.env.QDRANT_PORT || '6333',
      };
      logger.info(`Streaming upload spawn: python ${pythonArgs.join(' ')}`);

      const pythonProcess = spawn('python', pythonArgs, { env });
      let scriptOutput = '';
      let scriptError = '';

      pythonProcess.stdout.on('data', (data) => {
        scriptOutput += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        const chunk = data.toString();
        scriptError += chunk;
        logger.error(`Embedding stderr: ${chunk}`);
        chunk.split(/\r?\n/).forEach((line) => {
          const phase = inferIngestPhase(line);
          if (phase) emitPhase(phase.phase, phase.detail);
        });
      });

      pythonProcess.on('close', async (code) => {
        try {
          if (code !== 0 || !scriptOutput) {
            throw new Error(`Embedding script failed (code ${code}). ${scriptError || 'Check logs.'}`);
          }
          const result = JSON.parse(scriptOutput);
          if (!result.success) {
            throw new Error(`Embedding script failed: ${result.error || 'Unknown script error'}`);
          }
          emitPhase('finalizing', 'Refreshing document library');
          const updatedPdf = await withMongoRetry(
            () => PDFModel.findByIdAndUpdate(
              pdfId,
              { pageCount: result.page_count, processed: true },
              { new: true },
            ),
            `Mark PDF ${pdfId} processed`,
          );
          try {
            const fetch = (await import('node-fetch')).default;
            await fetch(`${PYTHON_QUERY_URL}/cache/clear`, { method: 'POST', timeout: 2000 });
          } catch (_cacheErr) {
            logger.warn('Could not clear Python pipeline cache after streamed upload.');
          }
          sendSse(res, {
            done: true,
            phase: 'ingest_done',
            pdf: updatedPdf,
            result,
          });
          res.end();
        } catch (closeErr) {
          logger.error('Streaming upload processing failed:', closeErr);
          if (savedPdf) {
            try {
              await withMongoRetry(
                () => PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }),
                `Mark PDF ${savedPdf._id} failed`,
              );
            } catch (_dbErr) { /* ignore */ }
          }
          sendSse(res, { error: closeErr.message || 'Upload processing failed' });
          res.end();
        }
      });

      pythonProcess.on('error', (spawnError) => {
        logger.error('Streaming upload spawn error:', spawnError);
        sendSse(res, { error: `Failed to start embedding process: ${spawnError.message}` });
        res.end();
      });
    } catch (streamErr) {
      logger.error('Streaming upload route error:', streamErr);
      if (savedPdf) {
        try {
          await withMongoRetry(
            () => PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }),
            `Mark PDF ${savedPdf._id} failed`,
          );
        } catch (_dbErr) { /* ignore */ }
      }
      sendSse(res, { error: streamErr.message || 'Upload/Processing Error' });
      res.end();
    }
  });
});

// Get all PDFs (include processed status)
router.get('/pdfs', async (req, res) => {
  // ... (same as previous version) ...
  logger.info('Fetching PDFs...'); try { const pdfs = await PDFModel.find({}, { filename: 1, originalName: 1, size: 1, pageCount: 1, uploadDate: 1, processed: 1 }).sort({ uploadDate: -1 }); res.status(200).json({ success: true, pdfs }); } catch (err) { logger.error('Fetch PDFs failed:', err); res.status(500).json({ success: false, message: 'Error fetching PDFs' }); }
});

// --- /query route: HTTP to persistent Python query server ---
router.post('/query', async (req, res) => {
  logger.info('Received query request.');
  try {
    const { pdfId, query, history } = req.body;

    // Validation
    if (!pdfId || !query) {
      return res.status(400).json({
        success: false,
        message: 'PDF ID and query are required'
      });
    }

    if (typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Query must be a non-empty string'
      });
    }

    if (query.length > 1000) {
      return res.status(400).json({
        success: false,
        message: 'Query too long. Maximum 1000 characters.'
      });
    }

    // Check if PDF is processed
    const pdf = await PDFModel.findById(pdfId, { processed: 1 });
    if (!pdf) { return res.status(404).json({ success: false, message: 'PDF not found' }); }
    if (pdf.processed !== true) { return res.status(400).json({ success: false, message: 'PDF is still processing or failed.' }); }

    logger.info(`Querying PDF ID: ${pdfId} via persistent Python server`);

    const fetch = (await import('node-fetch')).default;
    const pyResponse = await fetch(`${PYTHON_QUERY_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        pdf_id: pdfId,
        collection_name: 'documents',
        history: (history && Array.isArray(history)) ? history : [],
      }),
      timeout: 120000,  // 2 minute timeout
    });

    const result = await pyResponse.json();

    if (pyResponse.ok) {
      res.status(200).json({
        success: true,
        answer: result.answer,
        sources: result.sources || [],
      });
    } else {
      logger.error(`Python query server returned ${pyResponse.status}:`, result);
      res.status(pyResponse.status).json({
        success: false,
        message: result.answer || 'Query processing failed',
      });
    }
  } catch (err) {
    logger.error('Query Route Error:', err.message || err);
    res.status(500).json({ success: false, message: 'Server Error: ' + (err.message || 'Unknown') });
  }
});

router.post('/query/stream', async (req, res) => {
  logger.info('Received streaming query request.');
  try {
    const { pdfId, query, history } = req.body;

    if (!pdfId || !query) {
      return res.status(400).json({
        success: false,
        message: 'PDF ID and query are required',
      });
    }

    if (typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Query must be a non-empty string',
      });
    }

    if (query.length > 1000) {
      return res.status(400).json({
        success: false,
        message: 'Query too long. Maximum 1000 characters.',
      });
    }

    const pdf = await PDFModel.findById(pdfId, { processed: 1 });
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }
    if (pdf.processed !== true) {
      return res
        .status(400)
        .json({ success: false, message: 'PDF is still processing or failed.' });
    }

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    if (typeof res.flushHeaders === 'function') {
      res.flushHeaders();
    }

    const fetch = (await import('node-fetch')).default;
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const onClose = () => {
      if (controller) {
        try {
          controller.abort();
        } catch (_e) {
          /* ignore */
        }
      }
    };
    req.on('close', onClose);

    const pyRes = await fetch(`${PYTHON_QUERY_URL}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        pdf_id: pdfId,
        collection_name: 'documents',
        history: (history && Array.isArray(history)) ? history : [],
      }),
      timeout: 300000,
      signal: controller ? controller.signal : undefined,
    });

    if (!pyRes.ok) {
      let msg = 'Query processing failed';
      try {
        const errBody = await pyRes.text();
        const parsed = errBody && JSON.parse(errBody);
        if (parsed && (parsed.message || parsed.answer)) {
          msg = parsed.message || parsed.answer;
        } else if (errBody) {
          msg = errBody.slice(0, 200);
        }
      } catch (_p) {
        /* keep default */
      }
      res.write(`data: ${JSON.stringify({ error: msg })}\n\n`);
      res.end();
      return;
    }

    if (pyRes.body) {
      pyRes.body.on('error', (err) => {
        logger.error('Python stream read error:', err);
        if (!res.writableEnded) {
          res.write(`data: ${JSON.stringify({ error: 'Stream interrupted' })}\n\n`);
          res.end();
        }
      });
      pyRes.body.pipe(res);
    } else {
      res.write(
        `data: ${JSON.stringify({ error: 'No response body from query server' })}\n\n`,
      );
      res.end();
    }
  } catch (err) {
    logger.error('Query stream route error:', err.message || err);
    if (!res.headersSent) {
      return res
        .status(500)
        .json({ success: false, message: 'Server Error: ' + (err.message || 'Unknown') });
    }
    res.write(
      `data: ${JSON.stringify({ error: err.message || 'Server error' })}\n\n`,
    );
    res.end();
  }
});
// --- END /query route ---


// Reset route
router.post('/reset', async (req, res) => {
  logger.info('Received request to reset system.');
  try {
    await PDFModel.deleteMany({});

    // Clear uploads directory
    const uploadsDir = path.resolve(__dirname, '../uploads');
    fs.readdir(uploadsDir, (err, files) => {
      if (err) logger.error('Error reading uploads directory:', err);
      else {
        files.forEach(file => {
          if (file !== '.gitkeep') {
            const targetPath = path.join(uploadsDir, file);
            fs.rm(targetPath, { recursive: true, force: true }, err => {
              if (err) logger.error(`Error deleting path ${file}:`, err);
            });
          }
        });
      }
    });

    logger.info(`Running Qdrant reset script...`);
    const pythonScript = path.resolve(__dirname, '../../python/utils/qdrant_utils.py');

    // Pass environment variables explicitly to ensure they're available in the spawned process
    const env = {
      ...process.env,
      QDRANT_HOST: process.env.QDRANT_HOST || 'qdrant',
      QDRANT_PORT: process.env.QDRANT_PORT || '6333'
    };
    logger.info(`Using QDRANT_HOST=${env.QDRANT_HOST}, QDRANT_PORT=${env.QDRANT_PORT}`);

    const pythonProcess = spawn('python', [pythonScript, 'reset_collection'], { env });

    pythonProcess.stdout.on('data', (data) => {
      logger.info(`Reset script output: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      logger.error(`Reset script error: ${data}`);
    });

    res.status(200).json({ success: true, message: 'System reset initiated.' });
  } catch (err) {
    logger.error('Reset failed:', err);
    res.status(500).json({ success: false, message: 'Error resetting system' });
  }
});

module.exports = router;
