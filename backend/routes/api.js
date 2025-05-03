// FILE: backend/routes/api.js (Full Code - Reverted to spawn, kept pdf_id passing)

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // Re-added spawn
const PDFModel = require('../models/pdf');
const logger = console;

// Multer setup (same as before)
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) { fs.mkdirSync(uploadsDir, { recursive: true }); logger.info(`Created uploads directory: ${uploadsDir}`); }
const storage = multer.diskStorage({ destination: (req, file, cb) => { cb(null, uploadsDir); }, filename: (req, file, cb) => { const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_'); cb(null, `${Date.now()}-${safe}`); } });
const upload = multer({ storage });

// --- MODIFIED /upload route: Uses spawn, waits for completion ---
router.post('/upload', upload.single('file'), async (req, res) => {
  logger.info('Received file upload request.');
  if (!req.file) { logger.warn('No file uploaded.'); return res.status(400).json({ success: false, message: 'No file uploaded' }); }
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
              const updatedPdf = await PDFModel.findByIdAndUpdate(pdfId, { pageCount: result.page_count, processed: true }, { new: true });
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

    // Send final success response
    res.status(200).json({
        success: true, message: 'File uploaded and processed.', pdf: processingResult.pdf
    });

  } catch (err) {
    logger.error('Error during upload/processing:', err);
    if (savedPdf) { try { await PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }); } catch (dbErr) { logger.error("DB Update failed on error:", dbErr); }}
    res.status(500).json({ success: false, message: err.message || 'Upload/Processing Error' });
  }
});
// --- END MODIFIED /upload route ---

// Get all PDFs (include processed status)
router.get('/pdfs', async (req, res) => {
    // ... (same as previous version) ...
    logger.info('Fetching PDFs...'); try { const pdfs = await PDFModel.find({}, { filename: 1, originalName: 1, size: 1, pageCount: 1, uploadDate: 1, processed: 1 }).sort({ uploadDate: -1 }); res.status(200).json({ success: true, pdfs }); } catch (err) { logger.error('Fetch PDFs failed:', err); res.status(500).json({ success: false, message: 'Error fetching PDFs' }); }
});

// --- MODIFIED /query route: Uses spawn, passes pdf_id ---
router.post('/query', async (req, res) => {
  logger.info('Received query request.');
  try {
    const { pdfId, query, history } = req.body;
    if (!pdfId || !query) { return res.status(400).json({ success: false, message: 'PDF ID and query are required' }); }

    // Optional: Check if PDF is processed before spawning python
    const pdf = await PDFModel.findById(pdfId, { processed: 1 });
    if (!pdf) { return res.status(404).json({ success: false, message: 'PDF not found' }); }
    if (pdf.processed !== true) { return res.status(400).json({ success: false, message: 'PDF is still processing or failed.' }); }

    logger.info(`Querying PDF ID: ${pdfId}`);
    const pythonExecutable = 'python'; // Or python3
    const pythonScript = path.resolve(__dirname, '../../python/local_llm.py');
    const pythonArgs = [ pythonScript, query, '--collection_name', 'documents', '--pdf_id', pdfId ];
    if (history && Array.isArray(history)) { pythonArgs.push('--history', JSON.stringify(history)); }

    logger.info(`Spawning: ${pythonExecutable} ${pythonArgs.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`); // Log command properly
    
    // Pass environment variables explicitly to ensure they're available in the spawned process
    const env = {
      ...process.env,
      QDRANT_HOST: process.env.QDRANT_HOST || 'qdrant', 
      QDRANT_PORT: process.env.QDRANT_PORT || '6333'
    };
    logger.info(`Using QDRANT_HOST=${env.QDRANT_HOST}, QDRANT_PORT=${env.QDRANT_PORT}`);
    
    const pythonProcess = spawn(pythonExecutable, pythonArgs, { env });

    let pythonOutput = ''; let pythonError = '';
    pythonProcess.stdout.on('data', (data) => { pythonOutput += data.toString(); });
    pythonProcess.stderr.on('data', (data) => { logger.error(`Query script stderr: ${data}`); pythonError += data.toString(); });

    pythonProcess.on('close', (code) => {
      logger.info(`Query script exited code ${code}`);
      if (code === 0 && pythonOutput) {
        try {
          const result = JSON.parse(pythonOutput);
          res.status(200).json({ success: true, answer: result.answer, sources: result.sources || [] });
        } catch (err) { logger.error('Parse Error:', err); logger.error('Raw output:', pythonOutput); res.status(500).json({ success: false, message: 'Parse Error', error: err.message, rawOutput: pythonOutput }); }
      } else { logger.error(`Script Fail. Code: ${code}. Err: ${pythonError}. Out: ${pythonOutput}`); res.status(500).json({ success: false, message: 'Script Error', error: pythonError || `Code ${code}` }); }
    });
   pythonProcess.on('error', (spawnError) => { logger.error('Spawn Error:', spawnError); res.status(500).json({ success: false, message: 'Spawn Error', error: spawnError.message }); });

  } catch (err) { logger.error('Query Route Error:', err); res.status(500).json({ success: false, message: 'Server Error' }); }
});
// --- END MODIFIED /query route ---


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
                        fs.unlink(path.join(uploadsDir, file), err => {
                            if (err) logger.error(`Error deleting file ${file}:`, err);
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