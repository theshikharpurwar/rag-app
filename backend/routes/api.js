// FILE: backend/routes/api.js (Full Code)

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDFModel = require('../models/pdf');
const logger = console; // Use a proper logger if available

// Setup Multer Storage
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    logger.info(`Created uploads directory: ${uploadsDir}`);
}
const storage = multer.diskStorage({
  destination: (req, file, cb) => { cb(null, uploadsDir); },
  filename: (req, file, cb) => {
    const safeOriginalName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    const timestamp = Date.now();
    cb(null, `${timestamp}-${safeOriginalName}`);
  }
});
const upload = multer({ storage });

// --- MODIFIED /upload route to wait for embedding ---
router.post('/upload', upload.single('file'), async (req, res) => {
  logger.info('Received file upload request.');
  if (!req.file) {
    logger.warn('No file uploaded with the request.');
    return res.status(400).json({ success: false, message: 'No file uploaded' });
  }
  logger.info(`File received: ${req.file.originalname}, stored as ${req.file.filename}`);

  let savedPdf; // Declare here to access in finally block if needed
  try {
    // 1. Save initial metadata to MongoDB
    const pdf = new PDFModel({
      filename: req.file.filename,
      originalName: req.file.originalname,
      path: req.file.path,
      size: req.file.size,
      mimeType: req.file.mimetype || 'application/pdf',
      pageCount: 0, // Initial count
      processed: false // Add a status field
    });
    savedPdf = await pdf.save();
    const pdfId = savedPdf._id.toString();
    logger.info(`PDF metadata saved (ID: ${pdfId}), status: processing`);

    // 2. Run Embedding Script and WAIT for completion
    const runEmbeddingScript = () => new Promise((resolve, reject) => {
      const pythonScript = path.resolve(__dirname, '../../python/compute_embeddings.py');
      const pdfFilePath = req.file.path;
      const pythonArgs = [pythonScript, pdfFilePath, '--pdf_id', pdfId];

      logger.info(`Spawning Embedding Script: python "${pythonScript}" "${pdfFilePath}" --pdf_id ${pdfId}`);
      const pythonProcess = spawn('python', pythonArgs);

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
              // Update DB with final status and page count
              const updatedPdf = await PDFModel.findByIdAndUpdate(
                pdfId,
                { pageCount: result.page_count, processed: true }, // Mark as processed
                { new: true }
              );
              if (updatedPdf) {
                 logger.info(`Updated PDF status and page count in DB for ${pdfId}`);
                 resolve({ success: true, pdf: updatedPdf }); // Resolve with updated PDF data
              } else {
                 logger.error(`Failed to find PDF ${pdfId} in DB after processing.`);
                 reject(new Error(`Failed to update PDF status after processing.`));
              }
            } else {
              logger.error(`Embedding script reported failure for ${pdfId}: ${result.error}`);
              reject(new Error(`Embedding script failed: ${result.error || 'Unknown script error'}`));
            }
          } catch (parseError) {
            logger.error(`Error parsing embedding script JSON output for ${pdfId}: ${parseError}`);
            logger.error(`Raw output: ${scriptOutput}`);
            reject(new Error('Failed to parse embedding script result.'));
          }
        } else {
           logger.error(`Embedding script failed for ${pdfId} with code ${code}. Stderr: ${scriptError}`);
           reject(new Error(`Embedding script failed (code ${code}). ${scriptError || 'Check logs.'}`));
        }
      });

      pythonProcess.on('error', (spawnError) => {
        logger.error(`Failed to spawn embedding script for ${pdfId}:`, spawnError);
        reject(new Error(`Failed to start embedding process: ${spawnError.message}`));
      });
    }); // End of Promise definition

    // 3. Await the embedding process completion
    const processingResult = await runEmbeddingScript();

    // 4. Send success response *after* processing is done
    res.status(200).json({
        success: true,
        message: 'File uploaded and processed successfully.',
        pdf: { // Send back final PDF info
            _id: processingResult.pdf._id,
            originalName: processingResult.pdf.originalName,
            pageCount: processingResult.pdf.pageCount,
            processed: processingResult.pdf.processed // Include status
        }
    });

  } catch (err) {
    logger.error('Error during file upload or processing:', err);
    // Optionally update PDF status to 'error' in DB if savedPdf exists
    if (savedPdf) {
        try {
            await PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }); // Mark as not processed on error
            logger.info(`Marked PDF ${savedPdf._id} as not processed due to error.`);
        } catch (dbErr) {
            logger.error(`Failed to update PDF status on error: ${dbErr}`);
        }
    }
    res.status(500).json({ success: false, message: err.message || 'Error uploading or processing file' });
  }
});
// --- END MODIFIED /upload route ---


// Get all PDFs (Add 'processed' status)
router.get('/pdfs', async (req, res) => {
  logger.info('Received request to fetch PDFs.');
  try {
    // Include the 'processed' field
    const pdfs = await PDFModel.find({}, { filename: 1, originalName: 1, size: 1, pageCount: 1, uploadDate: 1, processed: 1 }).sort({ uploadDate: -1 });
    res.status(200).json({ success: true, pdfs });
  } catch (err) {
    logger.error('Error fetching PDFs from MongoDB:', err);
    res.status(500).json({ success: false, message: 'Error fetching PDFs' });
  }
});

// Query the RAG model (Pass pdf_id)
router.post('/query', async (req, res) => {
  // ... (Keep the query logic exactly as provided in the previous step) ...
  // ... (It correctly receives pdfId and passes it to local_llm.py) ...
    logger.info('Received query request.');
    try {
        const { pdfId, query, history } = req.body;
        if (!pdfId || !query) { logger.warn('Missing pdfId or query.'); return res.status(400).json({ success: false, message: 'PDF ID and query are required' }); }
        logger.info(`Querying PDF ID: ${pdfId}`);
        const pythonScript = path.resolve(__dirname, '../../python/local_llm.py');
        const pythonArgs = [ pythonScript, query, '--collection_name', 'documents', '--pdf_id', pdfId ];
        if (history && Array.isArray(history)) { pythonArgs.push('--history', JSON.stringify(history)); }
        logger.info(`Attempting to spawn query script with arguments:`); logger.info(JSON.stringify(pythonArgs, null, 2));
        const pythonProcess = spawn('python', pythonArgs);
        let pythonOutput = ''; let pythonError = '';
        pythonProcess.stdout.on('data', (data) => { pythonOutput += data.toString(); });
        pythonProcess.stderr.on('data', (data) => { logger.error(`Query script stderr: ${data}`); pythonError += data.toString(); });
        pythonProcess.on('close', (code) => { logger.info(`Query script exited code ${code}`); /* ... handle response/error ... */
           if (code === 0 && pythonOutput) { try { const result = JSON.parse(pythonOutput); logger.info(`Query processed. Answer: ${result.answer.substring(0,50)}...`); res.status(200).json({ success: true, answer: result.answer, sources: result.sources || [] }); } catch (err) { logger.error('Parse Error:', err); logger.error('Raw output:', pythonOutput); res.status(500).json({ success: false, message: 'Parse Error', error: err.message, rawOutput: pythonOutput }); } } else { logger.error(`Script Fail. Code: ${code}. Err: ${pythonError}. Out: ${pythonOutput}`); res.status(500).json({ success: false, message: 'Script Error', error: pythonError || `Code ${code}` }); }
        });
       pythonProcess.on('error', (spawnError) => { logger.error('Spawn Error:', spawnError); res.status(500).json({ success: false, message: 'Spawn Error', error: spawnError.message }); });
    } catch (err) { logger.error('Query Route Error:', err); res.status(500).json({ success: false, message: 'Server Error' }); }
});

// Reset the system
router.post('/reset', async (req, res) => {
    // ... (Keep the reset logic as provided previously) ...
    logger.info('Received request to reset system.');
    try {
        await PDFModel.deleteMany({}); logger.info('Cleared PDF MongoDB collection.');
        if (fs.existsSync(uploadsDir)) { const items = fs.readdirSync(uploadsDir); for (const item of items) { if (item === '.gitkeep') continue; const itemPath = path.join(uploadsDir, item); try { const stats = fs.lstatSync(itemPath); if (stats.isDirectory()) { fs.rmSync(itemPath, { recursive: true, force: true }); } else { fs.unlinkSync(itemPath); } } catch (err) { logger.error(`Failed to delete ${itemPath}: ${err}`); } } logger.info('Cleared uploads directory.'); }
        const pythonScript = path.resolve(__dirname, '../../python/utils/qdrant_utils.py'); logger.info(`Running Qdrant reset script...`); const pythonProcess = spawn('python', [pythonScript, 'reset_collection', '--collection_name', 'documents']); /* ... handle process output ... */
        res.status(200).json({ success: true, message: 'System reset initiated.' });
    } catch (err) { logger.error('Reset failed:', err); res.status(500).json({ success: false, message: 'Error resetting system' }); }
});

module.exports = router;