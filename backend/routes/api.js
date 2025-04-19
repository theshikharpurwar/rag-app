// FILE: backend/routes/api.js (Corrected for Setup A)

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // Correctly import spawn
const PDFModel = require('../models/pdf');
const logger = console; // Use a proper logger if available

// Setup for LOCAL Execution relative to this file's location (backend/routes/)
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) { fs.mkdirSync(uploadsDir, { recursive: true }); }
const pythonDir = path.resolve(__dirname, '../../python');
const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3'; // Check this matches your host env PATH

const storage = multer.diskStorage({ destination: (req, file, cb) => { cb(null, uploadsDir); }, filename: (req, file, cb) => { const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_'); cb(null, `${Date.now()}-${safe}`); } });
const upload = multer({ storage });

// Upload route (spawns python, waits for completion)
router.post('/upload', upload.single('file'), async (req, res) => {
  logger.info('Upload request...');
  if (!req.file) { logger.warn('No file uploaded.'); return res.status(400).json({ success: false, message: 'No file uploaded' }); }
  logger.info(`File received: ${req.file.originalname}, stored as ${req.file.filename}`);

  let savedPdf; // Define here for access in catch block
  try {
    const pdf = new PDFModel({
        filename: req.file.filename, originalName: req.file.originalname, path: req.file.path,
        size: req.file.size, mimeType: req.file.mimetype || 'application/pdf', pageCount: 0, processed: false
    });
    savedPdf = await pdf.save();
    const pdfId = savedPdf._id.toString();
    logger.info(`PDF metadata saved (ID: ${pdfId}), starting processing...`);

    // --- Use spawn and wrap in Promise to wait ---
    const runEmbeddingScript = () => new Promise((resolve, reject) => {
      const pythonScript = path.join(pythonDir, 'compute_embeddings.py');
      const pdfFilePath = req.file.path; // Path accessible by Node.js
      const pythonArgs = [pythonScript, pdfFilePath, '--pdf_id', pdfId];

      logger.info(`Spawning: ${pythonExecutable} "${pythonScript}" "${pdfFilePath}" --pdf_id ${pdfId}`);
      const pythonProcess = spawn(pythonExecutable, pythonArgs);

      let scriptOutput = "";
      let scriptError = "";
      // Make sure to handle encoding correctly, especially on Windows
      pythonProcess.stdout.on('data', (data) => { scriptOutput += data.toString('utf8'); });
      pythonProcess.stderr.on('data', (data) => { logger.error(`Embedding stderr: ${data.toString('utf8')}`); scriptError += data.toString('utf8'); });

      pythonProcess.on('close', (code) => {
        logger.info(`Embedding script exited code ${code} for PDF ID: ${pdfId}`);
        if (code === 0 && scriptOutput) {
          try {
            const result = JSON.parse(scriptOutput);
            if (result.success) {
              logger.info(`Embedding successful for ${pdfId}. Pages: ${result.page_count}, Embeddings: ${result.embeddings_count}`);
              // Use async/await here as it's generally safe within event handlers in modern Node.js
              PDFModel.findByIdAndUpdate(pdfId, { pageCount: result.page_count, processed: true }, { new: true })
                .then(updatedPdf => {
                    if (updatedPdf) {
                        resolve({ success: true, pdf: updatedPdf }); // Resolve with final PDF data
                    } else {
                        reject(new Error(`Failed to find PDF ${pdfId} after processing.`));
                    }
                })
                .catch(dbErr => {
                    logger.error(`DB Update failed for ${pdfId} after success: ${dbErr}`);
                    reject(new Error(`DB update failed after processing: ${dbErr.message}`));
                });
            } else {
                // Script ran but reported failure
                logger.error(`Embedding script failed: ${result.error || 'Unknown script error'}`);
                reject(new Error(`Embedding script failed: ${result.error || 'Unknown script error'}`));
            }
          } catch (parseError) {
            // Failed to parse successful script output
            logger.error(`Parse Error: ${parseError}\nRaw Output: ${scriptOutput}`);
            reject(new Error('Failed parsing embedding script result.'));
          }
        } else {
          // Script exited with non-zero code or no output
           logger.error(`Embedding script failed (code ${code}). Err: ${scriptError}`);
           reject(new Error(`Embedding script failed (code ${code}). ${scriptError || 'Check logs.'}`));
        }
      }); // End 'close' handler

      pythonProcess.on('error', (spawnError) => {
        // Error spawning the process itself
        logger.error(`Spawn Error:`, spawnError);
        reject(new Error(`Failed to start embedding process: ${spawnError.message}`));
      });
    }); // End Promise

    // Await the script completion
    const processingResult = await runEmbeddingScript();

    // Send final success response ONLY if promise resolved
    res.status(200).json({
        success: true,
        message: 'File uploaded and processed successfully.',
        pdf: { // Send back final data
            _id: processingResult.pdf._id,
            originalName: processingResult.pdf.originalName,
            pageCount: processingResult.pdf.pageCount,
            processed: processingResult.pdf.processed
        }
    });

  } catch (err) { // Catches errors from pdf.save() or runEmbeddingScript() promise rejection
    logger.error('Error during upload/processing promise:', err);
    // Attempt to mark PDF as failed in DB if it was saved
    if (savedPdf?._id) {
        try {
            await PDFModel.findByIdAndUpdate(savedPdf._id, { processed: false }); // Keep processed: false
            logger.info(`Marked PDF ${savedPdf._id} as unprocessed due to error.`);
        } catch (dbErr) {
            logger.error("DB Update failed on error catch:", dbErr);
        }
    }
    res.status(500).json({ success: false, message: err.message || 'Upload/Processing Error' });
  }
});


// Get PDFs route
router.get('/pdfs', async (req, res) => {
    logger.info('Fetching PDFs...');
    try {
        const pdfs = await PDFModel.find({}, { filename: 1, originalName: 1, size: 1, pageCount: 1, uploadDate: 1, processed: 1 }).sort({ uploadDate: -1 });
        res.status(200).json({ success: true, pdfs });
    } catch (err) {
        logger.error('Fetch PDFs failed:', err);
        res.status(500).json({ success: false, message: 'Error fetching PDFs' });
    }
});


// Query route (uses spawn)
router.post('/query', async (req, res) => {
    logger.info('Query request...');
    try {
        const { pdfId, query, history } = req.body;
        if (!pdfId || !query) { return res.status(400).json({ success: false, message: 'ID/query required' }); }

        const pdf = await PDFModel.findById(pdfId, { processed: 1 });
        if (!pdf) { return res.status(404).json({ success: false, message: 'PDF not found' }); }
        if (pdf.processed !== true) { return res.status(400).json({ success: false, message: 'PDF not processed' }); }

        logger.info(`Querying PDF ID: ${pdfId}`);
        const pythonScript = path.join(pythonDir, 'local_llm.py');
        const pythonArgs = [ script, query, '--collection_name', 'documents', '--pdf_id', pdfId ];
        if (history && Array.isArray(history) && history.length > 0) { // Only add if history exists and is not empty
             try {
                 pythonArgs.push('--history', JSON.stringify(history));
             } catch (jsonErr) {
                 logger.error("Failed to stringify history:", jsonErr);
                 // Proceed without history? Or return error? Let's proceed without for now.
             }
        }

        logger.info(`Spawning: ${pythonExecutable} ${pythonArgs.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`);
        const pythonProcess = spawn(pythonExecutable, pythonArgs);

        let pythonOutput = '';
        let pythonError = '';
        pythonProcess.stdout.on('data', (data) => { pythonOutput += data.toString('utf8'); });
        pythonProcess.stderr.on('data', (data) => { logger.error(`Query stderr: ${data.toString('utf8')}`); pythonError += data.toString('utf8'); });

        pythonProcess.on('close', (code) => {
            logger.info(`Query script exited code ${code}`);
            if (code === 0 && pythonOutput) {
                try {
                    const result = JSON.parse(pythonOutput);
                    // Check if the result itself indicates an internal error from Python script
                    if(result && typeof result.answer === 'string' && (result.answer.startsWith("Error:") || result.answer.includes("LLM generation error"))){
                         logger.error(`Python script returned an error state: ${result.answer}`);
                         res.status(500).json({ success: false, message: result.answer, sources: result.sources || [] });
                    } else {
                        res.status(200).json({ success: true, answer: result.answer, sources: result.sources || [] });
                    }
                } catch (e) {
                    logger.error('Query Parse Error:', e); logger.error('Raw output:', pythonOutput);
                    res.status(500).json({ success: false, message: 'Failed to parse query result', error: e.message, rawOutput: pythonOutput });
                }
            } else {
                logger.error(`Query Script Fail Code: ${code}. Err: ${pythonError}. Out: ${pythonOutput}`);
                res.status(500).json({ success: false, message: 'Query script failed', error: pythonError || `Script exited code ${code}` });
            }
        });
        pythonProcess.on('error', (e) => {
             logger.error('Query Spawn Error:', e);
             res.status(500).json({ success: false, message: 'Failed to start query script', error: e.message });
        });
    } catch (err) {
        logger.error('Query Route Error:', err);
        res.status(500).json({ success: false, message: 'Server Error in query route' });
    }
});

// Reset route
router.post('/reset', async (req, res) => {
    logger.info('Reset request...');
    try {
        await PDFModel.deleteMany({}); logger.info('Cleared PDF MongoDB.');
        if (fs.existsSync(uploadsDir)) {
             fs.readdirSync(uploadsDir).forEach(item => {
                 if (item !== '.gitkeep') {
                      const itemPath = path.join(uploadsDir, item);
                      try { fs.rmSync(itemPath, { recursive: true, force: true }); }
                      catch (e) { logger.error(`Delete failed: ${e}`); }
                 }
             });
             logger.info('Cleared uploads directory.');
        }
        const pythonScript = path.join(pythonDir, 'utils/qdrant_utils.py');
        const args = [pythonScript, 'reset_collection', '--collection_name', 'documents'];
        logger.info(`Spawning: ${pythonExecutable} ${args.join(' ')}`);
        const proc = spawn(pythonExecutable, args);
        // Log output/errors but don't wait for reset script
        proc.stdout.on('data', (data) => logger.info(`Reset script stdout: ${data}`));
        proc.stderr.on('data', (data) => logger.error(`Reset script stderr: ${data}`));
        proc.on('close', code => logger.info(`Reset script exit code: ${code}`));
        res.status(200).json({ success: true, message: 'Reset initiated.' });
    } catch (err) { logger.error('Reset failed:', err); res.status(500).json({ success: false, message: 'Error resetting' }); }
});

module.exports = router;