// FILE: backend/routes/api.js (Full Code with Modifications)

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDFModel = require('../models/pdf'); // Make sure the PDF model is imported
const logger = console; // Use a proper logger if available

// Setup Multer Storage
const uploadsDir = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    logger.info(`Created uploads directory: ${uploadsDir}`);
} else {
    logger.info(`Uploads directory found: ${uploadsDir}`);
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

// Handle PDF uploads --- MODIFIED TO PASS PDF ID ---
router.post('/upload', upload.single('file'), async (req, res) => {
  logger.info('Received file upload request.');
  try {
    if (!req.file) {
      logger.warn('No file uploaded with the request.');
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }
    logger.info(`File received: ${req.file.originalname}, stored as ${req.file.filename}`);

    const pdf = new PDFModel({
      filename: req.file.filename,
      originalName: req.file.originalname,
      path: req.file.path,
      size: req.file.size,
      mimeType: req.file.mimetype || 'application/pdf',
      pageCount: 0
    });

    // *** SAVE PDF TO GET THE ID ***
    const savedPdf = await pdf.save();
    const pdfId = savedPdf._id.toString(); // Get the MongoDB ID as string
    logger.info(`PDF metadata saved to MongoDB for ${pdf.originalName} (ID: ${pdfId})`);

    // --- Trigger Python Embedding Script ---
    const pythonScript = path.resolve(__dirname, '../../python/compute_embeddings.py');
    const pdfFilePath = req.file.path;
    // *** PASS PDF ID TO SCRIPT ***
    const pythonArgs = [pythonScript, pdfFilePath, '--pdf_id', pdfId];
    logger.info(`Spawning Python script: python "${pythonScript}" "${pdfFilePath}" --pdf_id ${pdfId}`);

    const pythonProcess = spawn('python', pythonArgs);

    let scriptOutput = "";
    let scriptError = "";
    pythonProcess.stdout.on('data', (data) => { scriptOutput += data.toString(); });
    pythonProcess.stderr.on('data', (data) => { logger.error(`Embedding script stderr: ${data}`); scriptError += data.toString(); });

    pythonProcess.on('close', async (code) => {
      logger.info(`Embedding script exited with code ${code} for ${pdf.originalName}`);
      if (code === 0 && scriptOutput) {
        try {
          const result = JSON.parse(scriptOutput);
          if (result.success && result.page_count !== undefined) {
            await PDFModel.findByIdAndUpdate(pdfId, { pageCount: result.page_count });
            logger.info(`Updated page count for ${pdf.originalName} to ${result.page_count}`);
          } else if (!result.success) {
             logger.error(`Embedding script reported failure: ${result.error || 'Unknown error'}`);
          }
        } catch (parseError) {
          logger.error(`Error parsing embedding script output JSON: ${parseError}`);
          logger.error(`Raw embedding script output: ${scriptOutput}`);
        }
      } else if (code !== 0) {
         logger.error(`Embedding script failed with code ${code}. Error: ${scriptError}`);
      }
    });
    pythonProcess.on('error', (spawnError) => {
        logger.error('Failed to spawn embedding script process:', spawnError);
        // Handle error appropriately, maybe update PDF status in DB
    });


    res.status(200).json({
      success: true,
      message: 'File uploaded successfully, processing started.',
      pdf: { _id: pdfId, originalName: pdf.originalName, pageCount: pdf.pageCount }
    });

  } catch (err) {
    logger.error('Error during file upload or DB save:', err);
    res.status(500).json({ success: false, message: 'Error uploading file' });
  }
});

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  logger.info('Received request to fetch PDFs.');
  try {
    const pdfs = await PDFModel.find().sort({ uploadDate: -1 });
    res.status(200).json({ success: true, pdfs });
  } catch (err) {
    logger.error('Error fetching PDFs from MongoDB:', err);
    res.status(500).json({ success: false, message: 'Error fetching PDFs' });
  }
});

// Query the RAG model --- MODIFIED TO PASS PDF ID ---
router.post('/query', async (req, res) => {
  logger.info('Received query request.');
  try {
    const { pdfId, query, history } = req.body; // Get history if frontend sends it

    if (!pdfId || !query) {
      logger.warn('Missing pdfId or query in request body.');
      return res.status(400).json({ success: false, message: 'PDF ID and query are required' });
    }

    // No need to fetch PDF here unless checking existence
    logger.info(`Querying for PDF ID: ${pdfId} with query: "${query.substring(0, 50)}..."`);

    const pythonScript = path.resolve(__dirname, '../../python/local_llm.py');
    // *** PASS PDF ID TO SCRIPT ***
    const pythonArgs = [
        pythonScript,
        query,
        '--collection_name', 'documents',
        '--pdf_id', pdfId // Pass the specific PDF ID for filtering
    ];

    // Pass history if available
    if (history && Array.isArray(history)) {
        pythonArgs.push('--history', JSON.stringify(history));
    }


    logger.info(`Running query script with args: ${pythonArgs.join(' ')}`);

    const pythonProcess = spawn('python', pythonArgs);

    let pythonOutput = '';
    let pythonError = '';
    pythonProcess.stdout.on('data', (data) => { pythonOutput += data.toString(); });
    pythonProcess.stderr.on('data', (data) => { logger.error(`Query script stderr: ${data}`); pythonError += data.toString(); });

    pythonProcess.on('close', (code) => {
      logger.info(`Query script exited with code ${code}`);
      if (code === 0 && pythonOutput) {
        try {
          const result = JSON.parse(pythonOutput);
          logger.info(`Query processed. Answer snippet: ${(result.answer || '').substring(0, 70)}...`);
          res.status(200).json({ success: true, answer: result.answer, sources: result.sources || [] });
        } catch (err) {
          logger.error('Error parsing query script JSON output:', err);
          logger.error('Raw query script output:', pythonOutput);
          res.status(500).json({ success: false, message: 'Error processing query script result.', error: err.message, rawOutput: pythonOutput });
        }
      } else {
         logger.error(`Query script failed. Code: ${code}. Stderr: ${pythonError}. Stdout: ${pythonOutput}`);
         res.status(500).json({ success: false, message: 'Error executing query script.', error: pythonError || `Script exited code ${code}` });
      }
    });
   pythonProcess.on('error', (spawnError) => {
        logger.error('Failed to spawn query script process:', spawnError);
        res.status(500).json({ success: false, message: 'Failed to start query script.', error: spawnError.message });
    });

  } catch (err) {
    logger.error('Error in /query route handler:', err);
    res.status(500).json({ success: false, message: 'Internal server error processing query.' });
  }
});

// Reset the system
router.post('/reset', async (req, res) => {
  logger.info('Received request to reset system.');
  try {
    // Delete MongoDB records
    const mongoResult = await PDFModel.deleteMany({});
    logger.info(`Deleted ${mongoResult.deletedCount} PDF records from MongoDB.`);

    // Clear uploads directory (robustly handle files/dirs)
    if (fs.existsSync(uploadsDir)) {
        logger.info(`Clearing uploads directory: ${uploadsDir}`);
        const items = fs.readdirSync(uploadsDir);
        let deletedFiles = 0;
        let deletedDirs = 0;
        for (const item of items) {
            if (item === '.gitkeep') continue; // Skip placeholder
            const itemPath = path.join(uploadsDir, item);
            try {
                const stats = fs.lstatSync(itemPath);
                if (stats.isDirectory()) {
                    // Use recursive delete for directories (like the 'images' folder)
                    fs.rmSync(itemPath, { recursive: true, force: true });
                    deletedDirs++;
                } else {
                    fs.unlinkSync(itemPath);
                    deletedFiles++;
                }
            } catch (err) {
                logger.error(`Failed to delete item ${itemPath}: ${err}`);
            }
        }
         logger.info(`Deleted ${deletedFiles} files and ${deletedDirs} directories from uploads.`);
    } else {
        logger.warn("Uploads directory does not exist, nothing to clear.");
    }

    // Reset Qdrant collection using the utility script
    const pythonScript = path.resolve(__dirname, '../../python/utils/qdrant_utils.py');
    logger.info(`Running Qdrant reset script: python "${pythonScript}" reset_collection --collection_name documents`);
    const pythonProcess = spawn('python', [pythonScript, 'reset_collection', '--collection_name', 'documents']);
    // ... (handle process output/errors as before) ...
     let pythonError = '';
     pythonProcess.stderr.on('data', (data) => { logger.error(`Qdrant reset script stderr: ${data}`); pythonError += data.toString(); });
     pythonProcess.on('close', (code) => { logger.info(`Qdrant reset script exited with code ${code}`); if (code !== 0) { logger.error(`Qdrant reset failed. Error: ${pythonError}`); } });
     pythonProcess.on('error', (spawnError) => { logger.error('Failed to spawn Qdrant reset script process:', spawnError); });

    res.status(200).json({ success: true, message: 'System reset initiated successfully.' });

  } catch (err) {
    logger.error('Error during system reset:', err);
    res.status(500).json({ success: false, message: 'Error resetting system' });
  }
});


module.exports = router;