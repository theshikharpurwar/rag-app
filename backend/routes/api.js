// D:\rag-app\backend\routes\api.js

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDFModel = require('../models/pdf');

// Set up multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage });

// Handle PDF uploads
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    // Create a new PDF document in MongoDB
    const pdf = new PDFModel({
      filename: req.file.filename,
      originalName: req.file.originalname,
      path: req.file.path,
      size: req.file.size,
      mimeType: req.file.mimetype || 'application/pdf',
      pageCount: 0 // Will be updated after processing
    });

    await pdf.save();

    // Process the PDF (generate embeddings)
    const pythonScript = path.resolve(__dirname, '../../python/compute_embeddings.py');
    const pythonProcess = spawn('python', [
      pythonScript,
      req.file.path,
      '--collection_name',
      'documents'
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });

    pythonProcess.on('close', async (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0 && pythonOutput) {
        try {
          const result = JSON.parse(pythonOutput);
          
          if (result.success && result.page_count) {
            // Update the page count in MongoDB
            await PDFModel.findByIdAndUpdate(
              pdf._id, 
              { pageCount: result.page_count },
              { new: true }
            );
            console.log(`Updated page count to ${result.page_count}`);
          }
        } catch (err) {
          console.error('Error parsing Python output:', err);
        }
      }
    });

    res.status(200).json({
      success: true,
      message: 'File uploaded successfully',
      pdf: {
        _id: pdf._id,
        filename: pdf.filename,
        originalName: pdf.originalName,
        pageCount: pdf.pageCount
      }
    });

  } catch (err) {
    console.error('Error uploading file:', err);
    res.status(500).json({ success: false, message: 'Error uploading file' });
  }
});

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  try {
    const pdfs = await PDFModel.find().sort({ createdAt: -1 });
    res.status(200).json({ success: true, pdfs });
  } catch (err) {
    console.error('Error fetching PDFs:', err);
    res.status(500).json({ success: false, message: 'Error fetching PDFs' });
  }
});

// Query the RAG model
router.post('/query', async (req, res) => {
  try {
    const { pdfId, query, modelPath } = req.body;

    if (!pdfId || !query) {
      return res.status(400).json({ success: false, message: 'PDF ID and query are required' });
    }

    // Find the PDF in the database
    const pdf = await PDFModel.findById(pdfId);
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }

    // Run the Python script to process the query
    const pythonScript = path.resolve(__dirname, '../../python/local_llm.py');
    
    // Prepare arguments 
    const pythonArgs = [pythonScript, query, '--collection_name', 'documents'];
    
    console.log(`Running query with args: ${pythonArgs.join(' ')}`);
    
    const pythonProcess = spawn('python', pythonArgs);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0) {
        try {
          // Parse the JSON output from the Python script
          const result = JSON.parse(pythonOutput);
          console.log('Query processed successfully with', result.sources ? result.sources.length : 0, 'sources');
          res.status(200).json({ 
            success: true, 
            answer: result.answer,
            sources: result.sources || [] 
          });
        } catch (err) {
          console.error('Error parsing Python output:', err);
          res.status(500).json({ 
            success: false, 
            message: 'Error processing query result',
            pythonOutput,
            error: err.message
          });
        }
      } else {
        res.status(500).json({ 
          success: false, 
          message: 'Error processing query',
          error: pythonError 
        });
      }
    });

  } catch (err) {
    console.error('Error processing query:', err);
    res.status(500).json({ success: false, message: 'Error processing query' });
  }
});

// Reset the system
router.post('/reset', async (req, res) => {
  try {
    // Delete all PDF records from MongoDB
    await PDFModel.deleteMany({});
    console.log('Deleted all PDF records from MongoDB');

    // Delete all files from uploads directory
    const uploadsDir = path.resolve(__dirname, '../../uploads');
    if (fs.existsSync(uploadsDir)) {
      const files = fs.readdirSync(uploadsDir);
      for (const file of files) {
        if (file !== '.gitkeep') {
          const filePath = path.join(uploadsDir, file);
          if (fs.lstatSync(filePath).isDirectory()) {
            // For directories, try to delete files inside first
            try {
              const subfiles = fs.readdirSync(filePath);
              for (const subfile of subfiles) {
                const subfilePath = path.join(filePath, subfile);
                if (fs.existsSync(subfilePath)) {
                  fs.unlinkSync(subfilePath);
                }
              }
              fs.rmdirSync(filePath);
            } catch (err) {
              console.error(`Error deleting directory ${filePath}:`, err);
            }
          } else {
            fs.unlinkSync(filePath);
          }
        }
      }
    }

    // Reset Qdrant collection
    const pythonScript = path.resolve(__dirname, '../../python/qdrant_utils.py');
    const pythonProcess = spawn('python', [
      pythonScript,
      'reset_collection',
      '--collection_name',
      'documents'
    ]);

    let pythonError = '';
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python reset process exited with code ${code}`);
    });

    res.status(200).json({ success: true, message: 'System reset successfully' });
  } catch (err) {
    console.error('Error resetting system:', err);
    res.status(500).json({ success: false, message: 'Error resetting system' });
  }
});

module.exports = router;