// D:\rag-app\backend\routes\api.js

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDF = require('../models/pdf');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '..', '..', 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const originalName = file.originalname;
    cb(null, `${timestamp}-${originalName}`);
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'), false);
    }
  }
});

// Upload PDF route
router.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    const pdfPath = req.file.path;
    const originalName = req.file.originalname;
    const fileName = req.file.filename;
    
    // Create output directory for images
    const outputDir = path.join(__dirname, '..', '..', 'uploads', 'images', fileName);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Collection name for Qdrant
    const collectionName = 'documents';
    
    // Run Python script to compute embeddings
    const pythonScript = path.join(__dirname, '..', '..', 'python', 'compute_embeddings.py');
    
    // Default model settings
    const modelName = req.body.modelName || 'clip';
    const modelPath = req.body.modelPath || 'openai/clip-vit-base-patch32';
    const modelParams = req.body.modelParams || '{}';
    
    const pythonProcess = spawn('python', [
      pythonScript,
      pdfPath,
      collectionName,
      outputDir,
      modelName,
      modelPath,
      modelParams
    ]);
    
    let pythonData = '';
    let pythonError = '';
    
    pythonProcess.stdout.on('data', (data) => {
      pythonData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });
    
    pythonProcess.on('close', async (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code !== 0) {
        return res.status(500).json({ 
          success: false, 
          message: 'Error processing PDF', 
          error: pythonError 
        });
      }
      
      try {
        const result = JSON.parse(pythonData);
        
        if (result.success) {
          // Save PDF info to MongoDB
          const newPDF = new PDF({
            fileName,
            originalName,
            path: pdfPath,
            pageCount: result.pageCount,
            embeddingCount: result.embeddingCount,
            collectionName
          });
          
          await newPDF.save();
          
          res.json({ 
            success: true, 
            message: 'PDF uploaded and processed successfully',
            pdf: newPDF
          });
        } else {
          res.status(500).json({ 
            success: false, 
            message: 'Error processing PDF', 
            error: result.error 
          });
        }
      } catch (err) {
        console.error('Error parsing Python output:', err);
        res.status(500).json({ 
          success: false, 
          message: 'Error parsing Python output', 
          error: err.message,
          pythonData
        });
      }
    });
  } catch (err) {
    console.error('Error uploading PDF:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Get all PDFs route
router.get('/pdfs', async (req, res) => {
  try {
    const pdfs = await PDF.find().sort({ createdAt: -1 });
    res.json({ success: true, pdfs });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});


// D:\rag-app\backend\routes\api.js

// Find the query route and update the spawn command:
router.post('/process', async (req, res) => {
    try {
      const { pdfId } = req.body;
      
      // Validate pdfId
      if (!pdfId) {
        return res.status(400).json({ success: false, message: 'PDF ID is required' });
      }
      
      // Get PDF info from MongoDB
      const pdf = await PDF.findById(pdfId);
      if (!pdf) {
        return res.status(404).json({ success: false, message: 'PDF not found' });
      }
      
      // Run Python script to compute embeddings
      const pythonScript = path.join(__dirname, '..', '..', 'python', 'compute_embeddings.py');
      
      // IMPORTANT: Fix the argument order and format
      const pythonProcess = spawn('python', [
        pythonScript,
        pdf.path,  // First positional argument should be the file path
        '--collection_name', 'documents',  // Named arguments with flags
        '--model_name', 'clip',
        '--model_path', 'ViT-B/32'
      ]);
    
    let pythonData = '';
    let pythonError = '';
    
    pythonProcess.stdout.on('data', (data) => {
      pythonData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code !== 0) {
        return res.status(500).json({ 
          success: false, 
          message: 'Error processing query', 
          error: pythonError 
        });
      }
      
      try {
        const result = JSON.parse(pythonData);
        res.json(result);
      } catch (err) {
        console.error('Error parsing Python output:', err);
        res.status(500).json({ 
          success: false, 
          message: 'Error parsing Python output', 
          error: err.message,
          pythonData
        });
      }
    });
} catch (err) {
    console.error('Error querying PDF:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;