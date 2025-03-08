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
    const uploadDir = path.join(__dirname, '../../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
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

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  try {
    const pdfs = await PDF.find().sort({ uploadDate: -1 });
    res.json(pdfs);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// Upload a PDF
router.post('/upload', upload.single('pdf'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'No file uploaded' });
  }

  try {
    // Create a new PDF document in MongoDB
    const newPdf = new PDF({
      path: req.file.path,
      originalName: req.file.originalname,
      size: req.file.size,
      uploadDate: new Date()
    });

    const savedPdf = await newPdf.save();
    
    res.json({ 
      success: true, 
      message: 'File uploaded successfully', 
      pdf: savedPdf 
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// Process a PDF
router.post('/process', async (req, res) => {
  try {
    const { pdfId, modelName = 'all-MiniLM-L6-v2', collectionName = 'documents' } = req.body;
    
    // Find the PDF in the database
    const pdf = await PDF.findById(pdfId);
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }

    // Path to the Python script
    const pythonScript = path.join(__dirname, '../../python/compute_embeddings.py');
    
    // Run the Python script to process the PDF
    const pythonProcess = spawn('python', [
      pythonScript,
      pdf.path,
      '--collection_name',
      collectionName,
      '--model_name',
      modelName
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Python stderr:', data.toString());
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
        // Parse the Python output
        const result = JSON.parse(pythonOutput);
        
        // Update the PDF document with the processing results
        pdf.processed = true;
        pdf.pageCount = result.page_count;
        pdf.pointsStored = result.points_stored;
        pdf.processingDate = new Date();
        
        await pdf.save();
        
        res.json({ 
          success: true, 
          message: 'PDF processed successfully', 
          result 
        });
      } catch (err) {
        res.status(500).json({ 
          success: false, 
          message: 'Error parsing Python output', 
          error: err.message,
          pythonOutput
        });
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// Query the RAG system
router.post('/query', async (req, res) => {
  try {
    const { query, modelName = 'phi', collectionName = 'documents' } = req.body;
    
    if (!query) {
      return res.status(400).json({ success: false, message: 'Query is required' });
    }

    // Path to the Python script
    const pythonScript = path.join(__dirname, '../../python/local_llm.py');
    
    // Run the Python script to query the RAG system
    const pythonProcess = spawn('python', [
      pythonScript,
      query,
      '--collection_name',
      collectionName,
      '--model_name',
      modelName
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Python error:', data.toString());
      pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({ 
          success: false, 
          message: 'Error querying RAG system', 
          error: pythonError 
        });
      }

      try {
        // Parse the Python output
        const result = JSON.parse(pythonOutput);
        
        res.json({ 
          success: true, 
          result 
        });
      } catch (err) {
        res.status(500).json({ 
          success: false, 
          message: 'Error parsing Python output', 
          error: err.message,
          pythonOutput
        });
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// Reset the system (clear all data)
router.post('/reset', async (req, res) => {
  try {
    // Delete all PDFs from MongoDB
    await PDF.deleteMany({});
    
    // Function to recursively delete files and directories
    const deleteFilesRecursive = (directory) => {
      if (fs.existsSync(directory)) {
        try {
          const files = fs.readdirSync(directory);
          
          for (const file of files) {
            if (file !== '.gitkeep') {
              const curPath = path.join(directory, file);
              
              if (fs.lstatSync(curPath).isDirectory()) {
                // Recursive call for directories
                deleteFilesRecursive(curPath);
                try {
                  fs.rmdirSync(curPath);
                } catch (e) {
                  console.error(`Could not remove directory ${curPath}: ${e.message}`);
                }
              } else {
                // Delete file
                try {
                  fs.unlinkSync(curPath);
                } catch (e) {
                  console.error(`Could not delete file ${curPath}: ${e.message}`);
                }
              }
            }
          }
        } catch (e) {
          console.error(`Error reading directory ${directory}: ${e.message}`);
        }
      }
    };
    
    // Delete all files from the uploads directory
    const uploadsDir = path.join(__dirname, '../../uploads');
    deleteFilesRecursive(uploadsDir);
    
    // Ensure uploads and images directories exist
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    
    const imagesDir = path.join(uploadsDir, 'images');
    if (!fs.existsSync(imagesDir)) {
      fs.mkdirSync(imagesDir, { recursive: true });
    }
    
    // Reset the Qdrant collection
    const pythonScript = path.join(__dirname, '../../python/utils/qdrant_utils.py');
    const pythonProcess = spawn('python', [pythonScript, 'reset', 'documents']);
    
    let pythonOutput = '';
    let pythonError = '';
    
    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
      console.log(`Python output: ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error('Python stderr:', data.toString().trim());
      pythonError += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({
          success: false,
          message: 'Error resetting Qdrant collection',
          error: pythonError
        });
      }
      
      res.json({
        success: true,
        message: 'System reset successfully'
      });
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;