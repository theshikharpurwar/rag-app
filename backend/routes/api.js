// D:\rag-app\backend\routes\api.js

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const router = express.Router();
const PDF = require('../models/pdf');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueFilename = Date.now() + '-' + file.originalname;
    cb(null, uniqueFilename);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'));
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB max file size
  }
});

// Upload and process a PDF
router.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    // Process the PDF with Python
    const pythonScript = path.join(__dirname, '../../python/compute_embeddings.py');
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
      console.error('Python stderr:', data.toString());
      pythonError += data.toString();
    });
    
    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        console.error(`Process exited with code ${code}`);
        console.error(`Python Error: ${pythonError}`);
        return res.status(500).json({ 
          success: false, 
          message: 'Error processing PDF',
          error: pythonError
        });
      }
      
      try {
        // Parse Python output
        const result = JSON.parse(pythonOutput);
        
        if (result.success) {
          // Save PDF metadata to MongoDB
          const pdf = new PDF({
            filename: req.file.filename,
            originalName: req.file.originalname,
            path: req.file.path,
            size: req.file.size,
            pageCount: result.page_count,
            embeddingsCount: result.embeddings_count
          });
          
          await pdf.save();
          
          res.json({
            success: true,
            message: 'PDF uploaded and processed successfully',
            pdf: {
              id: pdf._id,
              filename: pdf.filename,
              originalName: pdf.originalName,
              pageCount: pdf.pageCount,
              embeddingsCount: pdf.embeddingsCount
            }
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
        console.error('Python output:', pythonOutput);
        res.status(500).json({
          success: false,
          message: 'Error parsing Python output',
          error: err.message
        });
      }
    });
  } catch (err) {
    console.error('Error in upload route:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  try {
    const pdfs = await PDF.find({});
    res.json(pdfs);
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// Query a PDF
router.post('/query', async (req, res) => {
  try {
    const { query } = req.body;
    
    if (!query) {
      return res.status(400).json({ success: false, message: 'No query provided' });
    }
    
    // Process the query with Python - always use phi2 model
    const pythonScript = path.join(__dirname, '../../python/local_llm.py');
    const pythonProcess = spawn('python', [
      pythonScript,
      query,
      '--collection_name',
      'documents'
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
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Process exited with code ${code}`);
        console.error(`Python Error: ${pythonError}`);
        return res.status(500).json({ 
          success: false, 
          message: 'Error processing query',
          error: pythonError
        });
      }
      
      try {
        // Parse Python output
        const result = JSON.parse(pythonOutput);
        res.json(result);
      } catch (err) {
        console.error('Error parsing Python output:', err);
        console.error('Python output:', pythonOutput);
        res.status(500).json({
          success: false,
          message: 'Error parsing Python output',
          error: err.message
        });
      }
    });
  } catch (err) {
    console.error('Error in query route:', err);
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
    const pythonProcess = spawn('python', [
      pythonScript,
      'reset',
      'documents',
      '384'  // Explicitly set vector size to 384
    ]);
    
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