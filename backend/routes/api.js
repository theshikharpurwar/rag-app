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
    cb(null, `${Date.now()}-${file.originalname}`);
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
  }
});

// Upload PDF - Updated to handle duplicates
router.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }
    
    // Check if a PDF with this name already exists
    const existingPDF = await PDF.findOne({ originalName: req.file.originalname });
    
    // If it exists, delete it and its file
    if (existingPDF) {
      console.log(`Found existing PDF with same name: ${existingPDF.originalName}. Deleting...`);
      
      // Remove file from storage if it exists
      if (existingPDF.path && fs.existsSync(existingPDF.path)) {
        try {
          fs.unlinkSync(existingPDF.path);
          console.log(`Deleted existing file: ${existingPDF.path}`);
        } catch (err) {
          console.error(`Error deleting existing file: ${err.message}`);
        }
      }
      
      // Delete from database
      await PDF.findByIdAndDelete(existingPDF._id);
      console.log(`Deleted existing PDF document with ID: ${existingPDF._id}`);
    }

    const pdf = new PDF({
      filename: req.file.filename,
      originalName: req.file.originalname,
      path: req.file.path,
      size: req.file.size,
      mimeType: req.file.mimetype || 'application/pdf' // Ensure mimeType is set
    });

    await pdf.save();
    console.log(`PDF saved to database with ID: ${pdf._id}`);

    // Process the PDF with Python script
    const pythonScript = path.join(__dirname, '../../python/compute_embeddings.py');
    
    const pythonProcess = spawn('python', [
      pythonScript,
      pdf.path,
      '--collection_name',
      'documents'
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      pythonError += data.toString();
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', async (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0) {
        try {
          const result = JSON.parse(pythonOutput);
          
          // Update PDF document with page count - enhanced to ensure it works
          console.log(`Updating PDF ${pdf._id} with page count: ${result.page_count}`);
          
          // Use findOneAndUpdate to get the updated document back
          const updatedPdf = await PDF.findByIdAndUpdate(
            pdf._id, 
            { 
              pageCount: result.page_count || 0,
              processed: true
            },
            { new: true } // Return updated document
          );
          
          if (updatedPdf) {
            console.log(`Successfully updated PDF: ${updatedPdf.originalName} with page count: ${updatedPdf.pageCount}`);
          } else {
            console.error(`Failed to update PDF ${pdf._id} - document not found`);
          }
        } catch (err) {
          console.error(`Error parsing Python output: ${err.message}`);
          console.error(`Raw Python output: ${pythonOutput}`);
          // Still mark as processed even if we can't parse the output
          await PDF.findByIdAndUpdate(pdf._id, { processed: true });
        }
      } else {
        console.error(`Error processing PDF: ${pythonError}`);
        // Mark as failed
        await PDF.findByIdAndUpdate(pdf._id, { 
          processed: true, 
          processingError: pythonError 
        });
      }
    });

    res.json({ 
      success: true, 
      message: 'PDF uploaded successfully', 
      data: { id: pdf._id, name: pdf.originalName } 
    });
  } catch (err) {
    console.error(`Error uploading PDF: ${err.message}`);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  try {
    console.log('Fetching PDFs from database...');
    const pdfs = await PDF.find().sort({ createdAt: -1 });
    console.log(`Found ${pdfs.length} PDFs`);
    
    // Log each PDF to help debug
    pdfs.forEach(pdf => {
      console.log(`PDF: ${pdf.originalName}, ID: ${pdf._id}, Pages: ${pdf.pageCount}, Processed: ${pdf.processed}`);
    });
    
    // Return just the array of PDFs for easier frontend consumption
    res.json(pdfs);
  } catch (err) {
    console.error(`Error fetching PDFs: ${err.message}`);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Query RAG
router.post('/query', async (req, res) => {
  try {
    const { pdfId, query } = req.body;
    
    if (!pdfId || !query) {
      return res.status(400).json({ success: false, message: 'PDF ID and query are required' });
    }

    const pdf = await PDF.findById(pdfId);
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }

    console.log(`Processing query "${query}" for PDF ${pdf.originalName}`);
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
      pythonError += data.toString();
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Query process exited with code ${code}: ${pythonError}`);
        return res.status(500).json({ success: false, message: `Error processing query: ${pythonError}` });
      }

      try {
        const result = JSON.parse(pythonOutput);
        console.log(`Query processed successfully with ${result.sources?.length || 0} sources`);
        res.json({ success: true, data: result });
      } catch (err) {
        console.error(`Error parsing Python output: ${err.message}, Output: ${pythonOutput}`);
        res.status(500).json({ success: false, message: `Error parsing result: ${err.message}` });
      }
    });
  } catch (err) {
    console.error(`Error querying RAG: ${err.message}`);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Reset system
router.post('/reset', async (req, res) => {
  try {
    console.log('Resetting system...');
    
    // 1. Delete all PDF documents from MongoDB
    const deleteResult = await PDF.deleteMany({});
    console.log(`Deleted ${deleteResult.deletedCount} PDF documents from MongoDB`);
    
    // 2. Delete all files from uploads directory
    const uploadsDir = path.join(__dirname, '..', '..', 'uploads');
    
    // Function to recursively delete directory contents
    const deleteDirectoryContents = (directory) => {
      if (fs.existsSync(directory)) {
        fs.readdirSync(directory).forEach((file) => {
          const curPath = path.join(directory, file);
          if (fs.lstatSync(curPath).isDirectory()) {
            // Recursive call for directories
            deleteDirectoryContents(curPath);
            try {
              fs.rmdirSync(curPath);
              console.log(`Deleted directory: ${curPath}`);
            } catch (err) {
              console.error(`Error deleting directory ${curPath}:`, err.message);
            }
          } else {
            // Delete file
            try {
              fs.unlinkSync(curPath);
              console.log(`Deleted file: ${curPath}`);
            } catch (err) {
              console.error(`Error deleting file ${curPath}:`, err.message);
            }
          }
        });
      }
    };
    
    // Delete contents but keep the main directories
    try {
      // Delete files directly in uploads folder
      fs.readdirSync(uploadsDir).forEach((file) => {
        const curPath = path.join(uploadsDir, file);
        if (fs.lstatSync(curPath).isDirectory()) {
          if (file === 'images') {
            // For images directory, just delete its contents
            deleteDirectoryContents(curPath);
          } else {
            // Delete other directories completely
            deleteDirectoryContents(curPath);
            try {
              fs.rmdirSync(curPath);
            } catch (err) {
              console.error(`Could not delete directory ${curPath}:`, err.message);
            }
          }
        } else {
          // Delete files
          try {
            fs.unlinkSync(curPath);
            console.log(`Deleted file: ${curPath}`);
          } catch (err) {
            console.error(`Error deleting file ${curPath}:`, err.message);
          }
        }
      });
      console.log('Cleared uploads directory');
    } catch (err) {
      console.error('Error clearing uploads directory:', err.message);
    }
    
    // 3. Reset Qdrant collection using Python utility
    const pythonScript = path.join(__dirname, '..', '..', 'python', 'utils', 'qdrant_utils.py');
    const pythonProcess = spawn('python', [
      pythonScript,
      'reset_collection',
      '--collection_name', 'documents',
      '--vector_size', '384'
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data;
      console.log(`Python output: ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data;
      console.error(`Python error: ${data.toString().trim()}`);
    });
    
    const exitCode = await new Promise((resolve) => {
      pythonProcess.on('close', resolve);
    });
    
    if (exitCode === 0) {
      console.log('Successfully reset Qdrant collection');
      res.json({ success: true, message: 'System reset successful' });
    } else {
      console.error(`Error resetting Qdrant collection. Exit code: ${exitCode}`);
      res.status(500).json({ 
        success: false, 
        message: 'Error resetting Qdrant collection', 
        error: errorData 
      });
    }
  } catch (err) {
    console.error('Error resetting system:', err);
    res.status(500).json({ success: false, message: 'Error resetting system', error: err.message });
  }
});

module.exports = router;