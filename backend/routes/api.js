// D:\rag-app\backend\routes\api.js

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const PDF = require('../models/pdf');

// Configure multer storage for file uploads
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, path.join(__dirname, '..', '..', 'uploads'));
  },
  filename: function(req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: function(req, file, cb) {
    if (file.mimetype === 'application/pdf' || 
        file.mimetype === 'application/vnd.openxmlformats-officedocument.presentationml.presentation') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF and PPTX files are allowed'), false);
    }
  }
});

// Get all PDFs
router.get('/pdfs', async (req, res) => {
  try {
    const pdfs = await PDF.find().sort({ createdAt: -1 });
    res.json({ success: true, pdfs });
  } catch (err) {
    console.error('Error fetching PDFs:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Upload and process a PDF
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    const file = req.file;
    const apiKey = req.body.apiKey;
    const modelName = req.body.modelName || 'mistral';
    
    if (!apiKey) {
      return res.status(400).json({ success: false, message: 'API key is required' });
    }
    
    // Save PDF info to MongoDB
    const newPDF = new PDF({
      filename: file.filename,
      originalName: file.originalname,
      path: file.path,
      size: file.size,
      mimeType: file.mimetype,
      pageCount: 0  // Will be updated after processing
    });
    
    const savedPDF = await newPDF.save();

    // Process the PDF with Python script
    const pythonScript = path.join(__dirname, '..', '..', 'python', 'compute_embeddings.py');
    const pythonProcess = spawn('python', [
      pythonScript,
      file.path,
      '--collection_name', 'documents',
      '--model_name', modelName,
      '--api_key', apiKey
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log(`Python stdout: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python stderr: ${data}`);
    });
    
    pythonProcess.on('close', async (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0) {
        try {
          // Try to parse the output as JSON
          const result = JSON.parse(outputData);
          
          // Update the page count in MongoDB
          if (result.page_count) {
            await PDF.findByIdAndUpdate(savedPDF._id, { 
              pageCount: result.page_count 
            });
          }
          
          res.json({ 
            success: true, 
            message: 'PDF uploaded and processed successfully',
            pdf: {
              ...savedPDF.toObject(),
              pageCount: result.page_count || 0
            }
          });
        } catch (err) {
          console.error('Error parsing Python output:', err);
          res.json({ 
            success: true, 
            message: 'PDF uploaded but there may have been issues with processing',
            pdf: savedPDF
          });
        }
      } else {
        console.error(`Python processing failed with code ${code}: ${errorData}`);
        res.status(500).json({ 
          success: false, 
          message: 'Error processing PDF',
          error: errorData
        });
      }
    });
  } catch (err) {
    console.error('Error uploading PDF:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Query the RAG system
router.post('/query', async (req, res) => {
  try {
    const { query, pdfId, modelName, modelPath, apiKey } = req.body;
    
    if (!query) {
      return res.status(400).json({ success: false, message: 'Query is required' });
    }
    
    if (!apiKey) {
      return res.status(400).json({ success: false, message: 'API key is required' });
    }
    
    const pythonScript = path.join(__dirname, '..', '..', 'python', 'local_llm.py');
    
    // Build arguments for the Python script
    const pythonArgs = [
      pythonScript, 
      query, 
      '--collection_name', 'documents',
      '--api_key', apiKey
    ];
    
    if (modelName) {
      pythonArgs.push('--model_name', modelName);
    }
    
    if (modelPath) {
      pythonArgs.push('--model_path', modelPath);
    }
    
    const pythonProcess = spawn('python', pythonArgs);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData);
          res.json({ 
            success: true, 
            answer: result.answer,
            sources: result.sources || []
          });
        } catch (err) {
          console.error('Error parsing Python output:', err);
          res.status(500).json({ 
            success: false, 
            message: 'Error parsing response',
            error: err.message,
            output: outputData
          });
        }
      } else {
        console.error(`Python process exited with code ${code}: ${errorData}`);
        res.status(500).json({ 
          success: false, 
          message: 'Error processing query',
          error: errorData
        });
      }
    });
  } catch (err) {
    console.error('Error processing query:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// Route to clear Qdrant collection and uploaded files
router.post('/reset', async (req, res) => {
  try {
    console.log('Resetting application state...');
    
    // 1. Delete all PDFs from MongoDB
    const deleteResult = await PDF.deleteMany({});
    console.log(`Deleted ${deleteResult.deletedCount} documents from MongoDB`);
    
    // 2. Clear uploads directory
    const uploadsDir = path.join(__dirname, '..', '..', 'uploads');
    if (fs.existsSync(uploadsDir)) {
      const files = fs.readdirSync(uploadsDir);
      for (const file of files) {
        const filePath = path.join(uploadsDir, file);
        if (fs.lstatSync(filePath).isFile()) {
          fs.unlinkSync(filePath);
          console.log(`Deleted file: ${filePath}`);
        }
      }
    }
    
    // 3. Clear images directory
    const imagesDir = path.join(uploadsDir, 'images');
    if (fs.existsSync(imagesDir)) {
      const clearDir = (dir) => {
        if (fs.existsSync(dir)) {
          fs.readdirSync(dir).forEach((file) => {
            const curPath = path.join(dir, file);
            if (fs.lstatSync(curPath).isDirectory()) {
              clearDir(curPath);
            } else {
              fs.unlinkSync(curPath);
              console.log(`Deleted file: ${curPath}`);
            }
          });
        }
      };
      clearDir(imagesDir);
      console.log('Cleared images directory');
    }
    
    // 4. Clear Qdrant collection using the Python utility
    const pythonScript = path.join(__dirname, '..', '..', 'python', 'utils', 'qdrant_utils.py');
    
    // Execute the Python script
    const pythonProcess = spawn('python', [pythonScript]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python error: ${data}`);
    });
    
    // Wait for the Python process to complete
    const exitCode = await new Promise((resolve) => {
      pythonProcess.on('close', resolve);
    });
    
    if (exitCode === 0) {
      console.log('Successfully cleared Qdrant collection');
      
      // Send successful response
      res.json({
        success: true,
        message: 'Application reset successful',
        details: {
          mongoDocumentsDeleted: deleteResult.deletedCount,
          qdrantResult: outputData
        }
      });
    } else {
      throw new Error(`Python process exited with code ${exitCode}: ${errorData}`);
    }
  } catch (err) {
    console.error('Error resetting application:', err);
    res.status(500).json({ 
      success: false, 
      message: 'Error resetting application', 
      error: err.message 
    });
  }
});

// Delete a specific PDF
router.delete('/pdfs/:id', async (req, res) => {
  try {
    const pdf = await PDF.findById(req.params.id);
    
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }
    
    // Delete the file if it exists
    if (fs.existsSync(pdf.path)) {
      fs.unlinkSync(pdf.path);
    }
    
    // Delete from MongoDB
    await PDF.findByIdAndDelete(req.params.id);
    
    res.json({ success: true, message: 'PDF deleted successfully' });
  } catch (err) {
    console.error('Error deleting PDF:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;