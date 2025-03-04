// D:\rag-app\backend\server.js

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { QdrantClient } = require("@qdrant/js-client-rest");
const mongoose = require('mongoose');
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/rag_app')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Define MongoDB schema for PDF metadata
const pdfSchema = new mongoose.Schema({
  filename: String,
  originalName: String,
  uploadDate: { type: Date, default: Date.now },
  pageCount: Number,
  embeddingCount: Number,
  processingStatus: String
});

const PDF = mongoose.model('PDF', pdfSchema);

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
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
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Ensure Qdrant collection exists
async function ensureQdrantCollectionExists() {
  try {
    const client = new QdrantClient({ url: "http://localhost:6333" });
    
    // Check if collection exists
    const collections = await client.getCollections();
    const collectionExists = collections.collections.some(
      (collection) => collection.name === "documents"
    );
    
    // Create collection if it doesn't exist
    if (!collectionExists) {
      await client.createCollection("documents", {
        vectors: {
          size: 512,  // CLIP model output size
          distance: "Cosine"
        }
      });
      console.log("Created Qdrant collection 'documents'");
    } else {
      console.log("Qdrant collection 'documents' already exists");
    }
  } catch (error) {
    console.error("Error ensuring Qdrant collection exists:", error);
  }
}

// Call the function when the server starts
ensureQdrantCollectionExists();

// API Routes
// Upload PDF
app.post('/api/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    // Save PDF metadata to MongoDB
    const newPDF = new PDF({
      filename: req.file.filename,
      originalName: req.file.originalname,
      processingStatus: 'uploaded'
    });
    
    await newPDF.save();
    
    // Process the PDF (convert to images and compute embeddings)
    const pdfPath = req.file.path;
    const outputDir = path.join(__dirname, '../uploads/images', path.basename(pdfPath, '.pdf'));
    
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Update processing status
    newPDF.processingStatus = 'processing';
    await newPDF.save();
    
    // Run Python script to compute embeddings
    const pythonScript = path.join(__dirname, '../python/compute_embeddings.py');
    const collectionName = 'documents';
    
    const pythonProcess = spawn('python', [
      pythonScript,
      pdfPath,
      collectionName,
      outputDir
    ]);
    
    let pythonOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });
    
    pythonProcess.on('close', async (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0) {
        try {
          // Parse the JSON output from the Python script
          const result = JSON.parse(pythonOutput);
          
          if (result.success) {
            // Update PDF metadata with page count and embedding count
            newPDF.pageCount = result.pageCount;
            newPDF.embeddingCount = result.embeddingCount;
            newPDF.processingStatus = 'completed';
            await newPDF.save();
            
            console.log(`PDF processed successfully: ${result.pageCount} pages, ${result.embeddingCount} embeddings`);
          } else {
            newPDF.processingStatus = 'failed';
            await newPDF.save();
            console.error(`Python script error: ${result.error}`);
          }
        } catch (error) {
          console.error('Error parsing Python output:', error);
          console.error('Python output:', pythonOutput);
          
          newPDF.processingStatus = 'failed';
          await newPDF.save();
        }
      } else {
        newPDF.processingStatus = 'failed';
        await newPDF.save();
      }
    });
    
    res.json({ 
      success: true, 
      message: 'PDF uploaded successfully', 
      pdf: {
        id: newPDF._id,
        filename: newPDF.filename,
        originalName: newPDF.originalName,
        uploadDate: newPDF.uploadDate,
        processingStatus: newPDF.processingStatus
      }
    });
  } catch (error) {
    console.error('Error uploading PDF:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Get all PDFs
app.get('/api/pdfs', async (req, res) => {
  try {
    const pdfs = await PDF.find().sort({ uploadDate: -1 });
    res.json({ success: true, pdfs });
  } catch (error) {
    console.error('Error fetching PDFs:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Get PDF by ID
app.get('/api/pdfs/:id', async (req, res) => {
  try {
    const pdf = await PDF.findById(req.params.id);
    if (!pdf) {
      return res.status(404).json({ success: false, message: 'PDF not found' });
    }
    res.json({ success: true, pdf });
  } catch (error) {
    console.error('Error fetching PDF:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Query RAG
app.post('/api/query', async (req, res) => {
  try {
    const { query, pdfId, modelName, modelPath } = req.body;
    
    if (!query) {
      return res.status(400).json({ success: false, message: 'Query is required' });
    }
    
    // Get PDF info if pdfId is provided
    let pdfPath = null;
    if (pdfId) {
      const pdf = await PDF.findById(pdfId);
      if (!pdf) {
        return res.status(404).json({ success: false, message: 'PDF not found' });
      }
      pdfPath = path.join(__dirname, '../uploads', pdf.filename);
    }
    
    // Run Python script for local LLM
    const pythonScript = path.join(__dirname, '../python/local_llm.py');
    
    const args = [pythonScript, query];
    
    if (pdfPath) {
      args.push('--pdf_path');
      args.push(pdfPath);
    }
    
    if (modelName) {
      args.push('--model_name');
      args.push(modelName);
    }
    
    if (modelPath) {
      args.push('--model_path');
      args.push(modelPath);
    }
    
    const pythonProcess = spawn('python', args);
    
    let pythonOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      
      if (code === 0) {
        try {
          // Parse the JSON output from the Python script
          const result = JSON.parse(pythonOutput);
          
          if (result.success) {
            res.json({ 
              success: true, 
              answer: result.answer,
              sources: result.sources || []
            });
          } else {
            res.status(500).json({ 
              success: false, 
              message: result.error || 'Unknown error in Python script'
            });
          }
        } catch (error) {
          console.error('Error parsing Python output:', error);
          console.error('Python output:', pythonOutput);
          
          res.status(500).json({ 
            success: false, 
            message: 'Error parsing Python output',
            pythonOutput
          });
        }
      } else {
        res.status(500).json({ 
          success: false, 
          message: `Python process exited with code ${code}`
        });
      }
    });
  } catch (error) {
    console.error('Error querying RAG:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});