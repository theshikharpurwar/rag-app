// routes/api.js
const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const Document = require('../models/Document');
const ModelConfig = require('../models/ModelConfig');
const config = require('../config/default.json');
const fetch = require('node-fetch');

const router = express.Router();

// Ensure Qdrant collection exists
async function ensureQdrantCollection() {
  try {
    const url = config.qdrant.url || 'http://localhost:6333';
    const collectionName = config.qdrant.collectionName;
    
    // Check if collection exists
    const response = await fetch(`${url}/collections/${collectionName}`);
    
    if (response.status === 404) {
      console.log(`Creating Qdrant collection: ${collectionName}`);
      
      // Create collection with appropriate vector size
      const createResponse = await fetch(`${url}/collections/${collectionName}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          vectors: {
            size: 512,  // Default size for CLIP embeddings
            distance: 'Cosine'
          }
        })
      });
      
      if (createResponse.ok) {
        console.log(`Successfully created Qdrant collection: ${collectionName}`);
      } else {
        console.error(`Failed to create Qdrant collection: ${await createResponse.text()}`);
      }
    } else if (response.ok) {
      console.log(`Qdrant collection exists: ${collectionName}`);
    } else {
      console.error(`Error checking Qdrant collection: ${await response.text()}`);
    }
  } catch (error) {
    console.error(`Error ensuring Qdrant collection: ${error.message}`);
  }
}

// Call this when the server starts
ensureQdrantCollection();

// Get active model configurations
const getActiveModels = async () => {
  const embeddingModel = await ModelConfig.findOne({ type: 'embedding', isActive: true });
  const llmModel = await ModelConfig.findOne({ type: 'llm', isActive: true });
  
  return {
    embedding: embeddingModel || { 
      name: 'clip',  // Changed from colpali to clip
      path: 'openai/clip-vit-base-patch32',  // Changed to CLIP model
      parameters: {}
    },
    llm: llmModel || {
      name: 'qwen',
      path: 'Qwen/Qwen2.5-VL-3B-Instruct',
      parameters: {}
    }
  };
};

// Upload PDF
router.post('/upload/pdf', async (req, res) => {
  if (!req.files || !req.files.pdf) {
    return res.status(400).json({ error: 'PDF file is required' });
  }

  const { pdf } = req.files;
  // Use timestamp to avoid filename conflicts
  const storedName = `${Date.now()}_${pdf.name.replace(/\s+/g, '_')}`;
  const pdfPath = path.join(config.directories.uploads, storedName);
  
  try {
    // Save PDF file
    await pdf.mv(pdfPath);
    
    // Create document in MongoDB
    const document = new Document({
      originalName: pdf.name,
      storedName: storedName,
      status: 'uploaded'
    });
    
    await document.save();
    
    // Get active models
    const activeModels = await getActiveModels();
    
    // Process PDF in the background
    const imageDir = path.join(config.directories.images, storedName.split('.')[0]);
    if (!fs.existsSync(imageDir)) {
      fs.mkdirSync(imageDir, { recursive: true });
    }
    
    // Update document status
    document.status = 'processing';
    await document.save();
    
    // Run Python script to compute embeddings with model params
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), '..', 'python', 'compute_embeddings.py'),
      pdfPath,
      config.qdrant.collectionName,
      imageDir,
      activeModels.embedding.name,
      activeModels.embedding.path,
      JSON.stringify(activeModels.embedding.parameters)
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log(`Python Output: ${data.toString()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python Error: ${data.toString()}`);
    });
    
    pythonProcess.on('close', async (code) => {
      try {
        if (code === 0 && outputData.trim()) {
          let result;
          try {
            result = JSON.parse(outputData);
          } catch (parseError) {
            console.error('Error parsing Python output:', parseError);
            document.status = 'failed';
            document.error = `Error parsing output: ${parseError.message}`;
            await document.save();
            return;
          }
          
          // Update document with page count and status
          document.pageCount = result.pageCount || 0;
          document.status = 'indexed';
          document.indexedAt = new Date();
          await document.save();
          
          console.log(`Document ${storedName} processed successfully`);
        } else {
          document.status = 'failed';
          document.error = errorData.substring(0, 500); // Truncate very long errors
          await document.save();
          
          console.error(`Error processing document ${storedName}: ${errorData}`);
        }
      } catch (error) {
        console.error('Error updating document status:', error);
        document.status = 'failed';
        document.error = error.message;
        await document.save();
      }
    });
    
    // Return immediately to user
    res.status(200).json({
      message: 'PDF uploaded successfully and processing started',
      documentId: document._id,
      storedName: storedName
    });
    
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get all documents
router.get('/documents', async (req, res) => {
  try {
    const documents = await Document.find().sort({ uploadedAt: -1 });
    res.json(documents);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Query document
router.post('/query', async (req, res) => {
  const { question, documentId } = req.body;
  
  if (!question || !documentId) {
    return res.status(400).json({ error: 'Question and document ID are required' });
  }
  
  try {
    // Find document
    const document = await Document.findById(documentId);
    if (!document) {
      return res.status(404).json({ error: 'Document not found' });
    }
    
    if (document.status !== 'indexed') {
      return res.status(400).json({ 
        error: 'Document is not ready for querying', 
        status: document.status 
      });
    }
    
    // Get active models
    const activeModels = await getActiveModels();
    
    // Directory where images are stored for this document
    const imageDir = path.join(config.directories.images, document.storedName.split('.')[0]);
    
    // Run Python script to query using the active LLM model
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), '..', 'python', 'local_llm.py'),
      question,
      document.storedName,
      config.qdrant.collectionName,
      imageDir,
      activeModels.embedding.name,
      activeModels.embedding.path,
      activeModels.llm.name,
      activeModels.llm.path,
      JSON.stringify(activeModels.llm.parameters)
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log(`Python Output: ${data.toString()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python Error: ${data.toString()}`);
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0 && outputData.trim()) {
        try {
          // Try to parse as JSON first
          const result = JSON.parse(outputData);
          res.json({ answer: result.answer || outputData });
        } catch (e) {
          // If not JSON, return as plain text
          res.json({ answer: outputData });
        }
      } else {
        res.status(500).json({ 
          error: 'Error generating response', 
          details: errorData 
        });
      }
    });
    
  } catch (error) {
    console.error('Query error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;