// D:\rag-app\backend\server.js

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const PDF = require('./models/pdf');

// Import routes
const apiRoutes = require('./routes/api');

// Create Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/rag_app', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.error('MongoDB connection error:', err));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const imagesDir = path.join(uploadsDir, 'images');
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
}

// Routes
app.use('/api', apiRoutes);

// Start server
const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Function to clear Qdrant collection using Python utility
async function clearQdrantCollection(collectionName = 'documents') {
  return new Promise((resolve, reject) => {
    console.log(`Clearing Qdrant collection: ${collectionName}`);
    
    // Corrected path to the utils directory
    const pythonScript = path.join(__dirname, '..', 'python', 'utils', 'qdrant_utils.py');
    const pythonProcess = spawn('python', [
      pythonScript, 
      'reset_collection',
      '--collection_name', 
      collectionName
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log(`Python output: ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python error: ${data.toString().trim()}`);
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`Successfully cleared Qdrant collection: ${collectionName}`);
        resolve(outputData);
      } else {
        console.error(`Failed to clear Qdrant collection. Exit code: ${code}`);
        reject(new Error(`Python process exited with code ${code}: ${errorData}`));
      }
    });
  });
}

// Function to clear MongoDB collections
async function clearMongoDBCollections() {
  console.log('Clearing MongoDB collections...');
  
  try {
    if (mongoose.connection.readyState === 1) { // Check if connected (1 = connected)
      // Delete all PDF documents
      const result = await PDF.deleteMany({});
      console.log(`Deleted ${result.deletedCount} documents from PDF collection`);
      return true;
    } else {
      console.log('MongoDB not connected, no cleanup needed');
      return false;
    }
  } catch (error) {
    console.error('Error clearing MongoDB collections:', error);
    return false;
  }
}

// Function to clear uploads directory
async function clearUploadsDirectory() {
  console.log('Clearing uploads directory...');
  
  try {
    // Clear images directory
    const imagesDir = path.join(uploadsDir, 'images');
    if (fs.existsSync(imagesDir)) {
      // Read all files in the directory
      const files = fs.readdirSync(imagesDir);
      
      // Delete each file
      for (const file of files) {
        const filePath = path.join(imagesDir, file);
        fs.unlinkSync(filePath);
      }
      
      console.log(`Cleared ${files.length} files from images directory`);
    }

    // Clear main uploads directory (only PDF files)
    if (fs.existsSync(uploadsDir)) {
      const files = fs.readdirSync(uploadsDir);
      
      let deletedCount = 0;
      for (const file of files) {
        const filePath = path.join(uploadsDir, file);
        // Only delete files, not directories
        if (fs.statSync(filePath).isFile() && file.toLowerCase().endsWith('.pdf')) {
          fs.unlinkSync(filePath);
          deletedCount++;
        }
      }
      
      console.log(`Cleared ${deletedCount} PDF files from uploads directory`);
    }
    
    return true;
  } catch (error) {
    console.error('Error clearing uploads directory:', error);
    return false;
  }
}

// Comprehensive cleanup function that handles everything
async function performFullCleanup() {
  console.log('Performing full application cleanup...');
  
  let qdrantCleared = false;
  let mongoCleared = false;
  let uploadsCleared = false;
  
  try {
    // 1. Clear Qdrant
    try {
      await clearQdrantCollection('documents');
      qdrantCleared = true;
    } catch (error) {
      console.error('Qdrant cleanup failed:', error.message);
    }
    
    // 2. Clear MongoDB
    try {
      mongoCleared = await clearMongoDBCollections();
    } catch (error) {
      console.error('MongoDB cleanup failed:', error.message);
    }
    
    // 3. Clear uploads directory
    try {
      uploadsCleared = await clearUploadsDirectory();
    } catch (error) {
      console.error('Uploads directory cleanup failed:', error.message);
    }
    
    // 4. Close MongoDB connection
    if (mongoose.connection.readyState === 1) {
      await mongoose.connection.close();
      console.log('MongoDB connection closed');
    }
    
    console.log('Cleanup summary:');
    console.log(`- Qdrant collection: ${qdrantCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log(`- MongoDB collections: ${mongoCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log(`- Uploads directory: ${uploadsCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log('Application cleanup completed.');
    
  } catch (error) {
    console.error('Unexpected error during cleanup:', error);
  }
}

// Register server shutdown handlers
process.on('SIGINT', async () => {
  console.log('\nServer shutting down (SIGINT)...');
  await performFullCleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\nServer shutting down (SIGTERM)...');
  await performFullCleanup();
  process.exit(0);
});

// Close gracefully on uncaught exceptions as well
process.on('uncaughtException', async (error) => {
  console.error('Uncaught Exception:', error);
  await performFullCleanup();
  process.exit(1);
});