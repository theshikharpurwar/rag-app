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
const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/rag_app';
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log(`MongoDB connected at ${mongoURI}`))
.catch(err => console.error('MongoDB connection error:', err));

// Ensure uploads directory exists (corrected to point to backend/uploads)
const uploadsDir = path.join(__dirname, 'uploads'); // Changed from path.join(__dirname, '..', 'uploads')
console.log(`Uploads directory set to: ${uploadsDir}`);

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
    
    // Use path.resolve for more reliable path handling
    // In Docker, Python scripts are at /app/python
    const pythonScript = path.resolve(
      process.env.PYTHONPATH || path.join(__dirname, '..', 'python'), 
      'utils', 
      'qdrant_utils.py'
    );
    console.log(`Attempting to run Python script at: ${pythonScript}`);
    
    // Verify script exists
    if (!fs.existsSync(pythonScript)) {
      console.error(`Python script not found at: ${pythonScript}`);
      reject(new Error('Qdrant utility script not found'));
      return;
    }

    const pythonProcess = spawn('python3', [
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
    console.log(`Checking images directory: ${imagesDir}`);
    
    if (fs.existsSync(imagesDir)) {
      const files = fs.readdirSync(imagesDir);
      console.log(`Found ${files.length} files in images directory`);
      
      for (const file of files) {
        const filePath = path.join(imagesDir, file);
        try {
          if (fs.statSync(filePath).isFile()) {
            fs.unlinkSync(filePath);
            console.log(`Deleted image: ${filePath}`);
          }
        } catch (err) {
          console.error(`Error deleting image ${filePath}:`, err);
        }
      }
      
      // Try to remove the images directory if empty
      try {
        fs.rmdirSync(imagesDir);
        console.log('Removed images directory');
      } catch (err) {
        console.error('Error removing images directory:', err);
      }
    } else {
      console.log('Images directory does not exist');
    }

    // Clear main uploads directory
    console.log(`Checking uploads directory: ${uploadsDir}`);
    if (fs.existsSync(uploadsDir)) {
      const files = fs.readdirSync(uploadsDir);
      console.log(`Found ${files.length} files in uploads directory`);
      
      for (const file of files) {
        const filePath = path.join(uploadsDir, file);
        try {
          // Skip if it's a directory (we already handled images dir)
          if (fs.statSync(filePath).isDirectory()) {
            console.log(`Skipping directory: ${filePath}`);
            continue;
          }
          
          fs.unlinkSync(filePath);
          console.log(`Deleted file: ${filePath}`);
        } catch (err) {
          console.error(`Error deleting file ${filePath}:`, err);
        }
      }
    } else {
      console.log('Uploads directory does not exist');
    }
    
    console.log('Uploads directory cleared successfully');
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
      try {
        await mongoose.connection.close();
        console.log('MongoDB connection closed');
      } catch (err) {
        console.error('Error closing MongoDB connection:', err);
      }
    }
    
    // 5. Close server
    try {
      await new Promise((resolve) => {
        server.close(() => {
          console.log('Server closed');
          resolve();
        });
      });
    } catch (err) {
      console.error('Error closing server:', err);
    }
    
    console.log('Cleanup summary:');
    console.log(`- Qdrant collection: ${qdrantCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log(`- MongoDB collections: ${mongoCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log(`- Uploads directory: ${uploadsCleared ? 'Cleared ✓' : 'Failed ✗'}`);
    console.log('Application cleanup completed.');
    
    return {
      qdrantCleared,
      mongoCleared,
      uploadsCleared
    };
  } catch (error) {
    console.error('Unexpected error during cleanup:', error);
    return {
      qdrantCleared,
      mongoCleared,
      uploadsCleared
    };
  }
}

// Register server shutdown handlers
process.on('SIGINT', async () => {
  console.log('\nServer shutting down (SIGINT)...');
  const result = await performFullCleanup();
  console.log('Shutdown complete with results:', result);
  process.exit(result.qdrantCleared && result.mongoCleared && result.uploadsCleared ? 0 : 1);
});

process.on('SIGTERM', async () => {
  console.log('\nServer shutting down (SIGTERM)...');
  const result = await performFullCleanup();
  console.log('Shutdown complete with results:', result);
  process.exit(result.qdrantCleared && result.mongoCleared && result.uploadsCleared ? 0 : 1);
});

process.on('uncaughtException', async (error) => {
  console.error('Uncaught Exception:', error);
  const result = await performFullCleanup();
  console.log('Shutdown complete with results:', result);
  process.exit(1);
});