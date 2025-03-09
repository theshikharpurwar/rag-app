// D:\rag-app\backend\server.js

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

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
    
    // CORRECTED PATH - Uses the utils directory
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

// Register server shutdown handlers
process.on('SIGINT', async () => {
  console.log('Server shutting down (SIGINT)...');
  
  try {
    await clearQdrantCollection('documents');
    console.log('Cleanup completed successfully.');
  } catch (error) {
    console.error('Error during cleanup:', error.message);
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\nServer shutting down (SIGTERM)...');
  try {
    await clearQdrantCollection();
    console.log('Cleanup completed successfully.');
  } catch (error) {
    console.error('Error during cleanup:', error.message);
  }
  process.exit(0);
});

// Close gracefully on uncaught exceptions as well
process.on('uncaughtException', async (error) => {
  console.error('Uncaught Exception:', error);
  try {
    await clearQdrantCollection();
    console.log('Cleanup completed successfully.');
  } catch (cleanupError) {
    console.error('Error during cleanup:', cleanupError.message);
  }
  process.exit(1);
});