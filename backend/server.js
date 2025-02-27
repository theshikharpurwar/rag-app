// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const path = require('path');
const fs = require('fs');
const config = require('./config/default.json');
const fetch = require('node-fetch');

const app = express();
const port = config.port || 5000;

// Middleware
app.use(cors({ origin: config.corsOrigin }));
app.use(express.json());
app.use(fileUpload());

// Connect to MongoDB
mongoose.connect(config.mongodb.uri)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// Create directories if they don't exist
const directories = [config.directories.uploads, config.directories.images];
directories.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
});

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

// Add this function to server.js
async function recreateQdrantCollection() {
  try {
    const url = config.qdrant.url || 'http://localhost:6333';
    const collectionName = config.qdrant.collectionName;
    
    console.log(`Deleting Qdrant collection: ${collectionName}`);
    
    // Delete the collection
    const deleteResponse = await fetch(`${url}/collections/${collectionName}`, {
      method: 'DELETE'
    });
    
    if (deleteResponse.ok) {
      console.log(`Successfully deleted Qdrant collection: ${collectionName}`);
    } else {
      console.log(`Collection may not exist or couldn't be deleted: ${await deleteResponse.text()}`);
    }
    
    // Create collection with correct vector size
    console.log(`Creating Qdrant collection: ${collectionName}`);
    const createResponse = await fetch(`${url}/collections/${collectionName}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        vectors: {
          size: 512,  // Correct size for CLIP base model
          distance: 'Cosine'
        }
      })
    });
    
    if (createResponse.ok) {
      console.log(`Successfully created Qdrant collection: ${collectionName}`);
    } else {
      console.error(`Failed to create Qdrant collection: ${await createResponse.text()}`);
    }
  } catch (error) {
    console.error(`Error recreating Qdrant collection: ${error.message}`);
  }
}


// Routes
app.use('/api', require('./routes/api'));

app.listen(port, async () => {
  console.log(`Server running on port ${port}`);
  await recreateQdrantCollection();  // Use this instead of ensureQdrantCollection
});