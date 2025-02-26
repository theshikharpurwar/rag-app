// server.js
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const mongoose = require('mongoose');
const fileUpload = require('express-fileupload');
const { QdrantClient } = require('@qdrant/js-client-rest');
const { spawn } = require('child_process');
const config = require('./config/default.json');

// Import routes
const apiRoutes = require('./routes/api');
const modelConfigRoutes = require('./routes/model-config');

const app = express();

// Middleware
app.use(express.json());
app.use(cors({
  origin: config.corsOrigin,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type']
}));
app.use(fileUpload());

// Directory setup
const uploadsDir = path.join(__dirname, config.directories.uploads);
const imagesDir = path.join(__dirname, config.directories.images);

// Create directories if they don't exist
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
if (!fs.existsSync(imagesDir)) fs.mkdirSync(imagesDir, { recursive: true });

// MongoDB connection
mongoose.connect(config.mongodb.uri, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.error('MongoDB connection error:', err));

// Qdrant setup
const qdrantClient = new QdrantClient({ url: config.qdrant.url });
const ensureCollection = async () => {
  try {
    const collections = await qdrantClient.getCollections();
    const exists = collections.collections.some(c => c.name === config.qdrant.collectionName);
    
    if (!exists) {
      await qdrantClient.createCollection(config.qdrant.collectionName, {
        vectors: { 
          size: config.qdrant.vectorSize, 
          distance: config.qdrant.distance
        }
      });
      console.log(`Qdrant collection ${config.qdrant.collectionName} created`);
    }
  } catch (error) {
    console.error('Qdrant collection setup error:', error);
  }
};

ensureCollection();

// Routes
app.use('/api', apiRoutes);
app.use('/api/model-config', modelConfigRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});

const PORT = process.env.PORT || config.port;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));