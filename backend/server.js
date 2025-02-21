const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const routes = require('./routes');
require('dotenv').config();

const app = express();

// Middleware
app.use(express.json());
app.use(cors());
app.use('/api', routes);

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.error('MongoDB connection error:', err));

// Python script execution (using child_process)
const { spawn } = require('child_process');
const path = require('path');

app.post('/upload-pdf', async (req, res) => {
  const { pdfPath } = req.body; // Assume PDF path is sent from frontend

  if (!pdfPath) {
    return res.status(400).json({ error: 'PDF path is required' });
  }

  try {
    // Call Python script to compute embeddings
    const pythonProcess = spawn('python3', [
      path.join(__dirname, '../python/compute_embeddings.py'),
      pdfPath
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        const embeddings = JSON.parse(output);
        // Store embeddings in MongoDB
        const VectorStore = mongoose.model('VectorStore', vectorSchema); // Defined in model.js
        VectorStore.insertMany(embeddings)
          .then(() => res.json({ message: 'PDF processed and embeddings stored' }))
          .catch(err => res.status(500).json({ error: err.message }));
      } else {
        res.status(500).json({ error: 'Python script failed' });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/ask-question', async (req, res) => {
  const { question, pdfId } = req.body; // Assume PDF ID and question are sent

  if (!question || !pdfId) {
    return res.status(400).json({ error: 'Question and PDF ID are required' });
  }

  try {
    // Call Python script for LLM generation
    const pythonProcess = spawn('python3', [
      path.join(__dirname, '../python/local_llm.py'),
      question,
      pdfId
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        res.json({ answer: output });
      } else {
        res.status(500).json({ error: 'Python script failed' });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));