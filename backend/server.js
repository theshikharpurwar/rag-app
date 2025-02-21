const express = require('express');
const cors = require('cors');
const routes = require('./routes');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const app = express();

// Middleware
app.use(express.json());
app.use(cors());
app.use('/api', routes);

// JSON file path for vector store
const vectorStorePath = path.join(__dirname, 'vector_store.json');

// Helper function to read/write vector store
const readVectorStore = () => {
  if (fs.existsSync(vectorStorePath)) {
    const data = fs.readFileSync(vectorStorePath, 'utf8');
    return JSON.parse(data) || [];
  }
  return [];
};

const writeVectorStore = (data) => {
  fs.writeFileSync(vectorStorePath, JSON.stringify(data, null, 2));
};

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
        // Read existing vectors, append new ones, and write back
        const existingVectors = readVectorStore();
        const updatedVectors = [...existingVectors, ...embeddings];
        writeVectorStore(updatedVectors);
        res.json({ message: 'PDF processed and embeddings stored' });
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