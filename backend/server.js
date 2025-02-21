const express = require('express');
const cors = require('cors');
const routes = require('./routes');
const fs = require('fs');
const path = require('path');
const fileUpload = require('express-fileupload');
const { spawn } = require('child_process');

const app = express();

// Middleware
app.use(express.json());
app.use(cors());
app.use(fileUpload()); // Middleware for handling file uploads
app.use('/api', routes);

// JSON file path for vector store
const vectorStorePath = path.join(__dirname, 'vector_store.json');
const uploadsDir = path.join(__dirname, 'uploads');

// Ensure uploads directory exists
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

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

// Log actions to terminal
app.post('/api/log', (req, res) => {
  const { action } = req.body;
  if (action) {
    console.log(`Action: ${action}`);
    res.json({ message: 'Logged to terminal' });
  } else {
    res.status(400).json({ error: 'Action is required' });
  }
});

app.post('/api/upload/pdf', (req, res) => {
  const { pdf } = req.files;
  if (!pdf) {
    return res.status(400).json({ error: 'PDF file is required' });
  }

  const pdfPath = path.join(uploadsDir, pdf.name);
  pdf.mv(pdfPath, async (err) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to save PDF' });
    }

    try {
      // Log upload attempt
      await axios.post('http://localhost:5000/api/log', {
        action: `Uploading PDF: ${pdf.name}`,
      });

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
          const existingVectors = readVectorStore();
          const updatedVectors = [...existingVectors, ...embeddings];
          writeVectorStore(updatedVectors);
          fs.unlinkSync(pdfPath); // Clean up temporary PDF
          res.json({ message: 'PDF uploaded and processed successfully' });
        } else {
          res.status(500).json({ error: 'Python script failed' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
});

app.post('/api/query/question', (req, res) => {
  const { question, pdfId } = req.body;

  if (!question || !pdfId) {
    return res.status(400).json({ error: 'Question and PDF ID are required' });
  }

  try {
    // Log question attempt
    axios.post('http://localhost:5000/api/log', {
      action: `Asking question: ${question} for PDF: ${pdfId}`,
    });

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

// Catch-all for 404 errors
app.use((req, res) => {
  console.log(`404 - Not Found: ${req.method} ${req.url}`);
  res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = 5000; // Explicitly set port to avoid environment variable issues
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));