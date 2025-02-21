const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const fileUpload = require('express-fileupload');
const { QdrantClient } = require('qdrant-client');
const { spawn } = require('child_process');

const app = express();

// Middleware
app.use(express.json());
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
app.use(fileUpload());

// Qdrant client setup
const qdrantClient = new QdrantClient({ url: 'http://localhost:6333' });
const collectionName = 'multimodal_rag';

const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Ensure Qdrant collection exists with multivector config
const ensureCollection = async () => {
  try {
    const exists = await qdrantClient.getCollection(collectionName);
    if (!exists) {
      await qdrantClient.createCollection(collectionName, {
        vectors: {
          size: 128, // Matches ColPali's per-patch dimension
          distance: 'Cosine',
          on_disk: true,
        },
        multivector_config: {
          comparator: 'MaxSim', // For ColPali multivector retrieval
        },
      });
      console.log(`Qdrant collection ${collectionName} created`);
    }
  } catch (error) {
    console.error('Qdrant collection setup error:', error);
  }
};

ensureCollection();

// Routes
const uploadPdf = async (req, res) => {
  const { pdf } = req.files;
  if (!pdf) {
    return res.status(400).json({ error: 'PDF file is required' });
  }

  const pdfName = pdf.name;
  const pdfPath = path.join(uploadsDir, pdfName);
  pdf.mv(pdfPath, async (err) => {
    if (err) {
      console.error('File move error:', err);
      return res.status(500).json({ error: 'Failed to save PDF' });
    }

    try {
      console.log(`Uploading PDF: ${pdfName}`);
      const pythonProcess = spawn('python3', [
        path.join(__dirname, '../python/compute_embeddings.py'),
        pdfPath,
        collectionName
      ]);

      let output = '';
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
        console.log(`Python stdout: ${data.toString()}`);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data.toString()}`);
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          fs.unlinkSync(pdfPath); // Clean up temporary PDF
          res.json({ message: 'PDF uploaded and processed successfully', pdfId: pdfName });
        } else {
          res.status(500).json({ error: 'Embedding computation failed' });
        }
      });
    } catch (error) {
      console.error('Processing error:', error);
      res.status(500).json({ error: error.message });
    }
  });
};

const queryQuestion = async (req, res) => {
  const { question, pdfId } = req.body;

  if (!question || !pdfId) {
    return res.status(400).json({ error: 'Question and PDF ID are required' });
  }

  try {
    console.log(`Asking question: ${question} for PDF: ${pdfId}`);
    const pythonProcess = spawn('python3', [
      path.join(__dirname, '../python/local_llm.py'),
      question,
      pdfId,
      collectionName
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
      console.log(`Python stdout: ${data.toString()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        res.json({ answer: output.trim() });
      } else {
        res.status(500).json({ error: 'Query processing failed' });
      }
    });
  } catch (error) {
    console.error('Query error:', error);
    res.status(500).json({ error: error.message });
  }
};

app.post('/api/upload/pdf', uploadPdf);
app.post('/api/query/question', queryQuestion);

app.use((req, res) => {
  console.log(`404 - Not Found: ${req.method} ${req.url}`);
  res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));