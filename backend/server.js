// backend/server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const multer = require('multer');
const { Document, Chunk } = require('./models');
const { computeEmbeddings, generateResponse } = require('./colpali_utils');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/api/upload', multer({ dest: 'uploads/' }).single('file'), async (req, res) => {
  try {
    const { file } = req.file;
    if (!file) return res.status(400).json({ error: "No file uploaded" });

    const text = await extractTextFromFile(file.path);
    const document = new Document({
      filename: file.filename,
      originalName: file.originalname,
      text,
    });
    await document.save();

    fs.unlinkSync(file.path);

    const rawChunks = text.split('\n\n');
    for (let chunk of rawChunks) {
      const cleanedChunk = chunk.trim().replace(/\s+/g, ' ');
      if (cleanedChunk.length < 10) {
        console.log("Skipping chunk (too short):", cleanedChunk);
        continue;
      }
      try {
        const embedding = await computeEmbeddings(cleanedChunk);
        const chunkDoc = new Chunk({
          documentId: document._id,
          text: cleanedChunk,
          embedding,
        });
        await chunkDoc.save();
      } catch (err) {
        console.error("Error processing chunk:", err.message);
        continue;
      }
    }

    res.json({ message: "Document uploaded and processed", documentId: document._id });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing document" });
  }
});

app.post('/api/query', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: "No query provided" });

    const queryEmbedding = await computeEmbeddings(query);
    const topChunks = await findTopChunks(queryEmbedding);

    const prompt = createPromptFromChunks(topChunks);
    const answer = await generateResponse(prompt);

    res.json({ answer, prompt });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing query" });
  }
});

mongoose.connect('mongodb://localhost:27017/rag-app')
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("MongoDB connection error:", err));

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});