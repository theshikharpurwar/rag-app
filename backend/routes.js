// backend/routes.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { Document, Chunk } = require('./models');
const { spawn } = require('child_process');

// Set up multer for file uploads.
// Files will be temporarily stored in the "backend/uploads" folder.
const upload = multer({ dest: path.join(__dirname, 'uploads/') });

// Utility function to extract text from a file.
// For PDFs, we use the pdf-parse library; for text files, we simply read them.
const extractTextFromFile = async (filepath, mimetype) => {
  if (mimetype === 'application/pdf') {
    const pdf = require('pdf-parse');
    const dataBuffer = fs.readFileSync(filepath);
    const data = await pdf(dataBuffer);
    return data.text;
  } else {
    return fs.readFileSync(filepath, 'utf8');
  }
};

// Upload endpoint: Handles document uploads.
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: "No file uploaded" });

    // Extract text from the uploaded file.
    const text = await extractTextFromFile(file.path, file.mimetype);

    // Save document details in MongoDB.
    const document = new Document({
      filename: file.filename,
      originalName: file.originalname,
      text,
    });
    await document.save();

    // Optionally, remove the file from the uploads folder after processing.
    fs.unlinkSync(file.path);

    // Split the text into chunks. Here, we use paragraphs (separated by two newlines) as chunks.
    const chunks = text.split('\n\n').filter(chunk => chunk.trim().length > 0);
    
    // For each chunk, compute its embedding by calling a Python script.
    for (let chunk of chunks) {
      // Call the Python script "compute_embeddings.py" and pass the chunk text via stdin.
      const pyProcess = spawn('python', [path.join(__dirname, '../python/compute_embeddings.py')]);
      
      // Write the chunk text to the Python process.
      pyProcess.stdin.write(chunk);
      pyProcess.stdin.end();

      let result = '';
      // Gather the output from the Python process.
      for await (const data of pyProcess.stdout) {
        result += data;
      }
      // Wait for the process to finish.
      await new Promise(resolve => pyProcess.on('close', resolve));

      // The Python script returns a JSON array representing the embedding.
      const embedding = JSON.parse(result);

      // Save this chunk along with its embedding to MongoDB.
      const chunkDoc = new Chunk({
        documentId: document._id,
        text: chunk,
        embedding,
      });
      await chunkDoc.save();
    }

    res.json({ message: "Document uploaded and processed", documentId: document._id });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing document" });
  }
});

// Query endpoint: Processes queries and returns generated answers.
router.post('/query', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: "No query provided" });

    // Compute the embedding for the query by calling the Python script.
    const pyProcess = spawn('python', [path.join(__dirname, '../python/compute_embeddings.py')]);
    pyProcess.stdin.write(query);
    pyProcess.stdin.end();

    let queryResult = '';
    for await (const data of pyProcess.stdout) {
      queryResult += data;
    }
    await new Promise(resolve => pyProcess.on('close', resolve));
    const queryEmbedding = JSON.parse(queryResult);

    // Retrieve all stored chunks from MongoDB.
    const chunks = await Chunk.find({});

    // Define a function to calculate cosine similarity between two vectors.
    const cosineSim = (vecA, vecB) => {
      const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
      const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
      const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
      return dot / (magA * magB);
    };

    // Calculate similarity scores for each chunk.
    const scoredChunks = chunks.map(chunk => ({
      text: chunk.text,
      score: cosineSim(queryEmbedding, chunk.embedding),
    }));

    // Sort the chunks by similarity and take the top 3.
    const topChunks = scoredChunks.sort((a, b) => b.score - a.score).slice(0, 3);

    // Build a prompt that includes the retrieved context and the query.
    const prompt = `
Your context:
${topChunks.map((c, i) => `Chunk ${i+1}: ${c.text}`).join('\n\n')}

Based on the above context, answer the query: ${query}
`;

    // Call the local language model (Python script "local_llm.py") with the prompt.
    const llmProcess = spawn('python', [path.join(__dirname, '../python/local_llm.py')]);
    llmProcess.stdin.write(prompt);
    llmProcess.stdin.end();

    let answer = '';
    for await (const data of llmProcess.stdout) {
      answer += data;
    }
    await new Promise(resolve => llmProcess.on('close', resolve));

    // Return the generated answer and the prompt (for debugging purposes).
    res.json({ answer, prompt });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing query" });
  }
});

module.exports = router;