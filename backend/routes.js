// backend/routes.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const pdfjsLib = require('pdfjs-dist');
const { Document, Chunk } = require('./models');

const router = express.Router();

const extractTextFromPDF = async (filepath) => {
  const data = new Uint8Array(fs.readFileSync(filepath));
  const pdfDoc = await pdfjsLib.getDocument({ data }).promise;
  let text = '';

  for (let i = 1; i <= pdfDoc.numPages; i++) {
    const page = await pdfDoc.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map(item => item.str);
    text += strings.join(' ') + '\n\n';
  }
  return text;
};

const extractTextFromFile = async (filepath, mimetype) => {
  if (mimetype === 'application/pdf') {
    return await extractTextFromPDF(filepath);
  } else {
    return fs.readFileSync(filepath, 'utf8');
  }
};

const runPythonScriptJson = async (scriptPath, inputText) => {
  return new Promise((resolve, reject) => {
    const pyProcess = spawn('python', [scriptPath]);
    pyProcess.stdin.write(inputText);
    pyProcess.stdin.end();

    let result = '';
    pyProcess.stdout.on('data', (data) => {
      result += data;
    });

    pyProcess.on('close', () => {
      if (!result) {
        return reject(new Error('No output received from Python script.'));
      }
      try {
        const parsed = JSON.parse(result);
        if (parsed && parsed.error) {
          return reject(new Error(parsed.error));
        }
        resolve(parsed);
      } catch (error) {
        console.error("Error parsing JSON output:", result);
        reject(new Error('Invalid JSON output from Python script.'));
      }
    });

    pyProcess.on('error', (err) => {
      reject(err);
    });
  });
};

const runPythonScriptText = async (scriptPath, inputText) => {
  return new Promise((resolve, reject) => {
    const pyProcess = spawn('python', [scriptPath]);
    pyProcess.stdin.write(inputText);
    pyProcess.stdin.end();

    let result = '';
    pyProcess.stdout.on('data', (data) => {
      result += data;
    });

    pyProcess.on('close', () => {
      if (!result) {
        return reject(new Error('No output received from Python script.'));
      }
      resolve(result);
    });

    pyProcess.on('error', (err) => {
      reject(err);
    });
  });
};

const uploadEndpoint = async (req, res) => {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: "No file uploaded" });

    const text = await extractTextFromFile(file.path, file.mimetype);
    console.log("Extracted text length:", text.length);

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
        const embedding = await runPythonScriptJson(
          path.join(__dirname, '../python/compute_embeddings.py'),
          cleanedChunk
        );
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
};

router.post('/upload', multer({ dest: path.join(__dirname, 'uploads/') }).single('file'), uploadEndpoint);

const queryEndpoint = async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: "No query provided" });

    const queryEmbedding = await runPythonScriptJson(
      path.join(__dirname, '../python/compute_embeddings.py'),
      query
    );

    const chunks = await Chunk.find({});

    const cosineSim = (vecA, vecB) => {
      const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
      const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
      const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
      return dot / (magA * magB);
    };

    const scoredChunks = chunks.map(chunk => ({
      text: chunk.text,
      score: cosineSim(queryEmbedding, chunk.embedding),
    }));

    const topChunks = scoredChunks.sort((a, b) => b.score - a.score).slice(0, 3);

    const prompt = `
Your context:
${topChunks.map((c, i) => `Chunk ${i + 1}: ${c.text}`).join('\n\n')}

Based on the above context, answer the query: ${query}
`;

    const answer = await runPythonScriptText(
      path.join(__dirname, '../python/local_llm.py'),
      prompt
    );

    res.json({ answer, prompt });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error processing query" });
  }
};

router.post('/query', queryEndpoint);

module.exports = router;