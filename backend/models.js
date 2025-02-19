// backend/models.js
const mongoose = require('mongoose');

// Schema for a chunk of text extracted from a document.
// Each chunk stores its original text and a computed "embedding" (a numerical representation).
const ChunkSchema = new mongoose.Schema({
  documentId: { type: mongoose.Schema.Types.ObjectId, ref: 'Document' },
  text: String,
  embedding: [Number], // An array of numbers representing the embedding vector.
});

// Schema for a document that is uploaded.
// This includes metadata like the original filename and the full text content.
const DocumentSchema = new mongoose.Schema({
  filename: String,
  originalName: String,
  text: String,
  createdAt: { type: Date, default: Date.now },
});

// Create Mongoose models for each schema.
const Document = mongoose.model('Document', DocumentSchema);
const Chunk = mongoose.model('Chunk', ChunkSchema);

// Export the models so other parts of our backend can use them.
module.exports = { Document, Chunk };
