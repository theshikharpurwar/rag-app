// backend/models.js
const mongoose = require('mongoose');

const DocumentSchema = new mongoose.Schema({
  filename: String,
  originalName: String,
  text: String,
  createdAt: { type: Date, default: Date.now },
});

const ChunkSchema = new mongoose.Schema({
  documentId: { type: mongoose.Schema.Types.ObjectId, ref: 'Document' },
  text: String,
  embedding: [Number],
});

const DocumentModel = mongoose.model('Document', DocumentSchema);
const Chunk = mongoose.model('Chunk', ChunkSchema);

module.exports = { DocumentModel, Chunk };