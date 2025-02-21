const mongoose = require('mongoose');

const vectorSchema = new mongoose.Schema({
  id: { type: String, required: true, unique: true }, // Unique identifier (e.g., page ID)
  embedding: { type: [Number], required: true }, // Array of numbers for embeddings (e.g., 128 dimensions)
  metadata: {
    imagePath: String, // Path to the image (if applicable)
    text: String, // Extracted text from PDF page
    pageNumber: Number // Page number in PDF
  },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('VectorStore', vectorSchema);