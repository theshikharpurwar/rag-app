// D:\rag-app\backend\models\pdf.js

const mongoose = require('mongoose');

const pdfSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true
  },
  originalName: {
    type: String,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  size: {
    type: Number,
    required: true
  },
  mimeType: {
    type: String,
    default: 'application/pdf'  // Add a default value
  },
  pageCount: {
    type: Number,
    required: true
  },
  embeddingsCount: {
    type: Number,
    required: true
  }
}, {
  timestamps: true
});

const PDF = mongoose.model('PDF', pdfSchema);

module.exports = PDF;