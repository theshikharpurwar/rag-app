// D:\rag-app\backend\models\pdf.js

const mongoose = require('mongoose');

const pdfSchema = new mongoose.Schema({
  originalName: {
    type: String,
    required: true
  },
  filename: {
    type: String,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  mimeType: {
    type: String,
    default: 'application/pdf'  // Default value added
  },
  size: {
    type: Number,
    required: true
  },
  pageCount: {
    type: Number,
    default: 0
  },
  processed: {
    type: Boolean,
    default: false
  },
  uploadDate: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('PDF', pdfSchema);