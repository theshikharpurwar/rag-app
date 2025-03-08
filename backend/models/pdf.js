// D:\rag-app\backend\models\pdf.js

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const PDFSchema = new Schema({
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
    required: true
  },
  pageCount: {
    type: Number,
    default: 0
  },
  processed: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('PDF', PDFSchema);