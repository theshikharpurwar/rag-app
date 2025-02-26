// models/Document.js
const mongoose = require('mongoose');

const DocumentSchema = new mongoose.Schema({
  originalName: {
    type: String,
    required: true
  },
  storedName: {
    type: String,
    required: true,
    unique: true
  },
  pageCount: {
    type: Number,
    default: 0
  },
  status: {
    type: String,
    enum: ['uploaded', 'processing', 'indexed', 'failed'],
    default: 'uploaded'
  },
  indexedAt: {
    type: Date
  },
  uploadedAt: {
    type: Date,
    default: Date.now
  },
  error: {
    type: String
  }
});

module.exports = mongoose.model('Document', DocumentSchema);