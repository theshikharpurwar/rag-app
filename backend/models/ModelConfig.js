// models/ModelConfig.js
const mongoose = require('mongoose');

const ModelConfigSchema = new mongoose.Schema({
  type: {
    type: String,
    enum: ['embedding', 'llm'],
    required: true
  },
  name: {
    type: String,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  parameters: {
    type: Object,
    default: {}
  },
  isActive: {
    type: Boolean,
    default: false
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('ModelConfig', ModelConfigSchema);