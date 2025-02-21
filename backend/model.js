const mongoose = require('mongoose');

const embeddingSchema = new mongoose.Schema({
  vector: { type: [Number], index: '2dsphere' },
  metadata: Object,
});

const Embedding = mongoose.model('Embedding', embeddingSchema);

module.exports = Embedding;