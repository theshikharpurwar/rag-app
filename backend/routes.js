const express = require('express');
const router = express.Router();
const Embedding = require('./model');

// Search route
router.get('/search', async (req, res) => {
  const { query } = req.query;
  const results = await Embedding.find({ vector: { $nearSphere: [query] } });
  res.send(results);
});

module.exports = router;