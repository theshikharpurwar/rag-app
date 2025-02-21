const express = require('express');
const router = express.Router();

router.use('/upload', require('./uploadRoutes')); // Optional: Separate route for uploads
router.use('/query', require('./queryRoutes'));   // Optional: Separate route for queries

module.exports = router;