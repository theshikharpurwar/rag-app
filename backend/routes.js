const express = require('express');
const router = express.Router();

// Import the new route files
const uploadRoutes = require('./uploadRoutes');
const queryRoutes = require('./queryRoutes');

// Use the routes
router.use('/upload', uploadRoutes);
router.use('/query', queryRoutes);

module.exports = router;