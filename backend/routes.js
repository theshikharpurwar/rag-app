const express = require('express');
const router = express.Router();

const uploadRoutes = require('./uploadRoutes');
const queryRoutes = require('./queryRoutes');

router.use('/upload', uploadRoutes);
router.use('/query', queryRoutes);

module.exports = router;