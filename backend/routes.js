const express = require('express');
const router = express.Router();
const { uploadPdf, queryQuestion, logAction } = require('./server'); // Import the functions

router.post('/upload/pdf', uploadPdf);
router.post('/query/question', queryQuestion);
router.post('/log', logAction);

module.exports = router;