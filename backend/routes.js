const express = require('express');
const router = express.Router();

router.post('/upload/pdf', require('./server').uploadPdf); // Direct reference for simplicity
router.post('/query/question', require('./server').queryQuestion);
router.post('/log', require('./server').logAction);

module.exports = router;