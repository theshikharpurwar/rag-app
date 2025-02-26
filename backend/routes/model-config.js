// routes/model-config.js
const express = require('express');
const ModelConfig = require('../models/ModelConfig');

const router = express.Router();

// Get all model configurations
router.get('/', async (req, res) => {
  try {
    const configs = await ModelConfig.find().sort({ type: 1, name: 1 });
    res.json(configs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add a new model configuration
router.post('/', async (req, res) => {
  try {
    const { type, name, path, parameters } = req.body;
    
    if (!type || !name || !path) {
      return res.status(400).json({ error: 'Type, name, and path are required' });
    }
    
    const modelConfig = new ModelConfig({
      type,
      name,
      path,
      parameters: parameters || {}
    });
    
    await modelConfig.save();
    res.status(201).json(modelConfig);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update a model configuration
router.put('/:id', async (req, res) => {
  try {
    const { type, name, path, parameters, isActive } = req.body;
    const modelConfig = await ModelConfig.findById(req.params.id);
    
    if (!modelConfig) {
      return res.status(404).json({ error: 'Model configuration not found' });
    }
    
    // If setting as active, deactivate all other models of the same type
    if (isActive && !modelConfig.isActive) {
      await ModelConfig.updateMany(
        { type: modelConfig.type, _id: { $ne: modelConfig._id } },
        { isActive: false }
      );
    }
    
    modelConfig.type = type || modelConfig.type;
    modelConfig.name = name || modelConfig.name;
    modelConfig.path = path || modelConfig.path;
    modelConfig.parameters = parameters || modelConfig.parameters;
    modelConfig.isActive = isActive !== undefined ? isActive : modelConfig.isActive;
    
    await modelConfig.save();
    res.json(modelConfig);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Delete a model configuration
router.delete('/:id', async (req, res) => {
  try {
    const modelConfig = await ModelConfig.findById(req.params.id);
    
    if (!modelConfig) {
      return res.status(404).json({ error: 'Model configuration not found' });
    }
    
    await modelConfig.remove();
    res.json({ message: 'Model configuration deleted' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;