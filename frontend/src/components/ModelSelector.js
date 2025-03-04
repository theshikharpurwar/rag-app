// D:\rag-app\frontend\src\components\ModelSelector.js

import React, { useState } from 'react';
import './ModelSelector.css';

const ModelSelector = ({ selectedModel, setSelectedModel }) => {
  const [customModelPath, setCustomModelPath] = useState('');
  const [showCustomPath, setShowCustomPath] = useState(false);

  // Predefined models
  const predefinedModels = [
    { name: 'clip', path: 'openai/clip-vit-base-patch32', label: 'CLIP (Default)' },
    { name: 'clip', path: 'openai/clip-vit-large-patch14', label: 'CLIP Large' }
  ];

  // Handle model selection
  const handleModelChange = (e) => {
    const value = e.target.value;
    
    if (value === 'custom') {
      setShowCustomPath(true);
      // Don't update the selected model yet, wait for custom path
    } else {
      setShowCustomPath(false);
      
      // Find the selected predefined model
      const model = predefinedModels.find(m => `${m.name}:${m.path}` === value);
      
      if (model && setSelectedModel) {
        setSelectedModel({
          name: model.name,
          path: model.path
        });
      }
    }
  };

  // Handle custom model path change
  const handleCustomPathChange = (e) => {
    setCustomModelPath(e.target.value);
  };

  // Handle custom model submission
  const handleCustomModelSubmit = (e) => {
    e.preventDefault();
    
    if (customModelPath.trim() && setSelectedModel) {
      setSelectedModel({
        name: 'custom',
        path: customModelPath.trim()
      });
    }
  };

  // Get the current model value for the select input
  const getCurrentModelValue = () => {
    if (selectedModel) {
      const predefinedModel = predefinedModels.find(
        m => m.name === selectedModel.name && m.path === selectedModel.path
      );
      
      if (predefinedModel) {
        return `${predefinedModel.name}:${predefinedModel.path}`;
      } else if (selectedModel.name === 'custom') {
        return 'custom';
      }
    }
    
    // Default to first model
    return `${predefinedModels[0].name}:${predefinedModels[0].path}`;
  };

  return (
    <div className="model-selector">
      <label htmlFor="model-select">Select Embedding Model:</label>
      <select
        id="model-select"
        value={getCurrentModelValue()}
        onChange={handleModelChange}
        className="model-select"
      >
        {predefinedModels.map((model, index) => (
          <option key={index} value={`${model.name}:${model.path}`}>
            {model.label}
          </option>
        ))}
        <option value="custom">Custom Model</option>
      </select>
      
      {showCustomPath && (
        <form onSubmit={handleCustomModelSubmit} className="custom-model-form">
          <input
            type="text"
            value={customModelPath}
            onChange={handleCustomPathChange}
            placeholder="Enter model path or identifier"
            className="custom-model-input"
          />
          <button 
            type="submit" 
            disabled={!customModelPath.trim()}
            className="custom-model-button"
          >
            Set Custom Model
          </button>
        </form>
      )}
      
      {selectedModel && (
        <div className="selected-model-info">
          <p><strong>Model:</strong> {selectedModel.name}</p>
          <p><strong>Path:</strong> {selectedModel.path}</p>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;