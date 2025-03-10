// D:\rag-app\frontend\src\components\ModelSelector.js

import React, { useState, useEffect } from 'react';
import './ModelSelector.css';

const ModelSelector = ({ selectedModel, setSelectedModel }) => {
  const [localModels, setLocalModels] = useState([
    { name: 'Phi-2', path: 'phi-2', description: 'Microsoft Phi-2 - small but capable local LLM' },
    { name: 'Llama3', path: 'llama3', description: 'Llama 3 - Meta\'s advanced open source LLM' },
    { name: 'Mistral', path: 'mistral', description: 'Mistral - efficient open-weight language model' }
  ]);
  const [customModel, setCustomModel] = useState('');
  
  useEffect(() => {
    if (!selectedModel && localModels.length > 0) {
      setSelectedModel(localModels[0].path);
    }
  }, [selectedModel, setSelectedModel, localModels]);

  const handleModelSelect = (modelPath) => {
    setSelectedModel(modelPath);
  };

  const handleCustomModelChange = (e) => {
    setCustomModel(e.target.value);
  };

  const handleCustomModelSubmit = (e) => {
    e.preventDefault();
    if (customModel.trim()) {
      setSelectedModel(customModel.trim());
    }
  };

  return (
    <div className="model-selector">
      <h3>Select Model</h3>
      <div className="model-options">
        {localModels.map((model) => (
          <div 
            key={model.path}
            className={`model-option ${selectedModel === model.path ? 'selected' : ''}`}
            onClick={() => handleModelSelect(model.path)}
          >
            <div className="model-info">
              <h4>{model.name}</h4>
              <p>{model.description}</p>
            </div>
          </div>
        ))}
      </div>
      
      <form className="custom-model-form" onSubmit={handleCustomModelSubmit}>
        <input
          type="text"
          placeholder="Enter custom model name"
          value={customModel}
          onChange={handleCustomModelChange}
        />
        <button type="submit">Use Custom Model</button>
      </form>
      
      <div className="selected-model">
        <p>Currently using: <strong>{selectedModel}</strong></p>
      </div>
    </div>
  );
};

export default ModelSelector;