// D:\rag-app\frontend\src\components\ModelSelector.js

import React, { useState } from 'react';
import './ModelSelector.css';

const ModelSelector = ({ selectedModel, setSelectedModel }) => {
  const [customModel, setCustomModel] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);

  const predefinedModels = [
    { id: 'phi', name: 'Phi-2 (Default)' },
    { id: 'llama2', name: 'Llama 2' },
    { id: 'mistral', name: 'Mistral' },
    { id: 'gemma', name: 'Gemma' }
  ];

  const handleModelChange = (e) => {
    const value = e.target.value;
    if (value === 'custom') {
      setShowCustomInput(true);
    } else {
      setShowCustomInput(false);
      setSelectedModel(value);
    }
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
      <select value={selectedModel} onChange={handleModelChange}>
        {predefinedModels.map(model => (
          <option key={model.id} value={model.id}>{model.name}</option>
        ))}
        <option value="custom">Custom Model</option>
      </select>
      
      {showCustomInput && (
        <form onSubmit={handleCustomModelSubmit} className="custom-model-form">
          <input
            type="text"
            value={customModel}
            onChange={handleCustomModelChange}
            placeholder="Enter model name"
          />
          <button type="submit">Set</button>
        </form>
      )}
      
      <p className="current-model">Current model: {selectedModel}</p>
    </div>
  );
};

export default ModelSelector;