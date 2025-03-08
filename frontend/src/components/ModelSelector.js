// D:\rag-app\frontend\src\components\ModelSelector.js

import React, { useState } from 'react';
import './ModelSelector.css';

const ModelSelector = ({ selectedModel, setSelectedModel, apiKey, setApiKey }) => {
  const [customPath, setCustomPath] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);

  const predefinedModels = [
    { name: 'Mistral (Default)', value: 'mistral' },
    { name: 'Mistral Large', value: 'mistral-large-latest' },
    { name: 'Mistral Medium', value: 'mistral-medium-latest' },
    { name: 'Mistral Small', value: 'mistral-small-latest' },
    { name: 'CLIP (Fallback for embeddings)', value: 'clip' }
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

  const handleCustomPathChange = (e) => {
    setCustomPath(e.target.value);
  };

  const handleApiKeyChange = (e) => {
    setApiKey(e.target.value);
  };

  const handleCustomPathSubmit = () => {
    if (customPath.trim()) {
      setSelectedModel(customPath);
    }
  };

  return (
    <div className="model-selector">
      <h3>Model Settings</h3>
      
      <div className="api-key-input">
        <label htmlFor="api-key">Mistral API Key:</label>
        <input 
          type="password" 
          id="api-key" 
          value={apiKey} 
          onChange={handleApiKeyChange}
          placeholder="Enter your Mistral API key" 
        />
      </div>
      
      <div className="selector-container">
        <label htmlFor="model-select">Select Model:</label>
        <select 
          id="model-select"
          value={predefinedModels.some(m => m.value === selectedModel) ? selectedModel : 'custom'} 
          onChange={handleModelChange}
        >
          {predefinedModels.map(model => (
            <option key={model.value} value={model.value}>
              {model.name}
            </option>
          ))}
          <option value="custom">Custom Model</option>
        </select>
        
        {showCustomInput && (
          <div className="custom-input">
            <input 
              type="text" 
              value={customPath} 
              onChange={handleCustomPathChange} 
              placeholder="Enter model name or path"
            />
            <button onClick={handleCustomPathSubmit}>Set</button>
          </div>
        )}
      </div>
      <div className="current-model">
        Current: {selectedModel}
      </div>
    </div>
  );
};

export default ModelSelector;