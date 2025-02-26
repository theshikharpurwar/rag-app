import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ModelSelector.css';

function ModelSelector({ embeddingModel, llmModel, onModelChange }) {
  const [models, setModels] = useState({
    embedding: [],
    llm: []
  });
  const [showAddModel, setShowAddModel] = useState(false);
  const [newModel, setNewModel] = useState({
    type: 'embedding',
    name: '',
    path: '',
    parameters: '{}'
  });

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/model-config');
      
      setModels({
        embedding: response.data.filter(m => m.type === 'embedding'),
        llm: response.data.filter(m => m.type === 'llm')
      });
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  };

  const handleAddModel = async (e) => {
    e.preventDefault();
    
    try {
      // Parse parameters to validate JSON
      const parameters = JSON.parse(newModel.parameters);
      
      await axios.post('http://localhost:5000/api/model-config', {
        ...newModel,
        parameters
      });
      
      setNewModel({
        type: 'embedding',
        name: '',
        path: '',
        parameters: '{}'
      });
      
      setShowAddModel(false);
      fetchModels();
    } catch (err) {
      console.error('Error adding model:', err);
      alert(`Error adding model: ${err.message}`);
    }
  };

  return (
    <div className="model-selector">
      <h4>Model Configuration</h4>
      
      <div className="model-section">
        <h5>Embedding Model</h5>
        <div className="current-model">
          Current: <strong>{embeddingModel.name}</strong>
        </div>
        
        <select 
          onChange={(e) => onModelChange('embedding', e.target.value)}
          value={embeddingModel._id || ''}
        >
          <option value="" disabled>Change model</option>
          {models.embedding.map(model => (
            <option key={model._id} value={model._id}>
              {model.name} ({model.path})
            </option>
          ))}
        </select>
      </div>
      
      <div className="model-section">
        <h5>LLM Model</h5>
        <div className="current-model">
          Current: <strong>{llmModel.name}</strong>
        </div>
        
        <select
          onChange={(e) => onModelChange('llm', e.target.value)}
          value={llmModel._id || ''}
        >
          <option value="" disabled>Change model</option>
          {models.llm.map(model => (
            <option key={model._id} value={model._id}>
              {model.name} ({model.path})
            </option>
          ))}
        </select>
      </div>
      
      <button 
        className="add-model-btn"
        onClick={() => setShowAddModel(!showAddModel)}
      >
        {showAddModel ? 'Cancel' : 'Add New Model'}
      </button>
      
      {showAddModel && (
        <form className="add-model-form" onSubmit={handleAddModel}>
          <div className="form-group">
            <label>Model Type</label>
            <select
              value={newModel.type}
              onChange={(e) => setNewModel({...newModel, type: e.target.value})}
              required
            >
              <option value="embedding">Embedding</option>
              <option value="llm">LLM</option>
            </select>
          </div>
          
          <div className="form-group">
            <label>Name</label>
            <input
              type="text"
              value={newModel.name}
              onChange={(e) => setNewModel({...newModel, name: e.target.value})}
              required
              placeholder="e.g., colpali, qwen"
            />
          </div>
          
          <div className="form-group">
            <label>Model Path</label>
            <input
              type="text"
              value={newModel.path}
              onChange={(e) => setNewModel({...newModel, path: e.target.value})}
              required
              placeholder="e.g., vidore/colpali-v1.2"
            />
          </div>
          
          <div className="form-group">
            <label>Parameters (JSON)</label>
            <textarea
              value={newModel.parameters}
              onChange={(e) => setNewModel({...newModel, parameters: e.target.value})}
              placeholder='{"param1": "value1"}'
            />
          </div>
          
          <button type="submit" className="submit-btn">Add Model</button>
        </form>
      )}
    </div>
  );
}

export default ModelSelector;