// D:\rag-app\frontend\src\components\ResetButton.js

import React, { useState } from 'react';
import { resetApplication } from '../api';
import './ResetButton.css';

const ResetButton = ({ onReset }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleReset = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await resetApplication();
      console.log('Reset successful:', result);
      
      // Call the callback if provided
      if (onReset) {
        onReset();
      }
      
      setShowConfirm(false);
    } catch (err) {
      setError(err.message || 'Failed to reset application');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reset-button-container">
      {!showConfirm ? (
        <button 
          className="reset-button danger" 
          onClick={() => setShowConfirm(true)}
          disabled={loading}
        >
          Reset Application
        </button>
      ) : (
        <div className="confirm-dialog">
          <p>Are you sure? This will delete all uploaded documents and clear the database.</p>
          <div className="confirm-actions">
            <button 
              className="confirm-button danger" 
              onClick={handleReset}
              disabled={loading}
            >
              {loading ? 'Resetting...' : 'Yes, Reset Everything'}
            </button>
            <button 
              className="cancel-button" 
              onClick={() => setShowConfirm(false)}
              disabled={loading}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ResetButton;