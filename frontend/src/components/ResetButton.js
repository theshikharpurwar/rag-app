// D:\rag-app\frontend\src\components\ResetButton.js

import React, { useState } from 'react';
import { resetSystem } from '../api';
import './ResetButton.css';

const ResetButton = ({ onReset }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClick = () => {
    setShowConfirm(true);
  };

  const handleReset = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await resetSystem();
      
      if (response.success) {
        setShowConfirm(false);
        if (onReset) onReset();
      } else {
        setError(response.message || 'Failed to reset system');
      }
    } catch (err) {
      setError('Error resetting system');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reset-button-container">
      <button 
        className="reset-button" 
        onClick={handleClick}
        disabled={loading || showConfirm}
      >
        Reset System
      </button>
      
      {showConfirm && (
        <div className="confirm-dialog">
          <p>Are you sure you want to reset the system? This will delete all uploaded PDFs and clear the vector database.</p>
          <button 
            className="confirm-button" 
            onClick={handleReset}
            disabled={loading}
          >
            {loading ? 'Resetting...' : 'Yes, Reset'}
          </button>
          <button 
            className="cancel-button" 
            onClick={() => setShowConfirm(false)}
            disabled={loading}
          >
            Cancel
          </button>
          {error && <p className="error">{error}</p>}
        </div>
      )}
    </div>
  );
};

export default ResetButton;