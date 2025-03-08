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
    setError(null);
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
        setError(`Reset failed: ${response.message}`);
      }
    } catch (err) {
      console.error('Error in handleReset:', err);
      setError('Error resetting system');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reset-button-container">
      {!showConfirm ? (
        <button 
          className="reset-button" 
          onClick={handleClick}
        >
          Reset System
        </button>
      ) : (
        <div className="confirm-dialog">
          <p>Are you sure? This will delete all PDFs and clear the database.</p>
          <div className="confirm-actions">
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
          </div>
          {error && <p className="error">{error}</p>}
        </div>
      )}
    </div>
  );
};

export default ResetButton;