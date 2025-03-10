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
          className="btn btn-danger reset-button" 
          onClick={handleClick}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
            <path fill="currentColor" d="M19.1,4.9C15.2,1 8.8,1 4.9,4.9C1,8.8 1,15.2 4.9,19.1C8.8,23 15.2,23 19.1,19.1C23,15.2 23,8.8 19.1,4.9M15.5,8.5L13.4,10.6L15.5,12.7L12.7,15.5L10.6,13.4L8.5,15.5L5.7,12.7L7.8,10.6L5.7,8.5L8.5,5.7L10.6,7.8L12.7,5.7L15.5,8.5Z" />
          </svg>
          Reset System
        </button>
      ) : (
        <div className="confirm-dialog card">
          <div className="confirm-header">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
              <path fill="currentColor" d="M13,13H11V7H13M13,17H11V15H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
            </svg>
            <h4>Confirm Reset</h4>
          </div>
          <p>Are you sure? This will delete all documents and clear the database.</p>
          <div className="confirm-actions">
            <button 
              className="btn btn-success" 
              onClick={handleReset}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Resetting...
                </>
              ) : (
                'Yes, Reset'
              )}
            </button>
            <button 
              className="btn btn-danger" 
              onClick={() => setShowConfirm(false)}
              disabled={loading}
            >
              Cancel
            </button>
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
      )}
    </div>
  );
};

export default ResetButton;