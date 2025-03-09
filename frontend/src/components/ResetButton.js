import React, { useState } from 'react';
import { resetSystem } from '../api';
import './ResetButton.css';

const ResetButton = ({ onReset }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleReset = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await resetSystem();
      if (response.success) {
        if (onReset) {
          await onReset(); // Call the callback to update the parent component
        }
      } else {
        setError(response.message || 'Reset failed');
      }
    } catch (err) {
      setError(err.message || 'Error during reset');
    } finally {
      setLoading(false);
      setShowConfirm(false);
    }
  };

  return (
    <div className="reset-button-container">
      {!showConfirm ? (
        <button 
          className="reset-button" 
          onClick={() => setShowConfirm(true)}
        >
          Reset System
        </button>
      ) : (
        <div className="confirm-dialog">
          <p>Are you sure you want to reset the system?</p>
          <p>This will delete all uploaded documents and embeddings.</p>
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