import React, { useState, useCallback } from 'react';
import './App.css';
import { FaCloudUploadAlt, FaSearch, FaTimesCircle, FaLeaf, FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = useCallback((event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
    setPredictionResult(null);
    setError(null);

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedImage) {
      setError('Please select an image.');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Prediction failed: ${response.status} - ${errorData?.detail || 'Something went wrong on the server.'}`);
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [selectedImage]);

  const handleClearImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    setPredictionResult(null);
    setError(null);
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <FaLeaf className="logo-icon" />
        <h1>Tomato Disease Detector</h1>
      </header>

      <main className="main-content">
        <section className="upload-section">
          <label htmlFor="image-upload" className="upload-label">
            <FaCloudUploadAlt className="upload-icon" />
            {selectedImage ? 'Change Image' : 'Upload Tomato Leaf Image'}
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="file-input"
          />
          {imagePreview && (
            <div className="image-preview-container">
              <h2>Image Preview:</h2>
              <img src={imagePreview} alt="Selected Tomato Leaf" className="image-preview" />
              <button className="clear-image-button" onClick={handleClearImage} aria-label="Clear image">
                <FaTimesCircle />
              </button>
            </div>
          )}
          <button
            className={`predict-button ${loading ? 'loading' : ''}`}
            onClick={handleSubmit}
            disabled={!selectedImage || loading}
          >
            {loading ? <FaSearch className="loading-icon spinning" /> : <FaSearch />} Predict Disease
          </button>
          {error && (
            <div className="error-container">
              <FaExclamationTriangle className="error-icon" />
              <p className="error-message">{error}</p>
            </div>
          )}
        </section>

        {predictionResult && (
          <section className={`prediction-section ${predictionResult ? 'show' : ''}`}>
            <h2><FaLeaf className="result-icon" /> Prediction Result <FaCheckCircle className="success-icon" /></h2>
            <div className="result-item">
              <strong>Predicted Class:</strong>
              <p className="predicted-class">{predictionResult.predicted_class}</p>
            </div>
            {predictionResult.confidence && (
              <div className="result-item">
                <strong>Confidence:</strong>
                <p>{(predictionResult.confidence * 100).toFixed(2)}%</p>
              </div>
            )}
            {predictionResult.title && (
              <div className="result-item">
                <strong>Title:</strong>
                <p>{predictionResult.title}</p>
              </div>
            )}
            {predictionResult.description && (
              <div className="result-item description">
                <strong>Description:</strong>
                <p>{predictionResult.description}</p>
              </div>
            )}
            {predictionResult.prevent && (
              <div className="result-item prevent">
                <strong>Possible Steps:</strong>
                <p>{predictionResult.prevent}</p>
              </div>
            )}
            {predictionResult.image_url && (
              <div className="result-item disease-image">
                <h3>Disease Image:</h3>
                <img src={predictionResult.image_url} alt={predictionResult.title} className="disease-image-display" />
              </div>
            )}
            {predictionResult.supplement_name && (
              <div className="result-item supplement-info">
                <h3>Recommended Supplement:</h3>
                <p><strong>Name:</strong> {predictionResult.supplement_name}</p>
                {predictionResult.supplement_image && (
                  <img
                    src={predictionResult.supplement_image}
                    alt={predictionResult.supplement_name}
                    className="supplement-image-display"
                  />
                )}
                {predictionResult.supplement_buy_link && (
                  <p>
                    <a
                      href={predictionResult.supplement_buy_link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="buy-link"
                    >
                      Buy on external site
                    </a>
                  </p>
                )}
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; 2025 Tomato Disease Detection</p>
      </footer>
    </div>
  );
}

export default App;