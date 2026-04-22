import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";

// Add CSS for spinner animation
const styleSheet = document.createElement("style");
styleSheet.textContent = `
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;
document.head.appendChild(styleSheet);

export default function InventoryPredictor() {
  const [trainingData, setTrainingData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [accuracy, setAccuracy] = useState(null);
  const [error, setError] = useState(null);
  const [model, setModel] = useState(null);
  const [normalizationParams, setNormalizationParams] = useState(null);

  // Handle training data CSV upload
  const handleTrainingCSVUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setError(null);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          // Validate CSV structure
          if (
            results.data.length === 0 ||
            !results.data[0].stock ||
            !results.data[0].avgSales ||
            !results.data[0].leadTime ||
            results.data[0].reorderStatus === undefined
          ) {
            setError(
              "CSV must contain: stock, avgSales, leadTime, reorderStatus"
            );
            return;
          }

          // Validate all data are numbers
          const validData = results.data.filter(row => 
            !isNaN(parseFloat(row.stock)) &&
            !isNaN(parseFloat(row.avgSales)) &&
            !isNaN(parseFloat(row.leadTime)) &&
            !isNaN(parseFloat(row.reorderStatus))
          );

          if (validData.length === 0) {
            setError("⚠️ No valid numeric data found in CSV. Check your values!");
            return;
          }

          if (validData.length !== results.data.length) {
            setError(`⚠️ Skipped ${results.data.length - validData.length} rows with invalid data. Using ${validData.length} valid rows.`);
          }

          console.log("Valid training samples:", validData.length);
          setTrainingData(validData);
          setPredictions([]);
          setAccuracy(null);
          setError(null);
        } catch (err) {
          setError("Error parsing CSV: " + err.message);
        }
      },
      error: (err) => {
        setError("Error reading file: " + err.message);
      },
    });
  };

  // Convert CSV data to TensorFlow tensors
  const convertToTensors = (data) => {
    const features = data.map((row) => [
      parseFloat(row.stock),
      parseFloat(row.avgSales),
      parseFloat(row.leadTime),
    ]);

    // Calculate min and max for normalization
    const mins = [
      Math.min(...features.map(f => f[0])),
      Math.min(...features.map(f => f[1])),
      Math.min(...features.map(f => f[2])),
    ];
    const maxs = [
      Math.max(...features.map(f => f[0])),
      Math.max(...features.map(f => f[1])),
      Math.max(...features.map(f => f[2])),
    ];

    // Normalize features to 0-1 range
    const normalizedFeatures = features.map((row) => [
      (row[0] - mins[0]) / (maxs[0] - mins[0] || 1),
      (row[1] - mins[1]) / (maxs[1] - mins[1] || 1),
      (row[2] - mins[2]) / (maxs[2] - mins[2] || 1),
    ]);

    const labels = data.map((row) => [parseFloat(row.reorderStatus)]);

    return {
      inputTensor: tf.tensor2d(normalizedFeatures),
      outputTensor: tf.tensor2d(labels),
      mins,
      maxs,
    };
  };

  // Train model
  const handleTrain = async () => {
    if (!trainingData || trainingData.length === 0) {
      setError("Please upload training data first");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const { inputTensor, outputTensor, mins, maxs } = convertToTensors(trainingData);
      setNormalizationParams({ mins, maxs });

      // Create model
      const newModel = tf.sequential({
        layers: [
          tf.layers.dense({
            inputShape: [3],
            units: 8,
            activation: "relu",
          }),
          tf.layers.dense({
            units: 1,
            activation: "sigmoid",
          }),
        ],
      });

      // Compile model
      newModel.compile({
        optimizer: "adam",
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
      });

      // Train model
      const history = await newModel.fit(inputTensor, outputTensor, {
        epochs: 200,
        shuffle: true,
        verbose: 0,
      });

      // Get final accuracy (safely handle the history object)
      console.log("Training history:", history);
      console.log("History data:", history.history);
      
      if (history.history && history.history.accuracy && history.history.accuracy.length > 0) {
        const finalAccuracy = history.history.accuracy[history.history.accuracy.length - 1];
        console.log("Final accuracy value:", finalAccuracy);
        setAccuracy((finalAccuracy * 100).toFixed(2));
      } else {
        console.warn("Accuracy data not found, trying alternative method");
        // Fallback: evaluate model on training data
        const evalResult = newModel.evaluate(inputTensor, outputTensor);
        const accuracyValue = await evalResult[1].data();
        const acc = accuracyValue[0];
        console.log("Fallback accuracy:", acc);
        setAccuracy((acc * 100).toFixed(2));
      }

      // Store model for predictions
      setModel(newModel);

      // Cleanup tensors
      inputTensor.dispose();
      outputTensor.dispose();

      setLoading(false);
    } catch (err) {
      console.error("Training error:", err);
      setError("❌ Error training model: " + err.message);
      setLoading(false);
    }
  };

  // Handle prediction data input
  const handlePredictionsInput = async (event) => {
    if (!model) {
      setError("Please train the model first");
      return;
    }

    const text = event.target.value;
    if (!text.trim()) {
      setPredictions([]);
      return;
    }

    try {
      setError(null);
      const lines = text
        .trim()
        .split("\n")
        .filter((line) => line.trim());
      const predictResults = [];

      for (const line of lines) {
        const parts = line
          .split(",")
          .map((part) => parseFloat(part.trim()))
          .filter((val) => !isNaN(val));

        if (parts.length !== 3) {
          setError(
            "Each line must have exactly 3 values: stock, avgSales, leadTime"
          );
          return;
        }

        const [stock, avgSales, leadTime] = parts;
        let normalizedInput = [stock, avgSales, leadTime];
        if (normalizationParams) {
          const { mins, maxs } = normalizationParams;
          normalizedInput = [
            (stock - mins[0]) / (maxs[0] - mins[0] || 1),
            (avgSales - mins[1]) / (maxs[1] - mins[1] || 1),
            (leadTime - mins[2]) / (maxs[2] - mins[2] || 1),
          ];
        }
        const inputTensor = tf.tensor2d([normalizedInput]);
        const result = model.predict(inputTensor);
        const value = (await result.data())[0];

        predictResults.push({
          stock,
          avgSales,
          leadTime,
          prediction: value > 0.5 ? "Reorder" : "No Reorder",
        });

        inputTensor.dispose();
        result.dispose();
      }

      setPredictions(predictResults);
    } catch (err) {
      setError("Error making predictions: " + err.message);
    }
  };

  return (
    <div style={styles.container}>
      <div style={{ marginBottom: "30px" }}>
        <h2 style={styles.title}>Inventory Reorder Predictor</h2>
        <p style={styles.subtitle}></p>
      </div>

      {error && <div style={styles.error}>⚠️ {error}</div>}

      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          <span style={styles.stepNumber}>1</span>
          Upload Training Data
        </div>
        <p style={styles.hint}>
          📋 CSV Format: stock, avgSales, leadTime, reorderStatus
        </p>
        <input
          type="file"
          accept=".csv"
          onChange={handleTrainingCSVUpload}
          style={styles.fileInput}
        />
        {trainingData && (
          <p style={styles.success}>
            ✅ Successfully loaded {trainingData.length} training samples
          </p>
        )}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          <span style={styles.stepNumber}>2</span>
          Train Machine Learning Model
        </div>
        <p style={styles.hint}>
          Train Model 
        </p>
        <button 
          onClick={handleTrain} 
          disabled={!trainingData || loading}
          style={{
            ...styles.button,
            ...((!trainingData || loading) ? styles.buttonDisabled : {})
          }}
          onMouseEnter={(e) => !(!trainingData || loading) && Object.assign(e.target.style, styles.buttonHover)}
          onMouseLeave={(e) => Object.assign(e.target.style, { backgroundColor: "#1976d2", boxShadow: "0 2px 4px rgba(25, 118, 210, 0.3)", transform: "translateY(0)" })}
        >
          {loading ? "🔄 Training..." : "Train"}
        </button>
        {loading && (
          <p style={styles.loading}>
            <span style={styles.spinner}></span>
            Training in progress... this typically takes 5-10 seconds
          </p>
        )}
        {accuracy && (
          <p style={styles.success}>
            ✨ Model trained successfully! Accuracy: <strong>{accuracy}%</strong>
          </p>
        )}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          <span style={styles.stepNumber}>3</span>
          Make Predictions
        </div>
        <p style={styles.hint}>
         Enter
        </p>
        <textarea
          onChange={handlePredictionsInput}
          placeholder="Examples:&#10;10, 45, 3&#10;20, 50, 4&#10;15, 35, 5"
          style={{
            ...styles.textarea,
            opacity: !model ? 0.6 : 1,
            cursor: !model ? "not-allowed" : "text",
          }}
          disabled={!model}
        />
        {!model && (
          <p style={styles.hint} style={{ marginTop: "12px", color: "#ff9800" }}>
            ⏳ Train the model first to unlock predictions
          </p>
        )}
      </div>

      {predictions.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>📈 Prediction Results</div>
          <div style={styles.resultsSummary}>
            Showing <strong>{predictions.length}</strong> prediction{predictions.length !== 1 ? 's' : ''}
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeader}>
                  <th style={{ ...styles.tableCell, color: "white", fontWeight: "700" }}>Stock</th>
                  <th style={{ ...styles.tableCell, color: "white", fontWeight: "700" }}>Avg Sales</th>
                  <th style={{ ...styles.tableCell, color: "white", fontWeight: "700" }}>Lead Time</th>
                  <th style={{ ...styles.tableCell, color: "white", fontWeight: "700" }}>Prediction</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((pred, idx) => (
                  <tr
                    key={idx}
                    style={{
                      ...styles.tableRow,
                      backgroundColor: idx % 2 === 0 ? "#ffffff" : "#f9f9f9",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "#f0f7ff";
                      e.currentTarget.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.05)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = idx % 2 === 0 ? "#ffffff" : "#f9f9f9";
                      e.currentTarget.style.boxShadow = "none";
                    }}
                  >
                    <td style={styles.tableCell}><strong>{pred.stock}</strong></td>
                    <td style={styles.tableCell}><strong>{pred.avgSales}</strong></td>
                    <td style={styles.tableCell}><strong>{pred.leadTime}</strong></td>
                    <td
                      style={{
                        ...styles.tableCell,
                        fontWeight: "bold",
                        color: pred.prediction === "Reorder" ? "#d32f2f" : "#388e3c",
                        fontSize: "15px",
                        backgroundColor: pred.prediction === "Reorder" ? "#ffebee" : "#e8f5e9",
                        borderRadius: "6px",
                      }}
                    >
                      {pred.prediction === "Reorder" ? "🔴" : "🟢"} {pred.prediction}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "40px 20px",
    maxWidth: "1000px",
    margin: "0 auto",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    backgroundColor: "#f0f4f8",
    minHeight: "100vh",
  },
  title: {
    fontSize: "32px",
    fontWeight: "700",
    color: "#1a237e",
    marginBottom: "10px",
    textAlign: "center",
  },
  subtitle: {
    fontSize: "16px",
    color: "#666",
    textAlign: "center",
    marginBottom: "40px",
  },
  section: {
    marginBottom: "25px",
    padding: "28px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#fff",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
    transition: "box-shadow 0.3s ease",
  },
  sectionTitle: {
    fontSize: "20px",
    fontWeight: "600",
    color: "#1a237e",
    marginBottom: "15px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  hint: {
    fontSize: "13px",
    color: "#777",
    fontStyle: "italic",
    margin: "10px 0 15px 0",
    padding: "10px 12px",
    backgroundColor: "#f5f5f5",
    borderLeft: "3px solid #1976d2",
    borderRadius: "4px",
  },
  fileInput: {
    padding: "12px",
    marginTop: "12px",
    cursor: "pointer",
    border: "2px solid #e0e0e0",
    borderRadius: "8px",
    fontSize: "14px",
    transition: "border-color 0.3s ease",
  },
  button: {
    padding: "12px 28px",
    backgroundColor: "#1976d2",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "15px",
    fontWeight: "600",
    marginTop: "12px",
    transition: "all 0.3s ease",
    boxShadow: "0 2px 4px rgba(25, 118, 210, 0.3)",
  },
  buttonHover: {
    backgroundColor: "#1565c0",
    boxShadow: "0 4px 8px rgba(25, 118, 210, 0.4)",
    transform: "translateY(-2px)",
  },
  buttonDisabled: {
    backgroundColor: "#ccc",
    cursor: "not-allowed",
    boxShadow: "none",
  },
  textarea: {
    width: "100%",
    height: "140px",
    padding: "14px",
    marginTop: "12px",
    borderRadius: "8px",
    border: "2px solid #e0e0e0",
    fontFamily: "'Courier New', monospace",
    fontSize: "14px",
    resize: "vertical",
    transition: "border-color 0.3s ease",
    boxSizing: "border-box",
  },
  loading: {
    color: "#ff9800",
    fontWeight: "600",
    marginTop: "15px",
    fontSize: "15px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  spinner: {
    display: "inline-block",
    width: "16px",
    height: "16px",
    border: "2px solid #ff9800",
    borderTop: "2px solid transparent",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  success: {
    color: "#388e3c",
    fontWeight: "600",
    marginTop: "15px",
    fontSize: "15px",
    padding: "12px 14px",
    backgroundColor: "#e8f5e9",
    borderRadius: "6px",
    border: "1px solid #81c784",
  },
  error: {
    padding: "16px 18px",
    backgroundColor: "#ffebee",
    color: "#c62828",
    borderRadius: "8px",
    marginBottom: "20px",
    border: "2px solid #ef5350",
    fontSize: "15px",
    fontWeight: "500",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: "20px",
    overflow: "hidden",
    borderRadius: "8px",
    border: "1px solid #e0e0e0",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.08)",
  },
  tableHeader: {
    backgroundColor: "#1a237e",
    color: "white",
    fontWeight: "700",
    fontSize: "14px",
  },
  tableCell: {
    padding: "16px 18px",
    textAlign: "center",
    borderBottom: "1px solid #e8e8e8",
    fontSize: "14px",
  },
  tableRow: {
    backgroundColor: "#fff",
    transition: "all 0.3s ease",
    borderRadius: "4px",
  },
  tableRowHover: {
    backgroundColor: "#f0f7ff",
    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.05)",
  },
  stepNumber: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: "28px",
    height: "28px",
    backgroundColor: "#1976d2",
    color: "white",
    borderRadius: "50%",
    fontSize: "14px",
    fontWeight: "700",
  },
  resultsSummary: {
    padding: "12px 16px",
    backgroundColor: "#e3f2fd",
    borderLeft: "4px solid #1976d2",
    borderRadius: "4px",
    marginBottom: "15px",
    fontSize: "14px",
    color: "#1a237e",
  },
};
