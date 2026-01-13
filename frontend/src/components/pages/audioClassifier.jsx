import { useState } from "react";

import SingleFileUpload from "../singleFileUpload";
import BatchFileUpload from "../batchFileUpload";
import ClassificationResults from "../ClassificationResults";
import PieChartDistribution from "../PieChartDistribution";
import { Trash2 } from "lucide-react";

export default function AudioClassifier() {
  const [singleFile, setSingleFile] = useState(null);
  const [batchFiles, setBatchFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // =========================
  // API CALL
  // =========================
  const classifyAudio = async (files) => {
    setIsProcessing(true);
    const formData = new FormData();
    if (files.length === 1) formData.append("audio_file", files[0]);
    else files.forEach((f) => formData.append("audio_files", f));

    try {
      const endpoint =
        files.length === 1
          ? "http://127.0.0.1:8000/api/classify/"
          : "http://127.0.0.1:8000/api/classify-batch/";

      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();

      if (data.success) {
        if (files.length === 1) {
          const result = data.data;
          setResults({
            classifications: [
              {
                filename: files[0].name,
                predictions: Object.entries(result.all_probabilities)
                  .map(([category, confidence]) => ({ category, confidence }))
                  .sort((a, b) => b.confidence - a.confidence),
              },
            ],
            distribution: [{ name: result.predicted_class, value: 1 }],
          });
        } else {
          const classifications = data.data.results.map((res) => ({
            filename: res.filename,
            predictions: Object.entries(res.all_probabilities)
              .map(([category, confidence]) => ({ category, confidence }))
              .sort((a, b) => b.confidence - a.confidence),
          }));

          const distributionMap = {};
          classifications.forEach((c) => {
            const topCat = c.predictions[0].category;
            distributionMap[topCat] = (distributionMap[topCat] || 0) + 1;
          });
          setResults({
            classifications,
            distribution: Object.entries(distributionMap).map(([name, value]) => ({ name, value })),
          });
        }
      } else {
        alert("Classification failed: " + JSON.stringify(data.errors || data.error));
      }
    } catch (err) {
      console.error("Error calling API:", err);
      alert("Failed to call backend API.");
    } finally {
      setIsProcessing(false);
    }
  };

  const clearAll = () => {
  setSingleFile(null);
  setBatchFiles([]);
  setResults(null);

  // Reset all processing flags
  setIsProcessing(false);
  setIsProcessingSingle(false);
  setIsProcessingBatch(false);
};

  return (
    <div className="min-h-screen bg-base-200 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-primary mb-2">Audio Classification System</h1>
          <p className="text-base-content/70">Multi-class audio analysis with ML</p>
        </div>

        {/* Upload Sections */}
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          <SingleFileUpload
            singleFile={singleFile}
            setSingleFile={setSingleFile}
            classifyAudio={classifyAudio}
            isProcessing={isProcessing}
            batchFiles={batchFiles}
          />
          <BatchFileUpload
            batchFiles={batchFiles}
            setBatchFiles={setBatchFiles}
            classifyAudio={classifyAudio}
            isProcessing={isProcessing}
            singleFile={singleFile}
          />
        </div>

        {/* Clear Button */}
        {(singleFile || batchFiles.length > 0 || results) && (
          <div className="flex justify-center mb-6">
            <button className="btn btn-error btn-outline" onClick={clearAll}>
              <Trash2 className="w-4 h-4" /> Clear All
            </button>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="space-y-6">
            <PieChartDistribution distribution={results.distribution} />
            <ClassificationResults classifications={results.classifications} />
          </div>
        )}
      </div>
    </div>
  );
}
