import { useState, useEffect } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

import AudioPlayer from "../AudioPlayer";
import ResultDisplay from "../ResultDisplay";

import useClassifyAudio from "../../hooks/classifyAudio";

export default function Classify() {
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  const { isLoading, error, result, fetchClassification, resetResult } = useClassifyAudio();

  const COLORS = ["#3b82f6", "#22c55e", "#f97316", "#FF0087"];

  // Prepare chart data
  const chartData = result
    ? Object.entries(result.data.all_probabilities).map(([name, value]) => ({
        name,
        value: value * 100,
      }))
    : [];

  // Save audio to localStorage
  useEffect(() => {
    if (audioFile) {
      const reader = new FileReader();
      reader.onload = () => {
        localStorage.setItem("savedAudio", reader.result);
        localStorage.setItem("savedAudioName", audioFile.name);
      };
      reader.readAsDataURL(audioFile);
    }
  }, [audioFile]);

  // Restore from localStorage
  useEffect(() => {
    const savedAudio = localStorage.getItem("savedAudio");
    const savedName = localStorage.getItem("savedAudioName");

    if (savedAudio && savedName) {
      fetch(savedAudio)
        .then((res) => res.blob())
        .then((blob) => {
          const restoredFile = new File([blob], savedName, { type: blob.type });
          setAudioFile(restoredFile);
          setAudioUrl(URL.createObjectURL(blob));
        });
    }
  }, []);

  // Reset result when new audio loaded
  useEffect(() => {
    if (audioFile) resetResult();
  }, [audioFile]);

  // Handle file selection
const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Revoke previous audio URL if exists
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }

  setAudioFile(file);
  setAudioUrl(URL.createObjectURL(file)); // create new URL immediately
  resetResult();
};

// Clear everything
const clearAll = () => {
  // Revoke audio URL to free memory
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }

  setAudioFile(null);
  setAudioUrl(null);
  resetResult();
  localStorage.removeItem("savedAudio");
  localStorage.removeItem("savedAudioName");
  localStorage.removeItem("audioClassificationResult");
};
  return (
    <div className="flex flex-col lg:flex-row w-full mt-4 gap-4">
      {/* Left panel */}
      <div className="card rounded-box flex flex-col items-center justify-center p-4 lg:w-1/4 gap-2">
        <label className="btn btn-primary w-full cursor-pointer">
          Choose Audio File
          <input type="file" accept=".wav,audio/wav" onChange={handleFileChange} className="hidden" />
        </label>

        {audioFile && (
          <p className="text-sm text-gray-500 mt-2 text-center">
            Loaded audio: <span className="font-medium">{audioFile.name}</span>
          </p>
        )}

        <button className="btn btn-secondary w-full cursor-pointer mt-2" onClick={clearAll}>
          Clear Audio
        </button>
      </div>

      {/* Right panel */}
      <div className="card bg-base-300 rounded-box flex-1 p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Audio Player */}
          <AudioPlayer
            audioUrl={audioUrl}
            audioFile={audioFile}
            onClassify={fetchClassification}
            isLoading={isLoading}
          />

          {/* PieChart */}
          <div className="bg-base-200 rounded-lg p-4 flex flex-col gap-4">
            <p className="font-semibold text-center mb-3">Chart Result</p>
            <div className="flex-1 min-h-62.5">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius="70%"
                      label
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={index} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend verticalAlign="bottom" />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-gray-500 mt-4">No data to display</p>
              )}
            </div>
          </div>

          {/* Result Display */}
          <ResultDisplay result={result} />
        </div>
      </div>
    </div>
  );
}
