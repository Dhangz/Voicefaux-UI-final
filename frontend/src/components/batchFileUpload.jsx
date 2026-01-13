import React, { useState } from "react";
import { Upload, Play } from "lucide-react";

export default function BatchFileUpload({ batchFiles, setBatchFiles, classifyAudio, isProcessing, singleFile }) {
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);

  const handleFileChange = (e) => {
  const files = Array.from(e.target.files).filter(f => f.type.startsWith("audio/"));
  setBatchFiles(files);
  setSingleFile(null); // clear single file when batch is selected
};

  const handleClassify = async () => {
    if (batchFiles.length === 0) return;
    setIsProcessingBatch(true);
    await classifyAudio(batchFiles);
    setIsProcessingBatch(false);


     setBatchFiles([]);
  };

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title text-secondary">
          <Upload className="w-5 h-5" /> Batch Classification
        </h2>
        <input type="file" accept=".wav" multiple onChange={handleFileChange} className="file-input file-input-bordered file-input-secondary w-full" />
        {batchFiles.length > 0 && (
          <div className="alert alert-success mt-4">
            <div className="flex items-center gap-2 mb-2">
              <Upload className="w-5 h-5" />
              <span className="text-sm">{batchFiles.length} files selected</span>
            </div>
            <div className="flex flex-col gap-2 max-h-64 overflow-y-auto">
              {batchFiles.map((file, idx) => (
                <audio key={idx} controls className="w-full h-2" src={URL.createObjectURL(file)} />
              ))}
            </div>
          </div>
        )}
        <div className="card-actions justify-end mt-4">
          <button className="btn btn-secondary" onClick={handleClassify} disabled={batchFiles.length === 0 || isProcessingBatch || singleFile}>
            {isProcessingBatch ? <span className="loading loading-spinner"></span> : <Play className="w-4 h-4" />}
            Classify Batch
          </button>
        </div>
      </div>
    </div>
  );
}
