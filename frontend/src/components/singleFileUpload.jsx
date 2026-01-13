import React, { useState } from "react";
import { FileAudio, Play } from "lucide-react";

export default function SingleFileUpload({ singleFile, setSingleFile, classifyAudio, isProcessing, batchFiles }) {
  const [isProcessingSingle, setIsProcessingSingle] = useState(false);

  const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (file && file.type.startsWith("audio/")) {
    setSingleFile(file);
    setBatchFiles([]); // clear batch files when a single file is selected
  }
};
 const handleClassify = async () => {
  if (!singleFile) return;
  setIsProcessingSingle(true);
  await classifyAudio([singleFile]);
  setIsProcessingSingle(false);

  // ✅ Clear single file after classification
  setSingleFile(null);
};


  

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title text-primary">
          <FileAudio className="w-5 h-5" /> Single Audio Classification
        </h2>
        <input type="file" accept=".wav" onChange={handleFileChange} className="file-input file-input-bordered file-input-primary w-full" />
        {singleFile && (
          <div className="alert alert-info mt-4 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <FileAudio className="w-5 h-5" />
              <span className="text-sm">{singleFile.name}</span>
            </div>
            <audio controls className="w-full" src={URL.createObjectURL(singleFile)} />
          </div>
        )}
        <div className="card-actions justify-end mt-4">
          <button className="btn btn-primary" onClick={handleClassify} disabled={!singleFile || isProcessingSingle || batchFiles.length > 0}>

          
            {isProcessingSingle ? <span className="loading loading-spinner"></span> : <Play className="w-4 h-4" />}
            Classify
          </button>
        </div>
      </div>
    </div>
  );
}
