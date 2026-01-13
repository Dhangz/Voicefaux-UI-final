import React from "react";

export default function ClassificationResults({ classifications }) {
  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title text-accent">Classification Results</h2>
        <div className="space-y-4 mt-4">
          {classifications.map((result, idx) => (
            <div key={idx} className="collapse collapse-arrow bg-base-200">
              <input type="radio" name="results-accordion" defaultChecked={idx === 0} />
              <div className="collapse-title text-lg font-medium">
                {result.filename}
                <span className="badge badge-primary ml-2">{result.predictions[0].category}</span>
              </div>
              <div className="collapse-content">
                <div className="space-y-2">
                  {result.predictions.slice(0, 5).map((pred, pidx) => (
                    <div key={pidx}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">{pred.category}</span>
                        <span className="text-sm">{(pred.confidence * 100).toFixed(2)}%</span>
                      </div>
                      <progress className="progress progress-primary" value={pred.confidence * 100} max="100"></progress>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
