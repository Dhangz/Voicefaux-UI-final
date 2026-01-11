export default function ResultDisplay({ result }) {
  if (!result) {
    return (
      <div className="md:col-span-2 bg-base-100 rounded-lg p-4 border border-primary text-center">
        <p className="text-gray-500">No classification result yet.</p>
      </div>
    );
  }

  return (
    <div className="md:col-span-2 bg-base-100 rounded-lg p-4 border border-primary">
      <h2 className="text-lg font-bold mb-2 text-center">
        Classification Result
      </h2>

      <p className="text-center text-primary text-xl font-semibold">
        {result.data.predicted_class}
      </p>

      <p className="text-center text-gray-500 mt-1">
        Confidence: {(result.data.confidence * 100).toFixed(2)}%
      </p>
    </div>
  );
}
