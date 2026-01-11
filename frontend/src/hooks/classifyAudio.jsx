import { useState } from "react";

export default function useClassifyAudio() {
  const API_URL = "http://127.0.0.1:8000/api/classify/";

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(
    () => JSON.parse(localStorage.getItem("audioClassificationResult")) || null
  );

  const fetchClassification = async (file) => {
    if (!file) {
      setResult(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("audio_file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to classify audio.");

      const data = await response.json();
      setResult(data);
      localStorage.setItem("audioClassificationResult", JSON.stringify(data));
    } catch (err) {
      console.error(err);
      setError("Error classifying audio.");
    } finally {
      setIsLoading(false);
    }
  };

  const resetResult = () => {
    setResult(null);
    localStorage.removeItem("audioClassificationResult");
  };

  return { isLoading, error, result, fetchClassification, resetResult };
}
