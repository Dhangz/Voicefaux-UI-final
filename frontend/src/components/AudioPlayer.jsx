import { useState, useRef, useEffect } from "react";

export default function AudioPlayer({ audioUrl, audioFile, onClassify, isLoading }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) setProgress(audioRef.current.currentTime);
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) setDuration(audioRef.current.duration);
  };

  const handleSeek = (value) => {
    if (audioRef.current) {
      audioRef.current.currentTime = value;
      setProgress(value);
    }
  };

  const handleClassify = async () => {
    if (audioFile && onClassify) await onClassify(audioFile);
  };

  useEffect(() => {
    setProgress(0);
    setIsPlaying(false);
  }, [audioUrl]);

  return (
    <div className="bg-base-200 rounded-lg p-4 flex flex-col justify-between min-h-60">
      {audioFile ? (
        <>
          <p className="font-semibold text-center mb-3">Audio Preview</p>

          <audio
            ref={audioRef}
            src={audioUrl}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />

          <div className="flex items-center gap-4 mb-4">
            <button className="btn btn-primary btn-circle" onClick={togglePlay}>
              {isPlaying ? (
                <div className="flex gap-1">
                  <span className="w-1.5 h-4 bg-white rounded"></span>
                  <span className="w-1.5 h-4 bg-white rounded"></span>
                </div>
              ) : (
                <div className="ml-1 w-0 h-0 border-t-8 border-b-8 border-l-12 border-t-transparent border-b-transparent border-l-white"></div>
              )}
            </button>

            <input
              type="range"
              min="0"
              max={duration || 0}
              value={progress}
              onChange={(e) => handleSeek(Number(e.target.value))}
              className="range range-primary flex-1"
            />
          </div>

          <button
            className="btn btn-primary w-full"
            onClick={handleClassify}
            disabled={isLoading}
          >
            {isLoading ? "Classifying..." : "Classify Audio"}
          </button>
        </>
      ) : (
        <p className="text-gray-500 text-center">
          Select an audio file to preview
        </p>
      )}
    </div>
  );
}
