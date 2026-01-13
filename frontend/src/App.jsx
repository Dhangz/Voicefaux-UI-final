import "./App.css";
import { Routes, Route } from "react-router-dom";

import AudioClassifier from "./components/pages/AudioClassifier";

function App() {

  return (
    <>
      <Routes>
        <Route path="/" element={<AudioClassifier />} />
        
      </Routes>
    </>
  );
}

export default App;
