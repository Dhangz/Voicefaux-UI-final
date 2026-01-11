import "./App.css";
import { Routes, Route } from "react-router-dom";

import Home from "./components/pages/Home";
import Classify from "./components/pages/Classify";
import About from "./components/pages/About";

import Layout from "./components/Layout";
function App() {

  return (
    <>
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/classify" element={<Classify />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Layout>
    </>
  );
}

export default App;
