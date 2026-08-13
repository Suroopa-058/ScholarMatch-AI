import { Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import About from "./pages/About";
import Home from "./pages/Home";
import Profile from "./pages/Profile";
import Recommendations from "./pages/Recommendations";
import ScholarshipDetails from "./pages/ScholarshipDetails";
import Scholarships from "./pages/Scholarships";
import Auth from "./pages/Auth";

function App() {
  return <div className="min-h-screen bg-[var(--color-paper)]"><Routes><Route path="/login" element={<Auth mode="login" />} /><Route path="/signup" element={<Auth mode="signup" />} /><Route path="*" element={<><Header /><main><Routes><Route path="/" element={<Home />} /><Route path="/profile" element={<Profile />} /><Route path="/recommendations" element={<Recommendations />} /><Route path="/scholarships" element={<Scholarships />} /><Route path="/scholarships/:scholarshipId" element={<ScholarshipDetails />} /><Route path="/about" element={<About />} /></Routes></main><Footer /></>} /></Routes></div>;
}

export default App;
