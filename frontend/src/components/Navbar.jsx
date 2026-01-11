import { Link } from "react-router-dom";
export default function Navbar() {
  return (
    <>
      <div className="navbar bg-base-100 mx-7xl- mx-auto px-4 sm:px-6 lg:px-8">
        <Link to="/" className="text-xl text-primary">Voicefaux</Link>
      </div>
    </>
  );
}
