import Navbar from "./Navbar";

export default function ({ children }) {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <Navbar />

      <main className="max-w-7xl mx-auto sm:px-6 lg:px-8">{children}</main>
    </div>
  )
}
