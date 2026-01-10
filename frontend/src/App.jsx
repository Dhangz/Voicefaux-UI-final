import "./App.css";

function App() {
  return (
    <>
      <div className="max-w-7xl mx-auto p-4 sm:px-6 lg:px-8 h-screen">
        {/* Navbar */}
        <div className="navbar bg-base-100 text-primary-content">
          <button className="btn btn-primary text-xl">daisyUI</button>
        </div>

        {/* Content cards */}
        <div className="flex w-full mt-4 gap-4">
          {/* Left card with button */}
          <div className="card rounded-box grid h-20 flex-1 place-items-center">
            <button className="btn btn-soft btn-primary btn-wide">
              Choose Audio File
            </button>
          </div>

          {/* Divider */}
          <div className="divider divider-primary divider-horizontal"></div>

          {/* Right card */}
          <div className="card bg-base-300 rounded-box flex-2 p-4 h-100">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 h-full">
              {/* Column 1 */}
              <div className="bg-base-200 rounded-lg p-4 flex items-center justify-center">
                Column 1 Content
              </div>

              {/* Column 2 with dropdown */}
             <div className="bg-base-200 rounded-lg p-4 flex flex-col items-start gap-4 border">
  <div className="dropdown dropdown-center w-full">
    <label
      tabIndex={0}
      className="btn btn-primary w-full m-1"
    >
      Select Option
    </label>
    <ul
      tabIndex={0}
      className="dropdown-content menu p-2 shadow bg-base-100 rounded-box w-full sm:w-52"
    >
      <li>
        <a>Option 1</a>
      </li>
      <li>
        <a>Option 2</a>
      </li>
      <li>
        <a>Option 3</a>
      </li>
    </ul>
  </div>

  {/* Optional content below dropdown */}
  <div className="mt-4 flex-1 flex items-center justify-center">
    Additional Content
  </div>
</div>

            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
