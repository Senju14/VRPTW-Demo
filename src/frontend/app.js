const { useState, useEffect } = React;

const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="5 3 19 12 5 21 5 3"></polygon>
  </svg>
);

const MaximizeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
  </svg>
);

const MinimizeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
  </svg>
);

const GridIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="7" height="7"></rect>
    <rect x="14" y="3" width="7" height="7"></rect>
    <rect x="14" y="14" width="7" height="7"></rect>
    <rect x="3" y="14" width="7" height="7"></rect>
  </svg>
);

function App() {
  const [instances, setInstances] = useState([]);
  const [selectedInstances, setSelectedInstances] = useState(new Set());
  const [algorithms, setAlgorithms] = useState({
    alns: true,
    dqn: true,
    dqn_alns: true,
    ortools: true
  });
  const [maxVehicles, setMaxVehicles] = useState('15');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    fetch('/api/instances')
      .then(res => res.json())
      .then(data => setInstances(data));
  }, []);

  const toggleInstance = (inst) => {
    const newSet = new Set(selectedInstances);
    if (newSet.has(inst)) newSet.delete(inst);
    else newSet.add(inst);
    setSelectedInstances(newSet);
  };

  const handleRun = async () => {
    if (selectedInstances.size === 0) return;
    
    setLoading(true);
    const selectedAlgos = Object.keys(algorithms).filter(k => algorithms[k]);
    
    try {
      const response = await fetch('/api/run_comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instances: Array.from(selectedInstances),
          algorithms: selectedAlgos,
          max_vehicles: parseInt(maxVehicles) || 15
        })
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const toggleExpand = (key) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="min-h-screen bg-white text-black p-6">
      <header className="mb-8 border-b-2 border-black pb-4">
        <h1 className="text-3xl font-bold">VRPTW Solver Comparison</h1>
      </header>

      <div className="mb-6 space-y-4">
        <div className="border-2 border-black p-4">
          <h2 className="font-bold mb-3">Select Instances</h2>
          <div className="grid grid-cols-8 gap-2">
            {instances.map(inst => (
              <label key={inst} className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedInstances.has(inst)}
                  onChange={() => toggleInstance(inst)}
                  className="w-4 h-4"
                />
                <span className="text-sm">{inst.toUpperCase()}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="border-2 border-black p-4">
            <h2 className="font-bold mb-3">Select Algorithms</h2>
            <div className="space-y-2">
              {Object.keys(algorithms).map(algo => (
                <label key={algo} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={algorithms[algo]}
                    onChange={(e) => setAlgorithms({...algorithms, [algo]: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span>{algo === 'ortools' ? 'OR-Tools' : algo.toUpperCase().replace('_', '+')}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="border-2 border-black p-4">
            <h2 className="font-bold mb-3">Configuration</h2>
            <label className="block mb-2">
              <span className="text-sm font-medium">Max Vehicles</span>
              <input
                type="text"
                value={maxVehicles}
                onChange={(e) => setMaxVehicles(e.target.value.replace(/\D/g, ''))}
                className="w-full border-2 border-black p-2 mt-1"
                placeholder="15"
              />
            </label>
            <button
              onClick={handleRun}
              disabled={loading || selectedInstances.size === 0}
              className="w-full bg-black text-white p-2 flex items-center justify-center space-x-2 hover:bg-gray-800 disabled:bg-gray-400"
            >
              <PlayIcon />
              <span>{loading ? 'Running...' : 'Run Comparison'}</span>
            </button>
          </div>
        </div>
      </div>

      {results && (
        <>
          <div className="mb-4 flex justify-end">
            <button
              onClick={() => setShowTable(!showTable)}
              className="border-2 border-black px-4 py-2 flex items-center space-x-2 hover:bg-gray-100"
            >
              <GridIcon />
              <span>{showTable ? 'Hide Table' : 'Show Table'}</span>
            </button>
          </div>

          {showTable && (
            <div className="mb-6 border-2 border-black p-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-black">
                    <th className="text-left p-2">Instance</th>
                    <th className="text-left p-2">Algorithm</th>
                    <th className="text-right p-2">Vehicles</th>
                    <th className="text-right p-2">Distance</th>
                    <th className="text-right p-2">Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.table.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-300">
                      <td className="p-2">{row.instance}</td>
                      <td className="p-2">{row.algorithm}</td>
                      <td className="p-2 text-right">{row.vehicles}</td>
                      <td className="p-2 text-right">{row.distance.toFixed(2)}</td>
                      <td className="p-2 text-right">{row.time.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            {results.solutions.map((sol, idx) => {
              const key = `${sol.instance}_${sol.algorithm}`;
              const isExpanded = expanded[key];
              
              return (
                <div key={idx} className="border-2 border-black relative">
                  <div className="flex items-center justify-between p-2 border-b-2 border-black bg-gray-50">
                    <h3 className="font-bold text-sm">
                      {sol.instance.toUpperCase()} - {sol.algorithm}
                    </h3>
                    <button
                      onClick={() => toggleExpand(key)}
                      className="hover:bg-gray-200 p-1"
                    >
                      {isExpanded ? <MinimizeIcon /> : <MaximizeIcon />}
                    </button>
                  </div>

                  <svg 
                    viewBox="0 0 600 400" 
                    className={`w-full transition-all ${isExpanded ? 'h-96' : 'h-64'}`}
                  >
                    <rect x="0" y="0" width="600" height="400" fill="#f5f5f5" />
                    
                    <defs>
                      <pattern id={`grid-${idx}`} width="20" height="20" patternUnits="userSpaceOnUse">
                        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#ddd" strokeWidth="0.5"/>
                      </pattern>
                    </defs>
                    <rect width="600" height="400" fill={`url(#grid-${idx})`} />
                    
                    {sol.routes.map((route, ridx) => {
                      const hue = (ridx * 360) / sol.routes.length;
                      const color = `hsl(${hue}, 65%, 45%)`;
                      const points = [sol.depot, ...route.nodes, sol.depot];
                      
                      return (
                        <g key={ridx}>
                          <polyline
                            points={points.map(p => `${p.x},${p.y}`).join(' ')}
                            fill="none"
                            stroke={color}
                            strokeWidth="2.5"
                            strokeLinecap="round"
                          />
                          {route.nodes.map((node, nidx) => (
                            <circle
                              key={nidx}
                              cx={node.x}
                              cy={node.y}
                              r="5"
                              fill="white"
                              stroke={color}
                              strokeWidth="2"
                            />
                          ))}
                        </g>
                      );
                    })}
                    
                    <rect
                      x={sol.depot.x - 8}
                      y={sol.depot.y - 8}
                      width="16"
                      height="16"
                      fill="#000"
                      stroke="white"
                      strokeWidth="2"
                    />
                  </svg>

                  <div className="p-2 border-t-2 border-black bg-gray-50 text-xs grid grid-cols-3 gap-2">
                    <div>
                      <span className="font-medium">V:</span> {sol.vehicles}
                    </div>
                    <div>
                      <span className="font-medium">D:</span> {sol.distance.toFixed(1)}
                    </div>
                    <div>
                      <span className="font-medium">T:</span> {sol.time.toFixed(2)}s
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
