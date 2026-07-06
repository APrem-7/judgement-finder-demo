import { Routes, Route, NavLink } from 'react-router-dom'
import { Scale, Upload, Search, FlaskConical, LayoutDashboard } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import IngestionPipeline from './pages/IngestionPipeline'
import JudgementFinder from './pages/JudgementFinder'
import ModelComparison from './pages/ModelComparison'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/pipeline', icon: Upload, label: 'Ingestion Pipeline' },
  { to: '/finder', icon: Search, label: 'Judgement Finder' },
  { to: '/models', icon: FlaskConical, label: 'Model Comparison' },
]

export default function App() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
        <div className="px-5 py-6 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Scale className="text-saffron-400" size={22} />
            <span className="font-bold text-lg tracking-tight text-white">KanoonSaathi</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Legal Intelligence Platform</p>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-navy-800 text-white'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
                }`
              }
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-slate-800">
          <p className="text-xs text-slate-600">Demo v1.0 · SC India 1950–2024</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/pipeline" element={<IngestionPipeline />} />
          <Route path="/finder" element={<JudgementFinder />} />
          <Route path="/models" element={<ModelComparison />} />
        </Routes>
      </main>
    </div>
  )
}
