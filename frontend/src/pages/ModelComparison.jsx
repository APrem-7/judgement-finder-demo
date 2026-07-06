import { useState, useEffect } from 'react'
import { FlaskConical, Loader2, AlertCircle, ChevronDown, ChevronUp, Trophy } from 'lucide-react'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts'
import { api } from '../api/client'

const SCORE_COLORS = {
  'Kimi K2 (Moonshot AI)': '#ff9933',
  'Llama 3.3 70B (Meta)': '#3b5bdb',
  'Gemma 2 9B (Google)': '#16a34a',
  'DeepSeek R1 Distill 70B': '#7c3aed',
  'Qwen QwQ 32B (Alibaba)': '#0891b2',
  'Mixtral 8x7B (Mistral AI)': '#dc2626',
}

function ScoreDimension({ label, value }) {
  if (value == null) return null
  const pct = Math.round(value * 10)
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{label}</span>
        <span className="text-white">{value.toFixed(1)}/10</span>
      </div>
      <div className="score-bar-track">
        <div className={`score-bar-fill ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ModelCard({ result, rank }) {
  const [expanded, setExpanded] = useState(false)
  const color = SCORE_COLORS[result.model_label] || '#64748b'
  const isWinner = rank === 1

  return (
    <div className={`card border ${isWinner ? 'border-saffron-500' : 'border-slate-800'}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          {isWinner && <Trophy size={18} className="text-saffron-400 shrink-0" />}
          <div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
              <p className="font-semibold text-white">{result.model_label}</p>
            </div>
            <p className="text-xs text-slate-500 font-mono">{result.model_id}</p>
          </div>
        </div>
        <div className="text-right">
          {result.score_total != null ? (
            <>
              <p className="text-2xl font-bold text-white">{result.score_total.toFixed(1)}</p>
              <p className="text-xs text-slate-400">/ 10</p>
            </>
          ) : (
            <span className="badge bg-red-900 text-red-400">Failed</span>
          )}
        </div>
      </div>

      {result.error && (
        <div className="flex items-center gap-2 text-red-400 text-xs mb-3 p-2 bg-red-950 rounded">
          <AlertCircle size={13} /> {result.error}
        </div>
      )}

      <div className="space-y-2 mb-3">
        <ScoreDimension label="Completeness" value={result.score_completeness} />
        <ScoreDimension label="PII Safety" value={result.score_pii_safety} />
        <ScoreDimension label="Readability" value={result.score_readability} />
        <ScoreDimension label="Structure" value={result.score_structure} />
        <ScoreDimension label="Legal Terms" value={result.score_legal_terms} />
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 border-t border-slate-800 pt-2">
        <span>{result.latency_ms != null ? `${result.latency_ms.toFixed(0)}ms` : '—'}</span>
        <span>{result.tokens_used != null ? `${result.tokens_used} tokens` : '—'}</span>
        <button
          className="text-navy-400 hover:text-navy-200 flex items-center gap-1"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />} Summary
        </button>
      </div>

      {expanded && result.generated_summary && (
        <div className="mt-3 p-3 bg-slate-800 rounded-lg text-xs text-slate-300 max-h-64 overflow-y-auto whitespace-pre-wrap">
          {result.generated_summary}
        </div>
      )}
    </div>
  )
}

function RadarComparison({ results }) {
  const dims = ['completeness', 'pii_safety', 'readability', 'structure', 'legal_terms']
  const data = dims.map((dim) => {
    const entry = { subject: dim.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) }
    results.forEach((r) => {
      if (r[`score_${dim}`] != null) entry[r.model_label] = r[`score_${dim}`]
    })
    return entry
  })

  return (
    <div className="card mb-6">
      <h3 className="font-semibold text-white mb-4">Scoring Radar</h3>
      <ResponsiveContainer width="100%" height={280}>
        <RadarChart data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          {results.map((r) => (
            <Radar
              key={r.model_id}
              name={r.model_label}
              dataKey={r.model_label}
              stroke={SCORE_COLORS[r.model_label] || '#64748b'}
              fill={SCORE_COLORS[r.model_label] || '#64748b'}
              fillOpacity={0.1}
              strokeWidth={2}
            />
          ))}
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ModelComparison() {
  const [cases, setCases] = useState([])
  const [selectedCaseId, setSelectedCaseId] = useState('')
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [pastResults, setPastResults] = useState([])

  useEffect(() => {
    api.getCases(0, 50).then(setCases).catch(console.error)
    api.getModelResults().then(setPastResults).catch(console.error)
  }, [])

  const runTest = async () => {
    if (!selectedCaseId) return
    setRunning(true)
    setError(null)
    setResults([])
    try {
      const data = await api.testModels(parseInt(selectedCaseId))
      const sorted = [...data.results].sort((a, b) => (b.score_total ?? 0) - (a.score_total ?? 0))
      setResults(sorted)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const sortedPast = [...pastResults]
    .sort((a, b) => (b.score_total ?? 0) - (a.score_total ?? 0))
    .slice(0, 12)

  return (
    <div className="p-8 max-w-6xl">
      <h1 className="text-2xl font-bold text-white mb-1 flex items-center gap-2">
        <FlaskConical size={22} className="text-violet-400" /> Model Comparison
      </h1>
      <p className="text-slate-400 mb-6 text-sm">
        Run all 6 LLMs on a single case and compare quality across 5 dimensions.
      </p>

      {/* Controls */}
      <div className="card mb-6 flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-48">
          <label className="block text-xs text-slate-400 mb-1">Select a case to test</label>
          <select
            className="input"
            value={selectedCaseId}
            onChange={(e) => setSelectedCaseId(e.target.value)}
          >
            <option value="">— choose a case —</option>
            {cases.map((c) => (
              <option key={c.id} value={c.id}>
                {c.ks_id} · {c.subject?.slice(0, 50) || 'No subject'}
              </option>
            ))}
          </select>
        </div>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={runTest}
          disabled={running || !selectedCaseId}
        >
          {running ? <Loader2 size={16} className="animate-spin" /> : <FlaskConical size={16} />}
          {running ? 'Running all models…' : 'Run Comparison'}
        </button>
        {running && <p className="text-xs text-slate-500">This may take 30–90 seconds (6 API calls)…</p>}
      </div>

      {error && (
        <div className="p-4 bg-red-950 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm mb-6">
          <AlertCircle size={15} /> {error}
        </div>
      )}

      {/* Radar + cards */}
      {results.length > 0 && (
        <>
          <RadarComparison results={results} />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.map((r, i) => (
              <ModelCard key={r.model_id} result={r} rank={i + 1} />
            ))}
          </div>
        </>
      )}

      {/* Past results table */}
      {sortedPast.length > 0 && results.length === 0 && (
        <div className="card mt-4">
          <h3 className="font-semibold text-white mb-3">Previous Test Results</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 text-left border-b border-slate-800">
                  <th className="pb-2 pr-4">Model</th>
                  <th className="pb-2 pr-4">Case</th>
                  <th className="pb-2 pr-4">Total</th>
                  <th className="pb-2 pr-4">Latency</th>
                  <th className="pb-2">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {sortedPast.map((r) => (
                  <tr key={r.id} className="text-slate-300">
                    <td className="py-2 pr-4">
                      <span className="w-2 h-2 rounded-full inline-block mr-2" style={{ background: SCORE_COLORS[r.model_label] || '#64748b' }} />
                      {r.model_label}
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">KS-{String(r.case_id).padStart(6, '0')}</td>
                    <td className="py-2 pr-4 font-bold">{r.score_total?.toFixed(1) ?? '—'}</td>
                    <td className="py-2 pr-4 text-xs">{r.latency_ms?.toFixed(0)}ms</td>
                    <td className="py-2 text-xs text-slate-500">{r.created_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
