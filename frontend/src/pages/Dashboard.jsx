import { useEffect, useState } from 'react'
import { Scale, FileText, FlaskConical, Database, TrendingUp } from 'lucide-react'
import { api } from '../api/client'

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="card flex items-center gap-4">
      <div className={`p-3 rounded-lg ${color}`}>
        <Icon size={22} className="text-white" />
      </div>
      <div>
        <p className="text-slate-400 text-sm">{label}</p>
        <p className="text-2xl font-bold text-white">{value ?? '—'}</p>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getStats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Scale className="text-saffron-400" size={30} />
          KanoonSaathi
        </h1>
        <p className="text-slate-400 mt-1">
          Legal intelligence platform for Supreme Court of India judgments (1950–2024)
        </p>
      </div>

      {loading ? (
        <p className="text-slate-500">Loading stats…</p>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <StatCard icon={Database} label="Cases Ingested" value={stats?.total_cases} color="bg-navy-500" />
            <StatCard icon={FileText} label="Documents Created" value={stats?.documents_generated} color="bg-emerald-600" />
            <StatCard icon={FlaskConical} label="Model Tests Run" value={stats?.model_tests_run} color="bg-violet-600" />
            <StatCard icon={TrendingUp} label="Cases Indexed" value={stats?.vector_store_size} color="bg-saffron-500" />
          </div>

          <div className="card mb-6">
            <h2 className="text-lg font-semibold text-white mb-3">Available LLM Models</h2>
            <div className="flex flex-wrap gap-2">
              {(stats?.models_available || []).map((m) => (
                <span key={m} className="badge bg-navy-800 text-navy-100">{m}</span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="card">
              <h3 className="font-semibold text-white mb-2">Ingestion Pipeline</h3>
              <p className="text-slate-400 text-sm">
                Upload your SC judgment CSV. The pipeline automatically scrubs PII (names,
                Aadhaar, PAN, phones) and generates anonymized markdown documents stored locally.
              </p>
            </div>
            <div className="card">
              <h3 className="font-semibold text-white mb-2">Judgement Finder</h3>
              <p className="text-slate-400 text-sm">
                Paste a snippet from your case. The vector similarity engine (FAISS +
                sentence-transformers) finds the most relevant precedents from the corpus.
              </p>
            </div>
            <div className="card">
              <h3 className="font-semibold text-white mb-2">Model Comparison</h3>
              <p className="text-slate-400 text-sm">
                Benchmark Kimi K2, Llama 3.3 70B, Gemma 2, DeepSeek R1, Qwen QwQ, and Mixtral
                on document quality across 5 scoring dimensions.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
