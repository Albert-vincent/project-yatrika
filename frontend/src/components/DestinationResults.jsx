import { motion } from 'framer-motion'

const ACCENTS = ['#f4a435','#2dd4bf','#fb7185','#a78bfa','#34d399']

export default function DestinationResults({ results, onChoose }) {
  return (
    <div className="results-grid">
      {results.map((dest, i) => {
        const accent = ACCENTS[i % ACCENTS.length]
        const tags = dest.activities.split(',').slice(0, 4).map(t => t.trim())
        const placeType = dest.landscape.includes(' in ')
          ? dest.landscape.split(' in ')[0]
          : dest.landscape

        return (
          <motion.div
            key={dest.destination}
            className="dest-card"
            style={{ '--card-accent': accent, '--card-color': accent }}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.35 }}
          >
            <div className="card-top">
              <span className="card-rank">#{i + 1}</span>
              <span className="card-dist">✓ {dest.dist_km.toLocaleString()} km</span>
              <span className="card-match">💜 {dest.match_pct}% mood match</span>
              <span className="card-tag" style={{ marginLeft: 'auto' }}>{placeType}</span>
            </div>

            <div className="card-name">{dest.destination}</div>
            <div className="card-state">{dest.state}{dest.region ? ` · ${dest.region}` : ''}</div>
            <div className="card-vibe">{dest.vibe}</div>
            <div className="card-desc">{dest.description.slice(0, 160)}{dest.description.length > 160 ? '...' : ''}</div>

            <div className="card-tags">
              {tags.map(t => (
                <span key={t} className="card-tag">{t}</span>
              ))}
              <span className="card-tag" style={{ color: 'var(--gold2)', borderColor: 'rgba(244,164,53,0.25)' }}>
                {dest.best_season}
              </span>
            </div>

            <button
              className="btn-choose"
              style={{ '--card-color': accent }}
              onClick={() => onChoose(dest.destination)}
            >
              Choose {dest.destination} →
            </button>
          </motion.div>
        )
      })}
    </div>
  )
}