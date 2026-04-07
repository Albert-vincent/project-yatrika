import { motion } from 'framer-motion'

const WEATHER_TILES = [
  { key: 'temp_min', label: 'Min',      suffix: '°C',    color: '#2dd4bf' },
  { key: 'temp_max', label: 'Max',      suffix: '°C',    color: '#fb7185' },
  { key: 'temp_avg', label: 'Avg',      suffix: '°C',    color: '#f4a435' },
  { key: 'humidity', label: 'Humidity', suffix: '%',     color: '#a78bfa' },
  { key: 'wind_kmh', label: 'Wind',     suffix: ' km/h', color: '#34d399' },
]

const PACK_SECTIONS = [
  { key: 'clothing',  title: 'Clothing' },
  { key: 'gear',      title: 'Gear' },
  { key: 'health',    title: 'Health' },
  { key: 'documents', title: 'Documents' },
  { key: 'extras',    title: 'Extras' },
]

export default function ItineraryView({ data }) {
  const { itinerary, weather, packing } = data
  if (!itinerary?.length) return null
  const primary = itinerary[0]

  return (
    <div className="itin-wrap">
      <div className="itin-header">
        {primary.destination}
        {itinerary.length > 1 && ` + ${itinerary.length - 1} more`}
      </div>

      <div className="sec-label">Day by day</div>
      {itinerary.map((stop, i) => (
        <motion.div
          key={i}
          className="stop-card"
          style={{ '--stop-accent': stop.type === 'primary' ? '#f4a435' : '#2dd4bf' }}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1 }}
        >
          <div className="stop-header">
            <span className="stop-badge">{stop.type === 'primary' ? 'Main stop' : 'Side trip'}</span>
            <span className="stop-days">
              {stop.day_range}{stop.date_range ? ` · ${stop.date_range}` : ''} · {stop.days} day{stop.days > 1 ? 's' : ''}
            </span>
          </div>

          <div className="stop-name">{stop.destination}</div>
          <div className="stop-state">{stop.state} · {stop.landscape}</div>

          {stop.narrative ? (
            <div className="stop-narrative">{stop.narrative}</div>
          ) : (
            <div className="stop-desc">{stop.description}</div>
          )}

          <div className="stop-acts">🎯 {stop.activities}</div>
        </motion.div>
      ))}

      <div className="sec-label" style={{ marginTop: '1rem' }}>
        Weather — {weather.city}
        {!weather.is_live && (
          <span style={{ marginLeft: '0.5rem', opacity: 0.45, fontSize: '0.7rem' }}>(estimated)</span>
        )}
      </div>
      {weather.conditions?.length > 0 && (
        <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>
          {weather.conditions.join(' · ')}
        </div>
      )}
      <div className="weather-grid">
        {WEATHER_TILES.map(t => (
          <div key={t.key} className="weather-tile" style={{ '--tile-color': t.color }}>
            <div className="val">{weather[t.key]}{t.suffix}</div>
            <div className="lbl">{t.label}</div>
          </div>
        ))}
      </div>

      <div className="sec-label" style={{ marginTop: '1rem' }}>Packing list</div>
      <div className="packing-grid">
        {PACK_SECTIONS.map(s =>
          packing[s.key]?.length > 0 && (
            <div key={s.key} className="pack-section">
              <div className="pack-title">{s.title}</div>
              {packing[s.key].map((item, idx) => (
                <div key={idx} className="pack-item">{item}</div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}