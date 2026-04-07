import { useState } from 'react'
import Calendar from 'react-calendar'

export default function DatePickerBubble({ days, onConfirm, onSkip }) {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const [picked, setPicked] = useState(today)

  const endDate = new Date(picked)
  endDate.setDate(endDate.getDate() + days - 1)

  const fmt = d => d.toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric'
  })

  return (
    <div className="date-picker-wrap">
      <div className="label">📅 When do you want to travel?</div>
      <div className="sublabel">
        Your trip is <strong>{days} day{days > 1 ? 's' : ''}</strong> long.
        Pick a start date — the end date sets itself.
      </div>

      <Calendar
        onChange={setPicked}
        value={picked}
        minDate={today}
        locale="en-IN"
        tileDisabled={({ date }) => date < today}
        showNeighboringMonth={false}
      />

      <div className="date-range-display">
        <span>🗓</span>
        <span>
          <strong>{fmt(picked)}</strong>
          {days > 1 && <> &rarr; <strong>{fmt(endDate)}</strong></>}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--muted)' }}>
          {days} day{days > 1 ? 's' : ''}
        </span>
      </div>

      <div className="cal-actions">
        <button className="btn-confirm" onClick={() => onConfirm(picked)}>
          ✓ Confirm dates
        </button>
        <button className="btn-skip" onClick={onSkip}>
          Skip
        </button>
      </div>
    </div>
  )
}