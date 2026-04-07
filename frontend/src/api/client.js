const BASE = '/api'

export async function geocodeCity(city, state = null) {
  const res = await fetch(`${BASE}/geocode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ city, state })
  })
  return res.json()
}

export async function getRadius(days, transport = 'any') {
  const res = await fetch(`${BASE}/radius?days=${days}&transport=${transport}`)
  return res.json()
}

export async function searchDestinations({ emotion, lat, lon, max_km, days, transport, origin_state }) {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ emotion, lat, lon, max_km, days, transport, origin_state })
  })
  return res.json()
}

export async function buildItinerary({ emotion, lat, lon, max_km, days, transport, origin_state, destination, start_date }) {
  const res = await fetch(`${BASE}/itinerary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ emotion, lat, lon, max_km, days, transport, origin_state, destination, start_date })
  })
  return res.json()
}

export async function warmup() {
  const res = await fetch(`${BASE}/warmup`)
  return res.json()
}