import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, ArrowRight, Sparkles, MapPin, Brain, Calendar } from 'lucide-react'
import './index.css'
import { geocodeCity, getRadius, searchDestinations, buildItinerary } from './api/client'
import DatePickerBubble from './components/DatePickerBubble'
import DestinationResults from './components/DestinationResults'
import ItineraryView from './components/ItineraryView'

// ── Conversation steps ────────────────────────────────────────────────────
const STATE_MAP = {
  tn:'tamil nadu', kl:'kerala', ka:'karnataka', mh:'maharashtra',
  goa:'goa', ap:'andhra pradesh', ts:'telangana', wb:'west bengal',
  dl:'delhi', rj:'rajasthan', up:'uttar pradesh', mp:'madhya pradesh',
  hr:'haryana', pb:'punjab', hp:'himachal pradesh', uk:'uttarakhand',
  jk:'jammu and kashmir', gj:'gujarat', or:'odisha', br:'bihar',
  sk:'sikkim', mn:'manipur', ml:'meghalaya', mz:'mizoram',
  nl:'nagaland', tr:'tripura', ar:'arunachal pradesh', ld:'ladakh', py:'puducherry',
}

function parseMarkdown(text) {
  if (!text) return ''
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br/>')
}

function Bubble({ role, text, children }) {
  return (
    <motion.div
      className={`bubble-wrap ${role}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
    >
      <div className={`avatar ${role === 'assistant' ? 'bot' : 'user'}`}>
        {role === 'assistant' ? '✈' : '🧳'}
      </div>
      <div className={`bubble ${role === 'assistant' ? 'bot' : 'user'}`}>
        {text && <p dangerouslySetInnerHTML={{ __html: parseMarkdown(text) }} />}
        {children}
      </div>
    </motion.div>
  )
}

function TypingBubble() {
  return (
    <motion.div className="bubble-wrap" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="avatar bot">✈</div>
      <div className="bubble bot">
        <div className="typing-dots"><span /><span /><span /></div>
      </div>
    </motion.div>
  )
}

// ─── Chat widget (the actual conversation) ────────────────────────────────
function ChatWidget() {
  const [messages, setMessages]       = useState([])
  const [step, setStep]               = useState('greeting')
  const [input, setInput]             = useState('')
  const [typing, setTyping]           = useState(false)
  const [disabled, setDisabled]       = useState(false)

  const [origin, setOrigin]           = useState('')
  const [originState, setOriginState] = useState(null)
  const [coords, setCoords]           = useState(null)
  const [transport, setTransport]     = useState('any')
  const [days, setDays]               = useState(null)
  const [startDate, setStartDate]     = useState(null)
  const [emotion, setEmotion]         = useState('')
  const [maxKm, setMaxKm]             = useState(99999)
  const [avoid, setAvoid]             = useState('')
  const [activities, setActivities]   = useState('')

  const [userId, setUserId]           = useState('')
  const [savedLocation, setSavedLocation] = useState(null)  // { city, state, coords }
  const [rlabel, setRlabel]           = useState('all India')
  const [results, setResults]         = useState(null)
  const [itinerary, setItinerary]     = useState(null)

  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)
  const emotionRef = useRef('')  // tracks emotion without stale closure risk

  const addMsg = useCallback((role, text) => {
    setMessages(m => [...m, { role, text, id: Date.now() + Math.random() }])
  }, [])

  const botSay = useCallback(async (text, delay = 380) => {
    setTyping(true)
    await new Promise(r => setTimeout(r, delay))
    setTyping(false)
    addMsg('assistant', text)
  }, [addMsg])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, typing, step])


  // Single startup effect — load user data first, then greet.
  // Merging into one effect eliminates the race condition where
  // the greeting ran before savedLocation was populated.
  useEffect(() => {
    const startup = async () => {
      // 1. Load or create user ID
      let id = localStorage.getItem("yatrika_user_id")
      if (!id) {
        id = "user_" + Math.random().toString(36).substring(2, 9)
        localStorage.setItem("yatrika_user_id", id)
      }
      setUserId(id)

      // 2. Load saved data synchronously before greeting
      const saved = localStorage.getItem(`yatrika_${id}`)
      const savedData = saved ? JSON.parse(saved) : null

      if (savedData) {
        setAvoid(savedData.avoid || "")
        setActivities(savedData.activities || "")
        setEmotion(savedData.emotion || "")
      }

      const hasLocation = savedData?.city && savedData?.originState && savedData?.coords
      if (hasLocation) {
        setSavedLocation({
          city: savedData.city,
          originState: savedData.originState,
          coords: savedData.coords,
        })
      }

      // 3. Greet — savedLocation is now guaranteed to be set before this runs
      if (savedData) {
        await botSay("Good to see you again! 😊 Ready for your next trip with Yatrika?", 700)
      } else {
        await botSay("Nice to meet you! 😊 Ready for your first trip with Yatrika?", 700)
      }

      if (hasLocation) {
        await botSay(
          `Last time you travelled from **${savedData.city}, ${savedData.originState.charAt(0).toUpperCase() + savedData.originState.slice(1)}**.

` +
          `Are you travelling from the same place?
- **yes** — use saved location
- **no** — enter a new location`
        )
        setStep('ask_same_location')
      } else {
        await botSay("First — **which state are you in?**\n\n*(e.g. Kerala, Tamil Nadu, Karnataka, Rajasthan)*")
        setStep('ask_state')
      }
    }
    startup()
  }, []) // eslint-disable-line

  const handleSubmit = async () => {
    const text = input.trim()
    if (!text || disabled) return
    setInput('')
    addMsg('user', text)
    setDisabled(true)
    await processInput(text)
    setDisabled(false)
    inputRef.current?.focus()
  }

  useEffect(() => {
    if (!disabled && !typing && inputRef.current) {
      const t = setTimeout(() => inputRef.current?.focus(), 50)
      return () => clearTimeout(t)
    }
  }, [disabled, typing, step])

  const processInput = async (text) => {
    const raw = text.toLowerCase().trim()

    if (step === 'ask_same_location') {
      const isYes = ['yes', 'y', 'same', 'yeah', 'yep', 'sure', 'ok', 'okay'].includes(raw)
      if (isYes && savedLocation) {
        // Restore saved location into state
        setOrigin(savedLocation.city)
        setOriginState(savedLocation.originState)
        setCoords(savedLocation.coords)
        await botSay(
          `📍 Using **${savedLocation.city}, ${savedLocation.originState.charAt(0).toUpperCase() + savedLocation.originState.slice(1)}**.\n\n` +
          `How will you travel?\n- **drive** — car / road trip\n- **fly** — include flights\n- **any** — no preference`
        )
        setStep('ask_transport')
      } else {
        await botSay("No problem! **Which state are you in?**\n\n*(e.g. Kerala, Tamil Nadu, Karnataka, Rajasthan)*")
        setStep('ask_state')
      }
      return
    }

    if (step === 'ask_state') {
      const normalised = STATE_MAP[raw] || raw
      const display = normalised.charAt(0).toUpperCase() + normalised.slice(1)
      setOriginState(normalised)
      const geo = await geocodeCity('', normalised)
      setCoords(geo.coords)
      await botSay(
        `Got it — **${display}**! Now, **which city are you travelling from?**\n\n*(Helps me calculate exact distances for you)*`
      )
      setStep('ask_origin')
      return
    }

    if (step === 'ask_origin') {
      const city = text.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
      setOrigin(city)
      setTyping(true)
      const geo = await geocodeCity(city, originState)
      setTyping(false)
      if (geo.source === 'city') {
        setCoords(geo.coords)
        await botSay(`📍 **${city}** located!\n\nHow will you travel?\n- **drive** — car / road trip\n- **fly** — include flights\n- **any** — no preference`)
      } else {
        await botSay(
          `Couldn't pinpoint **${city}** precisely — using **${originState || 'India centre'}** as your base.\n\n` +
          `How will you travel?\n- **drive** — car / road trip\n- **fly** — include flights\n- **any** — no preference`
        )
      }
      setStep('ask_transport')
      return
    }

    if (step === 'ask_transport') {
      const t = raw.includes('fly') || raw.includes('flight') || raw.includes('plane') ? 'fly'
              : raw.includes('drive') || raw.includes('car') || raw.includes('road') ? 'drive'
              : 'any'
      setTransport(t)
      await botSay(`Travelling by **${t}**.\n\nHow many days is your trip? *(e.g. 1, 3, 7)*`)
      setStep('ask_days')
      return
    }

    if (step === 'ask_days') {
      const d = parseInt(text.replace(/\D/g, ''))
      if (!d || d < 1) {
        await botSay("Please enter a valid number like **1**, **3**, or **7**.")
        return
      }
      setDays(d)
      const radius = await getRadius(d, transport)
      setMaxKm(radius.max_km)
      setRlabel(radius.label)
      const rnote = radius.max_km < 9999
        ? `For a **${d}-day ${transport} trip** I'll search within **${radius.max_km.toLocaleString()} km** of ${origin || 'you'}.`
        : `With **${d} days** all of India is on the table!`
      await botSay(`**${d} days** — perfect.\n\n${rnote}\n\n📅 Pick your travel dates below — or type a date *(DD-MM-YYYY)* or **skip**.`)
      setStep('ask_dates')
      return
    }

    if (step === 'ask_dates') {
      if (['skip','no','later','flexible','-','s'].includes(raw)) {
        await confirmDate(null)
        return
      }
      const parts = text.trim().split('-')
      if (parts.length === 3) {
        const d = new Date(`${parts[2]}-${parts[1]}-${parts[0]}`)
        if (!isNaN(d)) { await confirmDate(d); return }
      }
      await botSay("Couldn't read that date. Use **DD-MM-YYYY** *(e.g. 15-11-2025)* or type **skip**.")
      return
    }

    if (step === 'ask_emotion') {
  // Store in state AND update ref so ask_activities always gets the fresh value
  setEmotion(text)
  emotionRef.current = text

  await botSay(
    "Got it. 👍\n\nAnything you'd like to avoid on this trip?\n\n" +
    "- No crowds\n- No adventure activities\n- Avoid hot places\n- or type skip"
  )

  setStep('ask_avoid')
  return
}
if (step === 'ask_avoid') {
  const avoidInput = text
  setAvoid(avoidInput)

  await botSay(
    "Great.\n\nAny activities you'd like to include in your trip?\n\n" +
    "- Trekking and hiking\n- Just relaxing and sightseeing\n- Beach activities\n- or type skip"
  )

  setStep('ask_activities')
  return
}
if (step === 'ask_activities') {
  const activityInput = text

  // Fix: read emotion from ref — state update from ask_emotion may not have flushed yet
  const currentEmotion = emotionRef.current
  setActivities(activityInput)

  localStorage.setItem(`yatrika_${userId}`, JSON.stringify({
    avoid: avoid,
    activities: activityInput,
    emotion: currentEmotion,
    city: origin,
    originState: originState,
    coords: coords,
  }))

  await botSay(
    `*"${currentEmotion.slice(0, 80)}"*\n\nSearching within **${rlabel}** of **${origin}**...`,
    200
  )

  const oc = coords || { lat: 20.5937, lon: 78.9629 }

  try {
    const data = await searchDestinations({
      emotion: currentEmotion,
      lat: oc.lat,
      lon: oc.lon,
      max_km: maxKm,
      days,
      transport,
      origin_state: originState,
      avoid: avoid,
      activities: activityInput
    })

    setResults(data.results)

    await botSay(
      `Here are your **Top ${data.results.length} destinations** emotionally matched to ` +
      `*"${currentEmotion.slice(0, 50)}${currentEmotion.length > 50 ? '...' : ''}"* — ranked by emotional resonance:`
    )

    setStep('show_results')
  } catch {
    await botSay("Something went wrong with the search. Please try again.")
  }

  return
}
  }

  const confirmDate = async (date) => {
    setStartDate(date)
    const emotion_q =
      "**Last question — how are you feeling?** What emotional experience are you seeking?\n\n" +
      "- *\"Completely burned out, need silence and green hills\"*\n" +
      "- *\"Want heart-pounding adventure and adrenaline\"*\n" +
      "- *\"Need romance — misty mountains and slow mornings\"*"
    if (date) {
      const end = new Date(date); end.setDate(end.getDate() + days - 1)
      const fmt = d => d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
      addMsg('user', `📅 ${fmt(date)} → ${fmt(end)}`)
      await botSay(`Locked in: **${fmt(date)} → ${fmt(end)}** 🗓️\n\n${emotion_q}`)
    } else {
      addMsg('user', 'skip')
      await botSay(`Keeping dates flexible. 🗓️\n\n${emotion_q}`)
    }
    setStep('ask_emotion')
  }

  const handleChooseDestination = async (dest) => {
    addMsg('user', `I choose ${dest}`)
    await botSay(`Great choice! Building your **${days}-day ${dest}** journey...`, 300)
    setTyping(true)
    const oc = coords || { lat: 20.5937, lon: 78.9629 }
    try {
      const data = await buildItinerary({
        emotion, lat: oc.lat, lon: oc.lon, max_km: maxKm,
        days, transport, origin_state: originState, destination: dest,
        start_date: startDate
          ? `${startDate.getFullYear()}-${String(startDate.getMonth()+1).padStart(2,'0')}-${String(startDate.getDate()).padStart(2,'0')}`
          : null
      })
      setTyping(false)
      setItinerary(data)
      await botSay(`Here's your complete **${dest}** itinerary — weather, packing list and all! ✨`)
      setStep('show_itinerary')
    } catch {
      setTyping(false)
      await botSay("Couldn't build the itinerary. Please try again.")
    }
  }

  const handleReset = () => {
    // Reset all conversation state so user can plan a new trip
    setMessages([])
    setStep('greeting')
    setInput('')
    setDisabled(false)
    setOrigin('')
    setOriginState(null)
    setCoords(null)
    setTransport('any')
    setDays(null)
    setStartDate(null)
    setEmotion('')
    emotionRef.current = ''
    setMaxKm(99999)
    setAvoid('')
    setActivities('')
    setRlabel('all India')
    setResults(null)
    setItinerary(null)
    // Re-run greeting after reset
    setTimeout(async () => {
      const saved = localStorage.getItem(`yatrika_${userId}`)
      const savedData = saved ? JSON.parse(saved) : null
      const hasLocation = savedData?.city && savedData?.originState && savedData?.coords
      if (savedData) {
        await botSay("Welcome back! 😊 Ready to plan another trip?", 400)
      } else {
        await botSay("Let's plan your next trip! 😊", 400)
      }
      if (hasLocation) {
        setSavedLocation({ city: savedData.city, originState: savedData.originState, coords: savedData.coords })
        await botSay(
          `Travel from **${savedData.city}** again, or somewhere new?\n- **yes** — use saved location\n- **no** — enter a new location`
        )
        setStep('ask_same_location')
      } else {
        await botSay("First — **which state are you in?**\n\n*(e.g. Kerala, Tamil Nadu, Karnataka)*")
        setStep('ask_state')
      }
    }, 50)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }

  const showInput = !['show_results', 'show_itinerary'].includes(step)

  const placeholder =
    step === 'ask_same_location' ? "yes / no" :
    step === 'ask_state'     ? "Your state (e.g. Kerala, Rajasthan)..." :
    step === 'ask_origin'    ? "Your city (e.g. Kochi, Jaipur)..." :
    step === 'ask_transport' ? "drive / fly / any" :
    step === 'ask_days'      ? "Number of days (e.g. 3, 7)..." :
    step === 'ask_dates'     ? "DD-MM-YYYY or type skip" :
    step === 'ask_emotion'   ? "How are you feeling right now?" :
    step === 'ask_avoid' ? "Type what you want to avoid..." :
    step === 'ask_activities' ? "Type activities you like..." :
    "Type your answer..."

  return (
    <div className="chat-panel">
      {/* Browser chrome */}
      <div className="chat-chrome">
        <div className="chrome-dot red" />
        <div className="chrome-dot amber" />
        <div className="chrome-dot green" />
        <div className="chrome-url">yatrika.ai · concierge</div>
      </div>

      <div className="chat-area">
        {/* Chat header */}
        <div className="chat-header">
          <div className="chat-header-icon">✈</div>
          <div className="chat-header-text">
            <h2>Yatrika Concierge</h2>
            <p>AI · India Travel · Emotional Intelligence</p>
          </div>
          <div className="chat-status">online</div>
        </div>

        {/* Messages */}
        <div className="messages-list">
          <AnimatePresence initial={false}>
            {messages.map(m => (
              <Bubble key={m.id} role={m.role} text={m.text} />
            ))}

            {typing && <TypingBubble key="typing" />}

            {step === 'ask_dates' && days && !typing && (
              <motion.div
                key="datepicker"
                className="bubble-wrap"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="avatar bot">✈</div>
                <div className="bubble bot" style={{ maxWidth: '88%', width: '100%' }}>
                  <DatePickerBubble days={days} onConfirm={confirmDate} onSkip={() => confirmDate(null)} />
                </div>
              </motion.div>
            )}

            {step === 'show_results' && results && !typing && (
              <motion.div
                key="results"
                className="bubble-wrap"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="avatar bot">✈</div>
                <div className="bubble bot" style={{ maxWidth: '100%', width: '100%' }}>
                  <DestinationResults results={results} onChoose={handleChooseDestination} />
                </div>
              </motion.div>
            )}

            {step === 'show_itinerary' && itinerary && !typing && (
              <motion.div
                key="itinerary"
                className="bubble-wrap"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="avatar bot">✈</div>
                <div className="bubble bot" style={{ maxWidth: '100%', width: '100%' }}>
                  <ItineraryView data={itinerary} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        {showInput && (
          <div className="input-bar">
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled || typing}
              autoFocus
            />
            <button onClick={handleSubmit} disabled={disabled || typing || !input.trim()}>
              <Send size={13} />
            </button>
          </div>
        )}

        {/* Plan another trip button — shown after itinerary is complete */}
        {step === 'show_itinerary' && (
          <div className="input-bar" style={{ justifyContent: 'center' }}>
            <button
              onClick={handleReset}
              style={{
                width: '100%', padding: '0.6rem',
                background: 'transparent',
                border: '1px solid rgba(245,201,122,0.3)',
                borderRadius: '9px',
                color: 'var(--gold)',
                fontFamily: 'var(--sans)', fontSize: '0.84rem',
                cursor: 'pointer', transition: 'all 0.2s',
              }}
              onMouseOver={e => e.currentTarget.style.background = 'rgba(232,169,53,0.07)'}
              onMouseOut={e => e.currentTarget.style.background = 'transparent'}
            >
              ✦ Plan another trip
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Marquee destinations ─────────────────────────────────────────────────
const DESTINATIONS = [
  'Munnar', 'Spiti Valley', 'Hampi', 'Pondicherry', 'Ziro', 'Coorg',
  'Leh-Ladakh', 'Orchha', 'Gokarna', 'Valley of Flowers', 'Majuli',
  'Dhanushkodi', 'Tawang', 'Rann of Kutch', 'Kumarakom', 'Mawlynnong',
]

// ─── Main site ────────────────────────────────────────────────────────────
export default function App() {
  const chatRef = useRef(null)

  const scrollToChat = () => {
    chatRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div>
      {/* ── Navigation ── */}
      <nav className="site-nav">
        <div className="nav-brand">
          <h1>Yatrika</h1>
          <span>AI Concierge</span>
        </div>
        <div className="nav-links">
          <a href="#how">How it works</a>
          <a href="#destinations">Destinations</a>
          <a href="#concierge">Concierge</a>
          <a href="#concierge" className="nav-cta">Start Planning</a>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-eyebrow">
          AI-Powered · Emotionally Intelligent · India
        </div>

        <h1 className="hero-title">
          Travel that speaks<br />
          to your <em>soul</em>
        </h1>

        <p className="hero-sub">
          Tell Yatrika how you feel — burned out, seeking adventure, craving solitude.
          We find the Indian destination that matches your exact emotional state, reachable within your time and budget.
        </p>

        <div className="hero-actions">
          <button className="btn-primary" onClick={scrollToChat}>
            Plan my journey <ArrowRight size={15} />
          </button>
          <button className="btn-secondary" onClick={() => document.getElementById('how').scrollIntoView({ behavior: 'smooth' })}>
            See how it works
          </button>
        </div>

        <div className="hero-stats">
          <div className="stat-item">
            <span className="stat-val">500+</span>
            <span className="stat-lbl">Destinations</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">28</span>
            <span className="stat-lbl">States covered</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">∞</span>
            <span className="stat-lbl">Mood combinations</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">Live</span>
            <span className="stat-lbl">Weather data</span>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="features-strip">
        <div className="features-inner">
          <div className="feature-item">
            <div className="feature-icon" style={{ color: 'var(--gold-2)' }}>
              <Brain size={18} />
            </div>
            <div className="feature-title">Emotionally Matched</div>
            <p className="feature-desc">
              Describe your mood in plain language. Yatrika reads between the lines and finds places that heal, excite, or restore — whatever you need.
            </p>
          </div>
          <div className="feature-item">
            <div className="feature-icon" style={{ color: 'var(--teal)' }}>
              <MapPin size={18} />
            </div>
            <div className="feature-title">Actually Reachable</div>
            <p className="feature-desc">
              No fantasy destinations you can't reach in time. Every suggestion is calculated against your city, travel mode, and trip duration.
            </p>
          </div>
          <div className="feature-item">
            <div className="feature-icon" style={{ color: 'var(--purple)' }}>
              <Calendar size={18} />
            </div>
            <div className="feature-title">Complete Itinerary</div>
            <p className="feature-desc">
              Day-by-day plans, live weather, and a smart packing list tailored to your destination's climate and activities.
            </p>
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className="how-section" id="how">
        <div className="section-eyebrow">Process</div>
        <h2 className="section-title">From feeling to <em>journey</em> in minutes</h2>

        <div className="steps-grid">
          {[
            {
              num: '01', color: 'var(--gold-2)',
              title: 'Tell us where you are',
              desc: 'Your state and city lets us calculate real travel distances — no guessing.'
            },
            {
              num: '02', color: 'var(--teal)',
              title: 'Set your constraints',
              desc: 'Drive or fly? How many days? Dates? We work within your reality, not around it.'
            },
            {
              num: '03', color: 'var(--purple)',
              title: 'Describe your mood',
              desc: 'Be honest. "Exhausted and need silence." "Craving the ocean." We\'ve heard it all.'
            },
            {
              num: '04', color: 'var(--sage)',
              title: 'Receive your journey',
              desc: 'Emotionally-ranked destinations, a full itinerary, weather forecasts and a packing list.'
            },
          ].map(s => (
            <div key={s.num} className="step-card" style={{ '--step-color': s.color }}>
              <div className="step-num">{s.num}</div>
              <div className="step-title">{s.title}</div>
              <p className="step-desc">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Destinations Marquee ── */}
      <section className="marquee-section" id="destinations">
        <div className="marquee-track">
          {[...DESTINATIONS, ...DESTINATIONS].map((d, i) => (
            <span key={i} className="marquee-item">
              {d}
              <span className="marquee-dot" />
            </span>
          ))}
        </div>
      </section>

      {/* ── Quote ── */}
      <section className="quote-section">
        <div className="quote-inner">
          <span className="quote-mark">"</span>
          <p className="quote-text">
            The real voyage of discovery consists not in seeking new landscapes, but in having new eyes.
          </p>
          <p className="quote-attr">Marcel Proust · In Search of Lost Time</p>
        </div>
      </section>

      {/* ── Chat Section ── */}
      <section className="chat-section" id="concierge" ref={chatRef}>
        <div className="chat-section-header">
          <div className="section-eyebrow">Your Concierge</div>
          <h2 className="section-title" style={{ margin: '0 auto 0.75rem', textAlign: 'center', maxWidth: '100%' }}>
            Start your <em>conversation</em>
          </h2>
          <p style={{ color: 'var(--text-2)', fontSize: '0.88rem', lineHeight: 1.7 }}>
            Yatrika is ready. Tell her where you are and how you feel.
          </p>
        </div>

        <ChatWidget />
      </section>

      {/* ── Footer ── */}
      <footer>
        <div className="site-footer">
          <div className="footer-brand">Yatrika</div>
          <p className="footer-copy">© 2026 · AI Travel Concierge for India</p>
          <div className="footer-links">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
            <a href="#">About</a>
          </div>
        </div>
      </footer>
    </div>
  )
}