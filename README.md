# 🌍 Yatrika — Emotion-Based Travel AI

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/yatrika.git
cd yatrika
```

---

## 🧠 Backend Setup (FastAPI)

### 📦 Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### ▶️ Run Backend

```bash
uvicorn main:app --reload --port 8000
```

Backend will run at:
👉 http://localhost:8000

---

## 🎨 Frontend Setup (React / Vite)

### 📦 Install Dependencies

```bash
cd frontend
npm install
```

### ▶️ Run Frontend

```bash
npm run dev
```

Frontend will run at:
👉 http://localhost:5173

---

## 🔗 API Connection

Make sure frontend is calling backend at:

```js
http://localhost:8000
```

---

## 🔑 Environment Variables (Backend)

Create a `.env` file inside `backend/`:

```
HF_API_KEY=your_huggingface_api_key
OWM_API_KEY=your_openweather_api_key
```

---

## 🧪 Example Input

```
burned out and need peace
```

---

## 🎯 Output

* Recommended destinations
* Emotional match reasoning
* Travel itinerary
* Weather + packing suggestions

---
