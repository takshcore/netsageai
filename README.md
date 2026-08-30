# 🔬 NetSage AI — Network Fault Diagnostics

> A **human-in-the-loop AI assistant** for diagnosing Cisco Packet Tracer network lab faults.  
> Built with Python · Streamlit · OpenAI · Pydantic

---

## 📸 Demo

![NetSage AI Dashboard](docs/screenshot.png)

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📋 **Lab Case Loader** | 10 pre-loaded Cisco network fault cases (VLAN, DHCP, RIP, ACL, IPsec, and more) |
| 🔍 **Rule Checker** | 12 deterministic regex rules that auto-flag obvious config errors |
| 🤖 **AI Agent** | GPT-powered diagnosis with structured JSON output (root cause, confidence, fix steps) |
| 👤 **Human Review** | Accept / Edit / Reject panel — all decisions logged to CSV |
| 📊 **Dashboard** | Plotly charts tracking AI accuracy, confidence vs decisions, and responsible AI corrections |

---

## 🗂️ Project Structure

```
netsage-ai/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── README.md
├── data/
│   ├── cases.csv                   # 35 network lab cases
│   └── reviews.csv                 # Human review log (pre-seeded)
├── src/
│   ├── __init__.py
│   ├── rule_checker.py             # Deterministic rule engine (12 rules)
│   └── ai_agent.py                 # OpenAI agent + Pydantic schema
└── prompts/
    ├── system_prompt.txt           # AI system instructions
    └── user_prompt_template.txt    # Per-case prompt template
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/netsage-ai.git
cd netsage-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```
Open `.env` and replace `sk-your-openai-api-key-here` with your actual key.

> 💡 **No API key?** Toggle **Demo Mode** in the sidebar — the app works fully without one.

### 5. Run the app
```bash
streamlit run app.py
```
Visit **http://localhost:8501** in your browser.

---

## 🔑 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI secret key | *(required for live AI)* |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |

---

## 🧩 How It Works

```
Select Lab Case
     ↓
Rule Checker runs automatically (regex-based, no API)
     ↓
Click "Run AI Diagnosis" → OpenAI GPT analyzes the case
     ↓
AI returns: root_cause · osi_layer · confidence · evidence · fix_steps
     ↓
Human reviews and clicks: Accept / Edit / Reject
     ↓
Decision saved to data/reviews.csv → Dashboard updates
```

---

## 📊 Responsible AI

All human corrections are logged with:
- AI's original diagnosis
- AI's confidence level
- Human's corrected answer
- Reviewer's explanation

This log is viewable and downloadable from the **Dashboard** tab.

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.2.0
pydantic>=2.7.0
openai>=1.30.0
python-dotenv>=1.0.1
plotly>=5.22.0
```

---

## 📄 License

MIT — free to use for educational purposes.
