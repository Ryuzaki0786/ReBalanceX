# 🎯 ReBalanceX – AI-Powered Portfolio Rebalancer for Indian Investors

## 🔍 The Problem
In India, millions of retail investors actively participate in equity markets, mutual funds, and gold-based assets. While these investors often start with a target asset allocation (e.g., 60% equity, 30% debt, 10% gold), they rarely rebalance their portfolios, leading to:

- Overexposure to market volatility  
- Suboptimal risk-adjusted returns  
- Missed tax-efficiency opportunities  
- Emotional or uninformed investment decisions  

The existing solutions for rebalancing are either locked inside premium wealth apps or too manual and fragmented across spreadsheets and platforms.

---

## 🎯 The Solution – ReBalanceX
ReBalanceX is an AI-powered web application that helps Indian retail investors:

- Analyze their current portfolio against target allocation  
- Identify drift in asset classes like equity, debt, and gold  
- Receive actionable rebalancing suggestions  
- Get AI-generated advice tailored to their risk profile  
- Visualize portfolio composition and correction path  

---

## 👥 Target Users / Clients
- Retail investors managing portfolios via Zerodha, Groww, Upstox, etc.  
- Young professionals starting SIPs but unsure about balancing equity vs debt  
- HNI & DIY investors who want intelligent, tax-aware rebalancing  
- Robo-advisor startups or fintech apps looking for backend logic/API integration  
- Wealth managers offering digital advisory tools to clients  

---

## 💡 What Makes It Stand Out
- 🤖 GPT-4 backed AI Advisor for human-like guidance  
- 🧠 ML-powered risk profile classification  
- ⚙️ FastAPI-based backend API for scalable performance  
- 🎨 Fully responsive HTML/CSS/JS frontend for clean UX  
- 📈 Real-time integration with market data  

---

## 🌱 Future Vision
> “ReBalanceX is not just a tool — it’s the foundation of a new kind of digital advisor that empowers Indian investors to make rational, personalized, and tax-efficient decisions.”

Planned future enhancements:
- Mutual fund & ETF rebalancing integration (via CAMS/Karvy data)  
- Tax-loss harvesting recommendations  
- AI risk assessment quiz for new investors  
- Secure user authentication and portfolio storage  
- Email/PDF reports & rebalancing reminders  
- Deployment as a SaaS API for fintech partners  

---

## 🛠 Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** FastAPI, Python  
- **AI:** OpenAI GPT-4, Scikit-learn (ML)  
- **APIs:** yFinance for market data  

---

## 🚀 Getting Started
1. Run the backend:
```bash
uvicorn rebalancex_api:app --reload
```

2. Serve frontend:
```bash
python -m http.server 8001
```

3. Open `Frontend.html` in your browser and use the form!

---

## 📄 License
MIT License

---

## ✨ Author
Built by [Ekam Bhullar] – making quant tools that solve real problems.
