# ReBalanceX AI Integration - Complete Implementation
# This includes both GPT-based advisor and ML foundation

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import openai
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="ReBalanceX AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
OPENAI_API_KEY = "ADD-YOUR-OWN-API-KEY"  # Replace with your actual key
openai.api_key = OPENAI_API_KEY

# Data Models
class Portfolio(BaseModel):
    equity_current: float
    debt_current: float
    gold_current: float
    equity_target: float = 60.0
    debt_target: float = 30.0
    gold_target: float = 10.0
    total_value: float = 100000  # Default portfolio value in INR

class RebalanceAdvice(BaseModel):
    current_allocation: Dict[str, float]
    target_allocation: Dict[str, float]
    drift_analysis: Dict[str, float]
    rebalance_suggestions: Dict[str, float]
    ai_advice: str
    risk_profile: str
    rebalance_amount: Dict[str, float]

# Option A: GPT-based Natural Language Advisor
class GPTAdvisor:
    def __init__(self):
        self.client = openai
    
    def generate_advice(self, portfolio: Portfolio) -> str:
        """Generate human-readable investment advice using GPT"""
        
        # Calculate drifts
        equity_drift = portfolio.equity_current - portfolio.equity_target
        debt_drift = portfolio.debt_current - portfolio.debt_target
        gold_drift = portfolio.gold_current - portfolio.gold_target
        
        prompt = f"""
        You are a financial advisor for Indian retail investors. Analyze this portfolio and provide actionable rebalancing advice:
        
        Current Allocation:
        - Equity: {portfolio.equity_current}% (Target: {portfolio.equity_target}%)
        - Debt: {portfolio.debt_current}% (Target: {portfolio.debt_target}%)
        - Gold: {portfolio.gold_current}% (Target: {portfolio.gold_target}%)
        
        Portfolio Value: ₹{portfolio.total_value:,.0f}
        
        Drifts:
        - Equity drift: {equity_drift:+.1f}%
        - Debt drift: {debt_drift:+.1f}%
        - Gold drift: {gold_drift:+.1f}%
        
        Provide advice in this format:
        1. Risk Assessment
        2. Rebalancing Priority
        3. Specific Actions
        4. Market Context (considering Indian markets)
        
        Keep it concise, practical, and suitable for retail investors.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor specializing in Indian markets and portfolio rebalancing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI advisor temporarily unavailable. Manual analysis: {'Overweight equity' if equity_drift > 5 else 'Portfolio within acceptable drift ranges'}."

# Option B: ML-Based Portfolio Analyzer (Foundation)
class MLPortfolioAnalyzer:
    def __init__(self):
        self.risk_classifier = None
        self.portfolio_clusterer = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic portfolio data for training"""
        np.random.seed(42)
        
        # Generate diverse portfolio allocations
        data = []
        for _ in range(n_samples):
            # Random allocations that sum to 100
            equity = np.random.uniform(30, 85)
            debt = np.random.uniform(10, 60)
            gold = np.random.uniform(5, 25)
            
            # Normalize to sum to 100
            total = equity + debt + gold
            equity = (equity / total) * 100
            debt = (debt / total) * 100
            gold = (gold / total) * 100
            
            # Calculate features
            volatility = equity * 0.2 + debt * 0.05 + gold * 0.15  # Simplified volatility
            expected_return = equity * 0.12 + debt * 0.07 + gold * 0.08
            
            # Risk profile based on equity allocation
            if equity > 70:
                risk_profile = 2  # Aggressive
            elif equity > 50:
                risk_profile = 1  # Moderate
            else:
                risk_profile = 0  # Conservative
                
            data.append({
                'equity': equity,
                'debt': debt,
                'gold': gold,
                'volatility': volatility,
                'expected_return': expected_return,
                'risk_profile': risk_profile
            })
        
        return pd.DataFrame(data)
    
    def train_models(self):
        """Train ML models on synthetic data"""
        # Generate training data
        df = self.generate_synthetic_data()
        
        # Features for training
        features = ['equity', 'debt', 'gold', 'volatility', 'expected_return']
        X = df[features]
        y_risk = df['risk_profile']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train risk classifier
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_classifier.fit(X_scaled, y_risk)
        
        # Train portfolio clusterer
        self.portfolio_clusterer = KMeans(n_clusters=3, random_state=42)
        self.portfolio_clusterer.fit(X_scaled)
        
        self.is_trained = True
        
    def analyze_portfolio(self, portfolio: Portfolio):
        """Analyze portfolio using ML models"""
        if not self.is_trained:
            self.train_models()
        
        # Calculate features
        volatility = portfolio.equity_current * 0.2 + portfolio.debt_current * 0.05 + portfolio.gold_current * 0.15
        expected_return = portfolio.equity_current * 0.12 + portfolio.debt_current * 0.07 + portfolio.gold_current * 0.08
        
        features = np.array([[
            portfolio.equity_current,
            portfolio.debt_current,
            portfolio.gold_current,
            volatility,
            expected_return
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Predict risk profile
        risk_pred = self.risk_classifier.predict(features_scaled)[0]
        risk_labels = ['Conservative', 'Moderate', 'Aggressive']
        
        # Get cluster
        cluster = self.portfolio_clusterer.predict(features_scaled)[0]
        
        return {
            'risk_profile': risk_labels[risk_pred],
            'cluster': int(cluster),
            'volatility': float(volatility),
            'expected_return': float(expected_return)
        }

# Initialize AI components
gpt_advisor = GPTAdvisor()
ml_analyzer = MLPortfolioAnalyzer()

# Portfolio Analysis Engine
class PortfolioEngine:
    @staticmethod
    def calculate_rebalance_amounts(portfolio: Portfolio) -> Dict[str, float]:
        """Calculate exact amounts to buy/sell for rebalancing"""
        current_values = {
            'equity': portfolio.total_value * (portfolio.equity_current / 100),
            'debt': portfolio.total_value * (portfolio.debt_current / 100),
            'gold': portfolio.total_value * (portfolio.gold_current / 100)
        }
        
        target_values = {
            'equity': portfolio.total_value * (portfolio.equity_target / 100),
            'debt': portfolio.total_value * (portfolio.debt_target / 100),
            'gold': portfolio.total_value * (portfolio.gold_target / 100)
        }
        
        rebalance_amounts = {}
        for asset in ['equity', 'debt', 'gold']:
            diff = target_values[asset] - current_values[asset]
            rebalance_amounts[asset] = round(diff, 2)
        
        return rebalance_amounts
    
    @staticmethod
    def calculate_drifts(portfolio: Portfolio) -> Dict[str, float]:
        """Calculate drift from target allocation"""
        return {
            'equity': portfolio.equity_current - portfolio.equity_target,
            'debt': portfolio.debt_current - portfolio.debt_target,
            'gold': portfolio.gold_current - portfolio.gold_target
        }

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ReBalanceX AI API", "version": "1.0.0"}

@app.post("/analyze", response_model=RebalanceAdvice)
async def analyze_portfolio(portfolio: Portfolio):
    """Complete portfolio analysis with AI recommendations"""
    
    # Validate portfolio
    total_allocation = portfolio.equity_current + portfolio.debt_current + portfolio.gold_current
    if abs(total_allocation - 100) > 0.1:
        raise HTTPException(status_code=400, detail="Portfolio allocations must sum to 100%")
    
    # Calculate basic metrics
    drifts = PortfolioEngine.calculate_drifts(portfolio)
    rebalance_amounts = PortfolioEngine.calculate_rebalance_amounts(portfolio)
    
    # Get AI analysis
    try:
        ai_advice = gpt_advisor.generate_advice(portfolio)
    except Exception as e:
        ai_advice = "AI advisor temporarily unavailable. Please check your API configuration."
    
    # Get ML analysis
    ml_analysis = ml_analyzer.analyze_portfolio(portfolio)
    
    # Build response
    return RebalanceAdvice(
        current_allocation={
            'equity': portfolio.equity_current,
            'debt': portfolio.debt_current,
            'gold': portfolio.gold_current
        },
        target_allocation={
            'equity': portfolio.equity_target,
            'debt': portfolio.debt_target,
            'gold': portfolio.gold_target
        },
        drift_analysis=drifts,
        rebalance_suggestions={
            'equity': round(portfolio.equity_target - portfolio.equity_current, 2),
            'debt': round(portfolio.debt_target - portfolio.debt_current, 2),
            'gold': round(portfolio.gold_target - portfolio.gold_current, 2)
        },
        ai_advice=ai_advice,
        risk_profile=ml_analysis['risk_profile'],
        rebalance_amount=rebalance_amounts
    )

@app.post("/quick-advice")
async def quick_gpt_advice(portfolio: Portfolio):
    """Quick GPT-based advice only"""
    try:
        advice = gpt_advisor.generate_advice(portfolio)
        return {"advice": advice}
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI advisor error")

@app.post("/risk-analysis")
async def risk_analysis(portfolio: Portfolio):
    """ML-based risk analysis"""
    analysis = ml_analyzer.analyze_portfolio(portfolio)
    return analysis

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str, period: str = "1mo"):
    """Fetch market data for Indian stocks/indices"""
    try:
        # Add .NS suffix for NSE stocks if not present
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        latest = data.iloc[-1]
        return {
            'symbol': symbol,
            'current_price': float(latest['Close']),
            'change': float(latest['Close'] - data.iloc[-2]['Close']),
            'change_percent': float(((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) * 100),
            'volume': int(latest['Volume']),
            'last_updated': data.index[-1].isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_components": {
            "gpt_advisor": "active",
            "ml_analyzer": "active" if ml_analyzer.is_trained else "training"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting ReBalanceX AI API...")
    print("Option A (GPT Advisor): Ready")
    print("Option B (ML Analyzer): Training on startup...")
    
    # Pre-train ML models
    ml_analyzer.train_models()
    print("ML models trained successfully!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)