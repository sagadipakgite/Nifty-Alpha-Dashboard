import math
import io
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objs as go
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="NIFTY ALPHA INTELLIGENCE SYSTEM — v11", layout="wide")

# --- universe
NIFTY50 = [
    'ADANIENT.NS','ADANIPORTS.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AXISBANK.NS',
    'BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BEL.NS','BHARTIARTL.NS',
    'BPCL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DIVISLAB.NS',
    'DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HDFCBANK.NS',
    'HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS','ICICIBANK.NS',
    'INDUSINDBK.NS','INFY.NS','ITC.NS','JSWSTEEL.NS','KOTAKBANK.NS',
    'LT.NS','M&M.NS','MARUTI.NS','NESTLEIND.NS','NTPC.NS',
    'ONGC.NS','POWERGRID.NS','RELIANCE.NS','SBILIFE.NS','SBIN.NS',
    'SHREECEM.NS','SUNPHARMA.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS',
    'TCS.NS','TECHM.NS','TITAN.NS','ULTRACEMCO.NS','WIPRO.NS'
]

HORIZONS = {"7d":7,"15d":15,"1m":30,"3m":90,"6m":180,"1y":365}

# ---------------- utilities ----------------
def to_scalar(v):
    try:
        if isinstance(v, (pd.Series, np.ndarray, list)):
            return float(pd.Series(v).dropna().iloc[-1])
        return float(v)
    except Exception:
        return float('nan')

def display_val(v, is_currency=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        if is_currency:
            x = float(v)
            if x >= 1e12: return f"₹{x/1e12:,.2f}T"
            if x >= 1e9: return f"₹{x/1e9:,.2f}B"
            if x >= 1e6: return f"₹{x/1e6:,.2f}M"
            return f"₹{x:,.0f}"
        return str(v)
    except Exception:
        return str(v)

# ---------------- indicators ----------------
def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean(); ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['ATR'] = compute_atr(df)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

# ---------------- caching ----------------
@st.cache_data(ttl=1800)
def fetch_data_cached(ticker, days):
    try:
        df = yf.download(ticker,
                         start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                         end=datetime.now().strftime('%Y-%m-%d'),
                         progress=False, threads=False, timeout=15)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = compute_indicators(df)
    return df

# ---------------- fundamentals (best-effort) ----------------
@st.cache_data(ttl=1800)
def fetch_and_compute_fundamentals(ticker, price=None):
    t = yf.Ticker(ticker)
    basic = {"Market Cap": None, "P/E": None, "EPS": None, "52W High": None, "52W Low": None, "Dividend Yield": None}
    try:
        info = getattr(t, "info", {}) or {}
    except Exception:
        info = {}
    mcap = info.get("marketCap") or info.get("market_cap") or info.get("marketCapRaw")
    basic["Market Cap"] = mcap
    eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths") or info.get("earningsPerShare")
    basic["EPS"] = eps
    pe = info.get("trailingPE") or info.get("priceToEarnings") or info.get("forwardPE")
    if (pe is None or pe==0) and eps and price:
        try:
            pe = float(price)/float(eps)
        except Exception:
            pe = None
    basic["P/E"] = pe
    basic["52W High"] = info.get("fiftyTwoWeekHigh") or info.get("yearHigh") or info.get("52WeekHigh")
    basic["52W Low"] = info.get("fiftyTwoWeekLow") or info.get("yearLow") or info.get("52WeekLow")
    basic["Dividend Yield"] = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    return basic

# ---------------- scoring ----------------
def score_stock(df, horizon_label):
    last = df.iloc[-1]
    Close = to_scalar(last.get('Close', np.nan))
    EMA20 = to_scalar(last.get('EMA20', np.nan))
    EMA50 = to_scalar(last.get('EMA50', np.nan))
    EMA200 = to_scalar(last.get('EMA200', np.nan))
    RSI = to_scalar(last.get('RSI', np.nan))
    ATR = to_scalar(last.get('ATR', np.nan))
    MACDh = to_scalar(last.get('MACD_hist', np.nan))
    score = 0.0
    if horizon_label in ["7d", "15d", "1m"]:
        try:
            closes = df['Close'].dropna()
            if len(closes) >= 2:
                recent_return = float(closes.iloc[-1]) / float(closes.iloc[0]) - 1.0
            else:
                recent_return = 0.0
        except Exception:
            recent_return = 0.0
        if recent_return > 0: score += 0.3
        if Close > EMA20: score += 0.25
        if EMA20 > EMA50: score += 0.15
        if MACDh > 0: score += 0.15
        if ATR > 0 and Close>0 and (ATR / Close) < 0.02: score += 0.15
    elif horizon_label in ["3m","6m"]:
        if Close > EMA20: score += 0.35
        if 40 < RSI < 70: score += 0.25
        if MACDh > 0: score += 0.2
        if Close > EMA50: score += 0.2
    else:
        if Close > EMA200: score += 0.4
        if Close > EMA50: score += 0.25
        if Close > EMA20: score += 0.15
        if MACDh > 0: score += 0.1
        if 30 < RSI < 80: score += 0.1
    return round(min(score, 1.0), 4)

def recommendation_from_score(score):
    if score >= 0.8: return "Strong Buy"
    if score >= 0.6: return "Buy"
    if score >= 0.4: return "Hold"
    return "Avoid / Sell"



# ---------------- plotting ----------------
def plot_interactive(df, ticker, points=500):
    df = df.copy().tail(points)
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
    # EMA lines
    for ema in ['EMA20','EMA50','EMA200']:
        if ema in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema], name=ema, line={'width':1}))
    # MACD as separate yaxis
    if 'MACD_hist' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD_hist', yaxis='y2', opacity=0.6))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', yaxis='y2', line={'width':1}))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD_sig', yaxis='y2', line={'width':1, 'dash':'dash'}))
    # RSI on its own axis
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', yaxis='y3', line={'width':1}))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, hovermode="x unified",
                      yaxis=dict(domain=[0.35,1]), yaxis2=dict(domain=[0.15,0.33], anchor='x', title='MACD'),
                      yaxis3=dict(domain=[0.0,0.12], anchor='x', title='RSI'))
    return fig

# ---------------- end plotting ----------------


# ---------------- allocation helpers ----------------
def equal_weight_alloc(tickers):
    n = len(tickers)
    if n == 0: return {t:1/n for t in tickers}
    w = {t: 1.0/n for t in tickers}
    return w

def risk_parity_alloc(price_dict, atr_dict):
    vols = {}
    for t,p in price_dict.items():
        a = atr_dict.get(t, np.nan)
        vols[t] = (a / p) if p and a and p>0 else 0.0
    inv = {t: (1.0/max(v,1e-9)) for t,v in vols.items()}
    s = sum(inv.values())
    if s==0: return {t:0 for t in price_dict}
    return {t: inv[t]/s for t in price_dict}

def factor_weight_alloc(scores):
    s = sum(scores.values())
    if s==0: return {t:0 for t in scores}
    return {t: scores[t]/s for t in scores}

def monte_carlo_max_sharpe(price_df, returns_df, n_portfolios=5000, risk_free=0.06):
    tickers = list(price_df.columns)
    mean_ret = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    best = None
    best_sharpe = -999
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(tickers)))
        port_ret = np.dot(w, mean_ret)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (port_ret - risk_free) / (port_vol+1e-9)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = w
    return {t: float(best[i]) for i,t in enumerate(tickers)}, float(best_sharpe)

# ---------------- portfolio metrics ----------------
def portfolio_metrics(weights, mean_ret, cov, risk_free=0.06):
    w = np.array([weights[t] for t in weights])
    port_ret = float(np.dot(w, mean_ret) * 252)
    port_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    sharpe = (port_ret - risk_free) / (port_vol+1e-9)
    return {"exp_return": port_ret, "volatility": port_vol, "sharpe": sharpe}

# ---------------- UI ----------------
st.title("Nifty Alpha Dashboard")
st.markdown("""
**The Nifty Alpha Dashboard is a data-driven investment analysis system designed to help investors evaluate NIFTY 50 opportunities with clarity and structure. The platform brings together market trends, volatility behavior, and essential fundamentals to highlight stocks showing improving strength or emerging weakness.**

Each company is assessed through a consistent scoring framework that reflects its technical momentum and overall market positioning. Using these insights, the dashboard identifies potential opportunities, benchmarks performance across time horizons, and generates portfolio allocations aligned with an investors risk tolerance.

The interface focuses on clear visuals and decision-ready information, including expected returns, allocation breakdowns, share quantities, and a concise explanation behind each recommendation. By simplifying complex data into a straightforward workflow, the Nifty Alpha Dashboard supports more informed, disciplined, and confident investment decisions.
""")

with st.sidebar:
    st.header("Controls")
    keys = list(HORIZONS.keys())
    default_index = keys.index("1y") if "1y" in keys else 0
    horizon_choice = st.selectbox("Time Horizon", keys, index=default_index)
    capital = st.number_input("Portfolio Capital (₹)", value=500000, step=10000, min_value=10000)
    max_drawdown = st.number_input("Max Portfolio Drawdown (%)", value=10.0, step=1.0, min_value=0.0)
    run_button = st.button("Run Analysis")

# run button placed just below controls


horizon_days = HORIZONS[horizon_choice]

if run_button:
    st.info("Fetching data and running analysis... (parallel fetch)")
    results = []
    fetch_days = max(365, horizon_days*3)
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_ticker = {ex.submit(fetch_data_cached, t, fetch_days): t for t in NIFTY50}
        for fut in as_completed(future_to_ticker):
            ticker = future_to_ticker[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is None:
                continue
            s = score_stock(df, horizon_choice)
            last = df.iloc[-1]
            price = to_scalar(last['Close'])
            atr = to_scalar(last['ATR'])
            vol = float(df['Close'].pct_change().dropna().std() * math.sqrt(252))
            fundamentals = fetch_and_compute_fundamentals(ticker, price=price)
            results.append({"Ticker": ticker, "Score": s, "DF": df, "Price": price, "ATR": atr, "Vol": vol, "Fundamentals": fundamentals})
    if not results:
        st.error("No data fetched.")
    else:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
        st.subheader("Top Recommendations (by score)")
        st.dataframe(df_res[['Ticker','Score','Price']].head(20))

        # Candidate lists
        top_momentum = df_res.head(10)['Ticker'].tolist()
        low_vol = df_res.sort_values("Vol")['Ticker'].head(10).tolist()
        balanced = list(pd.concat([df_res.head(6)['Ticker'], df_res.sort_values("Score").head(6)['Ticker']]).unique())[:10]

        # Use internal defaults for allocation engine
        price_dict = {r['Ticker']: r['Price'] for r in results}
        atr_dict = {r['Ticker']: r['ATR'] for r in results}
        score_dict = {r['Ticker']: r['Score'] for r in results}

        st.markdown("### Alternative Portfolios (with theory & per-stock rationale)")
        
        # Determine stock counts based on investor risk capacity (max_drawdown)
        # Interpretation: larger allowed drawdown => higher risk capacity
        md = float(max_drawdown)  # percentage number from control
        if md >= 15:
            # high risk capacity
            counts = {"High Momentum": 5, "Balanced": 5, "Low Volatility": 3}
        elif md >= 8:
            # moderate risk capacity
            counts = {"High Momentum": 4, "Balanced": 6, "Low Volatility": 6}
        else:
            # low risk capacity
            counts = {"High Momentum": 2, "Balanced": 7, "Low Volatility": 9}
        # ensure minimum 2
        for k in counts:
            if counts[k] < 2:
                counts[k] = 2
        # Build portfolios dynamically from ranked lists
        portfolios = {"High Momentum": top_momentum[:counts["High Momentum"]], "Balanced": balanced[:counts["Balanced"]], "Low Volatility": low_vol[:counts["Low Volatility"]]}
        for name, tickers in portfolios.items():
    
            st.markdown(f"#### {name} — Theory & Explanation")
            if name == "High Momentum":
                st.write("Theory: Concentrated momentum strategy — selects recent market leaders. Goal: capture trend continuation; higher expected returns at higher volatility. Appropriate for investors with higher risk tolerance and shorter time horizons.")
            elif name == "Low Volatility":
                st.write("Theory: Defensive low-volatility portfolio — selects less volatile names aiming for lower drawdowns and steadier returns. Appropriate for conservative investors.")
            else:
                st.write("Theory: Balanced mix combining momentum and fundamental stability — designed for medium risk appetite and longer horizons.")

            # Allocation logic (internal defaults): Factor-weighted for momentum, risk-parity for low-vol, equal for balanced
            if name == "High Momentum":
                weights = factor_weight_alloc({t: score_dict.get(t,0) for t in tickers})
            elif name == "Low Volatility":
                weights = risk_parity_alloc({t: price_dict.get(t, np.nan) for t in tickers}, {t: atr_dict.get(t, np.nan) for t in tickers})
            else:
                weights = equal_weight_alloc(tickers)

            # show table and per-stock reasons
            rows = []
            for t in tickers:
                pr = price_dict.get(t)
                w = weights.get(t,0)
                alloc = w * capital
                qty = math.floor(alloc / pr) if pr and pr>0 else 0
                fund = next((r['Fundamentals'] for r in results if r['Ticker']==t), {})
                reason = []
                # generate short reason using score and fundamentals
                sc = score_dict.get(t,0)
                if sc >= 0.7:
                    reason.append("Strong technical score (momentum + trend)." )
                elif sc >=0.5:
                    reason.append("Moderate technical score; watch momentum." )
                else:
                    reason.append("Lower technical score; use conservative sizing.")
                if fund.get('Market Cap'):
                    reason.append(f"Market cap {display_val(fund.get('Market Cap'), True)} indicates {'large' if fund.get('Market Cap')>5e10 else 'mid/small'}-cap status.")
                if fund.get('P/E'):
                    reason.append(f"P/E: {round(fund.get('P/E'),2)}" )
                if fund.get('EPS'):
                    reason.append(f"EPS: {display_val(fund.get('EPS'))}" )
                rows.append({"Ticker":t, "Weight(%)": round(w*100,2), "Allocation(₹)": display_val(alloc, True), "Price": display_val(pr, True), "Quantity": qty, "Fundamentals": fund, "Reason": ' '.join(reason)})
            table = pd.DataFrame(rows).sort_values("Weight(%)", ascending=False).reset_index(drop=True)
            st.dataframe(table, use_container_width=True)
            # detailed per-stock fundamentals + explicit reason

            # --- Forecasts for each stock (simple mean-return projection)
            horizon_days = HORIZONS[horizon_choice]
            forecasts = []
            for t in tickers:
                df_t = next((r['DF'] for r in results if r['Ticker']==t), None)
                price_t = price_dict.get(t, None)
                if df_t is None or df_t.empty or price_t is None:
                    forecasts.append({"Ticker": t, "ExpectedPrice": None, "ExpectedReturn%": None})
                    continue
                mean_daily = df_t['Close'].pct_change().dropna().mean()
                expected_price = price_t * ((1 + mean_daily) ** horizon_days)
                expected_ret = float((expected_price / price_t - 1) * 100.0)
                expected_price = float(expected_price) if expected_price is not None else None
                forecasts.append({"Ticker": t, "ExpectedPrice": expected_price, "ExpectedReturn%": expected_ret})
            # portfolio-level forecast (weighted)
            port_expected = 0.0
            for f in forecasts:
                if f['ExpectedReturn%'] is not None:
                    port_expected += (f['ExpectedReturn%']/100.0) * (weights.get(f['Ticker'],0))
            # show compact forecast table
            fc_df = pd.DataFrame(forecasts).dropna(subset=['ExpectedReturn%']).sort_values('ExpectedReturn%', ascending=False)
            if not fc_df.empty:
                st.markdown('**Forecast (simple projection)**') 
                # format ExpectedPrice and ExpectedReturn%
                fc_df_display = fc_df.copy()
                fc_df_display['ExpectedPrice'] = fc_df_display['ExpectedPrice'].apply(lambda v: display_val(v, True) if v is not None else 'N/A')
                fc_df_display['ExpectedReturn%'] = fc_df_display['ExpectedReturn%'].apply(lambda v: f"{v:.2f}%" if v is not None else 'N/A')
                st.dataframe(fc_df_display[['Ticker','ExpectedPrice','ExpectedReturn%']], use_container_width=True)
                st.write(f"Portfolio expected return (weighted): {port_expected*100:.2f}% over chosen horizon (simple projection)") 
            else:
                st.write('Forecasts not available for selected tickers.') 
                st.markdown("**Per-stock fundamentals & rationale**")
            for r in rows:
                fundamentals = r.get('Fundamentals', {})
                st.markdown(f"- **{r['Ticker']}** — Price: {r['Price']}, Allocation: {r['Allocation(₹)']}")
                st.markdown(f"  - Fundamentals: Market Cap: {display_val(fundamentals.get('Market Cap'), True)}, P/E: {display_val(fundamentals.get('P/E'))}, EPS: {display_val(fundamentals.get('EPS'))}, 52W H/L: {display_val(fundamentals.get('52W High'))}/{display_val(fundamentals.get('52W Low'))}")
                st.markdown(f"  - Rationale: {r['Reason']}")

                # Chart expander for this stock
                df_chart = next((rr['DF'] for rr in results if rr['Ticker']==r['Ticker']), None)
                if df_chart is not None and not df_chart.empty:
                    with st.expander(f"Show chart — {r['Ticker']}", expanded=False):
                        fig_chart = plot_interactive(df_chart, r['Ticker'])
                        st.plotly_chart(fig_chart, use_container_width=True, key=f"chart_{r['Ticker']}_{uuid.uuid4()}")

            # TradingView removed as requested# TradingView removed as requested

            
            st.markdown("### Correlation Matrix (Top 20 by score)")
        top20 = df_res.head(20)['Ticker'].tolist()
        price_series = []
        for t in top20:
            df = next((r['DF'] for r in results if r['Ticker']==t), None)
            if df is not None:
                try:
                    s = df['Close'].copy()
                    s.name = str(t)
                    price_series.append(s)
                except Exception:
                    continue
        if price_series:
            price_df = pd.concat(price_series, axis=1).dropna()
            corr = price_df.pct_change().corr()
            st.dataframe(corr.round(3))

        
        # Recommend one portfolio based on investor risk capacity (simple rule)
        md = float(max_drawdown)
        if md >= 15:
            recommended = "High Momentum"
        elif md >= 8:
            recommended = "Balanced"
        else:
            recommended = "Low Volatility"
        st.markdown(f"### Recommended portfolio for your risk profile: **{recommended}** — based on max drawdown = {md}%") 
    
        st.success("Analysis complete. The sidebar contains only three controls as requested.")

        st.caption("v11 — TradingView embed + custom indicators, fixed bugs, three controls only.")