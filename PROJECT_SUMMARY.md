# Investment Analyzer - Project Summary

A Streamlit app for analyzing stocks, ETFs, and investment portfolios using Yahoo Finance
data: historical and year-to-date performance, correlations, portfolio backtesting and
optimization, and Monte Carlo-style simulation.

See [README.md](README.md) for full feature descriptions, methodology, and project structure.

---

## 🚀 Deployment

- **Live app:** [Investment Analyzer](https://investment-analyzer-portfolios.streamlit.app/)
- **GitHub repository:** [colinlpaterson/investment-analyzer](https://github.com/colinlpaterson/investment-analyzer)
- **Branch:** `main`
- **Entrypoint:** `app.py`
- **Hosting:** Streamlit Community Cloud

---

## 🧭 Pages

| # | Page | What it does |
|---|------|--------------|
| 1 | 📊 Multi-ticker comparison | Compare price or total-return YTD performance for 2–5 tickers against the prior period |
| 2 | 📈 Historical YTD analysis | Compare a ticker's current year against historical years |
| 3 | 🔗 Correlation analysis | Static correlation heatmaps for 2–10 tickers and rolling correlation for a pair |
| 4 | 🖩 Portfolio analysis & optimization | Backtest rebalancing, review risk and drawdowns, optimize with weight caps, and assess allocations on held-out history |
| 5 | 🎲 Portfolio simulation | Simulate contributions and rebalancing with correlated historical return blocks |

Pages 4 and 5 share the saved portfolio, and an optimized allocation can be handed off
from page 4 to page 5.

---

## 🧪 Running Locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the offline test suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs the suite on every push and pull request against both the minimum
supported and current dependency versions.

---

## 📝 Data Notes

- Price return uses split-adjusted close and excludes distributions.
- Total return uses Yahoo Finance adjusted close as a proxy for reinvested distributions.
- For informational and educational purposes only; not financial advice.
