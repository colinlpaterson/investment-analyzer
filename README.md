# Stock Performance Analyzer

Powerful tools for exploring and analyzing stocks, ETFs, and investment portfolios—from historical performance and correlations to portfolio optimization and simulation.

## Features

### 📊 Multi-Ticker Comparison
- Compare price or total-return YTD performance across multiple tickers simultaneously
- View last full year's performance (dashed lines) alongside current YTD
- Color-coded visualization for easy comparison
- Available for any ticker supported by Yahoo Finance

### 📈 Historical YTD Performance
- View price return or adjusted-close total return for any stock ticker
- Compare current year against historical years
- Available for any ticker supported by Yahoo Finance

### 🔗 Correlation Analysis
- Compare Pearson correlations across 2–10 stocks and ETFs
- Choose price or total returns, lookback period, and return frequency
- View an annotated heatmap and the highest and lowest correlation pairs
- Track rolling correlation through time for a selected ticker pair
- Export the correlation matrix or rolling series as CSV

### 🖩 Portfolio analysis & optimization (stages 1–5)
- Build a long-only, fully invested portfolio with an editable ticker/weight table
- Validate weights or explicitly normalize them to 100%
- Save dates, rebalancing frequency, an optional benchmark, an annual risk-free assumption, and an annual minimum acceptable return (MAR)
- Share the saved portfolio between the Portfolio Analysis and Portfolio Simulation pages
- Check adjusted-close availability, quote currencies, and the common historical period
- View per-ticker coverage and explanations when the requested history is shortened
- Backtest monthly, quarterly, or annual rebalancing, or buy and hold
- View total return, CAGR, annualized volatility, Sharpe ratio, Sortino ratio, and maximum drawdown
- Compare growth of 10,000 and drawdowns against an optional benchmark
- Review standalone holding metrics, ending weights, and target-weight risk contributions
- Set a maximum holding weight and optional tighter per-ticker caps
- Compare current, minimum-volatility, and maximum-Sharpe allocations on a constrained efficient frontier
- Inspect estimation-window sensitivity and download the allocation comparison
- Assess allocations on held-out history and export metrics, risk, daily performance, and validation results

### 🎲 Portfolio simulation
- Simulate the current portfolio directly, or hand off a minimum-volatility or maximum-Sharpe allocation
- Sample correlated historical return blocks with a reproducible random seed
- Model a starting investment, fixed monthly contributions, and rebalancing
- View percentile bands, ending-value distributions, and probabilities of terminal loss or reaching a target
- Optionally view ending values in today's money under an inflation assumption
- Download monthly percentiles and total contributions

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/colinlpaterson/stock-performance-analyzer.git
cd stock-performance-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the App

**Multi-Ticker Comparison:**
1. Navigate to the "Ticker Comparison" page
2. Add 2–5 popular or custom ticker symbols
3. View comparative YTD performance and last year's full performance
4. Use the color-coded charts to identify relative performance

**Historical YTD Analysis:**
1. Navigate to the "Historical YTD" page
2. Enter a stock ticker (e.g., SPY, AAPL, MSFT)
3. Select a start year (e.g., 2004)

**Correlation Analysis:**
1. Navigate to the "Correlation Analysis" page
2. Choose the static matrix or rolling-pair view
3. Select or enter the ticker symbols to analyze
4. Choose the return type, lookback period, and observation frequency
5. Calculate and explore the heatmap or rolling-correlation chart

**Portfolio setup:**
1. Open Portfolio Analysis or Portfolio Simulation.
2. Edit the ticker and weight table; add or delete rows as needed.
3. Set the historical dates and rebalancing schedule. Optional assumptions include
   a benchmark ticker, a constant annual risk-free rate for Sharpe (default 0%, not a live rate),
   and an annual minimum acceptable return (MAR) for Sortino (default 0%).
   MAR is independent of the risk-free rate and is saved with the portfolio and included in exports.
4. Click **Save and check data**, or **Normalize and save** to proportionally rescale
   the weights before saving and checking data.
5. Review the saved allocation, common date range, and data coverage. Both pages
   use this saved portfolio without downloading again when switching pages.

Save before switching pages: unsubmitted edits are not shared. Saved portfolios
last for the browser session; this release does not provide permanent storage.
Editing portfolio inputs immediately hides dependent results and downloads until
the portfolio is saved again. Editing optimization constraints or the training split
clears optimization results and simulations using optimized allocations; simulations
of the current portfolio remain valid. Editing simulation inputs clears outcomes.
These edits do not start downloads or expensive calculations automatically.
The builder checks empty symbols, duplicate holdings, and invalid weights locally
before saving. Symbol availability and historical coverage are checked on save.

Portfolio data uses adjusted-close total returns only. All entered holdings
(including zero-weight optimization candidates) and any benchmark must have a
verified, matching quote currency. Missing tickers, invalid prices, duplicate dates,
and gaps within the common period block data readiness rather than silently changing
the portfolio or filling prices. Holdings should have matching trading calendars.
The requested end date is inclusive and must precede today. A benchmark must cover
every portfolio date and cannot silently shorten the portfolio's history.

On Portfolio Analysis, select **Analyze** or **Optimize** below the saved portfolio.
On Portfolio Simulation, choose the allocation and simulation settings. Data readiness requires at least two
common price observations; volatility, Sharpe, Sortino, and covariance require at least two
daily returns. Short histories are flagged and should be interpreted cautiously.
Saving or reloading clears dependent data and results before loading new inputs.
Taxes, trading costs, and currency conversion are not modeled.

### Historical portfolio methodology

- Start with 10,000 units of the verified quote currency at the first common close.
  Fractional holdings are allowed; adjusted-price ratios represent reinvested total
  returns. There are no external cash flows. Weights drift between rebalances.
- Monthly, quarterly, and annual schedules restore target weights at the last observed
  close before the next calendar period. Trading does not change portfolio value;
  new weights affect only subsequent returns. The final observation is not treated
  as a scheduled rebalance. Buy and hold never rebalances.
- Total return is ending value / starting value − 1. CAGR uses elapsed calendar
  years of 365.25 days. Annualized volatility uses daily sample standard deviation
  (`ddof=1`) × √252.
- Sharpe uses mean daily excess return / sample daily standard deviation × √252.
  Convert the effective annual risk-free assumption to a constant daily rate with
  `(1 + annual rate) ** (1/252) - 1`. CAGR is not used in the Sharpe numerator.
  This conventional annualization does not adjust for serial correlation.
- Sortino uses mean daily return above the daily MAR / daily downside deviation × √252.
  Convert annual MAR to a daily target with `(1 + annual MAR) ** (1/252) - 1`.
  Downside deviation is the square root of mean squared shortfalls across all daily
  returns; days meeting the target contribute zero. The metric help explains the
  definition and displays the saved MAR. Zero downside deviation or fewer than two
  returns produces N/A.
- Drawdown is value / running peak − 1, including the initial value in the peak
  series. Maximum drawdown is the most negative value.
- Standalone holding and benchmark metrics use the same common dates. The benchmark
  is buy-and-hold with distributions reinvested.
- Risk decomposition uses annualized historical **sample covariance** at the entered
  target weights. For covariance Σ and weights w, volatility is √(wᵀΣw), component
  volatility is `w_i (Σw)_i / volatility`, and risk share is `w_i (Σw)_i / (wᵀΣw)`.
  Components sum to target-weight volatility, which can differ from the realized
  volatility of a portfolio whose weights drift. Negative risk shares are retained.
- Undefined statistics display N/A or an empty table cell: for example, Sharpe for
  constant returns and risk shares for zero target-weight volatility. Short periods
  can produce extreme annualized results; numerically overflowing CAGR is undefined.

Methodology references: [Sharpe (1994)](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)
and [MOSEK risk decomposition](https://docs.mosek.com/portfolio-cookbook/risk_parity.html).

### Constrained portfolio optimization

After saving a portfolio, choose **Optimize** under Portfolio view. Set the global
maximum weight and optionally enter tighter individual caps; a blank cell inherits
the global cap and 0% excludes a holding from optimized allocations. Click
**Optimize portfolio** to apply the constraints. The current allocation is preserved,
even if it exceeds a proposed cap, and is flagged in the comparison.

- At least 60 aligned daily returns are required. Expected annual arithmetic returns
  are daily means × 252, distinct from historical CAGR.
- Risk estimates use annualized Ledoit–Wolf covariance shrinkage from scikit-learn.
  Historical analysis continues to use sample covariance for its risk decomposition.
- All allocations are long-only and fully invested. Individual caps cannot exceed
  the global cap; their sum must allow at least 100% allocation. The optimizer does
  not relax constraints silently.
- Minimum volatility and frontier points solve convex variance-minimization problems
  with SciPy SLSQP. The efficient branch starts at minimum volatility and ends at the
  maximum feasible estimated return. It is not a random portfolio cloud.
- Maximum Sharpe uses a convex transformation: minimize `yᵀΣy` with `eᵀy = 1` and
  `0 <= y_i <= cap_i * sum(y)` for scaled excess returns `e`, then recover
  `w = y / sum(y)`. This requires a feasible positive excess return.
- To match historical Sharpe's daily-rate convention, convert the saved effective
  annual risk-free rate to `(1 + rate) ** (1/252) - 1`, then multiply by 252 when
  comparing with annual arithmetic expected returns.
- Every solution must converge, be finite, sum to 100%, and respect caps within a
  numerical weight tolerance of `1e-7` (0.00001 percentage points). Frontier solutions
  must also satisfy their return targets. Failed minimum-volatility solves block the
  run; failed maximum-Sharpe solves leave it unavailable. Failed frontier targets are
  displayed as gaps, not interpolated valid solutions.
- If no feasible allocation has a positive estimated excess return, maximum Sharpe
  is explicitly unavailable. Zero-volatility positive-excess solutions have an
  undefined ratio and are not reported as a finite maximum-Sharpe allocation.
- Caps that permit only one allocation, or equal return estimates on the efficient
  branch, produce a single-point frontier. Unavailable comparison cells are empty.
- With at least 120 observations, the sensitivity panel repeats the estimates and
  optimization using the most recent half of the history under identical caps.
  It reports weight changes; this is not an out-of-sample performance test.
- The separate **Held-out historical evaluation** fits estimates and allocations
  on the earlier 70% of return observations by default (adjustable from 50–80%).
  It requires at least 60 training and 30 test returns. The last training close is
  the initial investment date; only subsequent returns enter the test performance.
  Training target weights stay fixed, with the saved rebalancing schedule and no
  refitting. Current and any benchmark are evaluated over the same test dates.
  Training maximum Sharpe can be unavailable independently of the full-history solve.
  Choosing tickers or tuning constraints after seeing the test period weakens its
  independence; one split does not establish future superiority.

Metric hover explanations describe interpretation and units. CSV downloads preserve
unrounded numeric results, explicit percentage/percentage-point units, currency,
MM/DD/YYYY dates, and portfolio assumptions. Undefined results remain blank.
Historical exports include metrics, holdings/risk, daily values and weights, and asset
returns. Optimization exports include allocations, estimates, frontier, sensitivity,
and held-out metrics/growth/training weights. Simulation exports include monthly
quantiles and individual ending outcomes with allocation, seed, and run settings.

Estimates and optimized allocations use the selected historical sample. They are
not backtest performance or forecasts guaranteed to occur. Caps address concentration,
not estimation error. Use **Simulate current**, **Simulate minimum volatility**, or
**Simulate maximum sharpe** to carry an available allocation to simulation. This
does not overwrite the saved current portfolio.

Implementation references: [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
and [Ledoit–Wolf covariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

### Portfolio simulation methodology

Save a portfolio with at least 60 common daily returns on either portfolio page.
Open Portfolio Simulation directly or use a simulation handoff button from analysis
or optimization. Select an available allocation, enter the starting investment,
horizon, monthly contribution, optional ending-value target, and rebalancing schedule,
then click **Run simulation**. Unavailable optimized allocations cannot be selected.

- **Correlated circular-block bootstrap:** draw historical starting dates uniformly
  and sample consecutive daily return rows in blocks (default 21 days), wrapping
  from the end of the history to its beginning. Every holding uses the same sampled
  dates within a path. Blocks preserve observed joint returns and within-block time
  dependence; block boundaries introduce new transitions. The block length affects
  results and is adjustable up to 252 days or the available history, whichever is less.
- **Modeled calendar:** a month is exactly 21 trading days and a year is 12 months.
  The horizontal axis is years from start, not future calendar dates.
- **Cash-flow accounting:** start at target weights, apply daily total returns,
  then add each contribution at month-end in target proportions. New cash earns
  returns starting the next month. Buy and hold adds cash without selling existing
  holdings to restore targets. Contributions are fixed nominal amounts, not returns.
- **Rebalancing:** restore targets after month-end returns and contributions every
  1, 3, or 12 modeled months, or never for buy and hold. Trades are self-financing.
  Optimization caps apply to target allocations, not continuously to drifted weights.
- **Outcomes:** show monthly 10th–90th and 25th–75th percentile bands, monthly medians,
  and a terminal distribution. The median curve is not a single simulated path.
  The 10th percentile is not a worst case. Probabilities are conditional on the data
  and model, not guarantees of future results.
- **Loss and goal probabilities:** terminal nominal value below initial investment
  plus all contributions counts as a loss; terminal nominal value at or above the
  optional target counts as success. They do not describe losses during the path.
  Numerical equality is treated as equality to avoid floating-point false losses.
- **Inflation:** an optional effective annual assumption adds a supplementary terminal
  value column in today's money, dividing by `(1 + inflation) ** years`. Charts,
  targets, cash flows, and probabilities remain nominal; deposits do not escalate.
- **Optimized allocations:** the selected weights are retained, but simulation uses
  raw historical return blocks, not synthetic draws from the optimizer's shrunk
  covariance. It is not an independent validation of the optimized portfolio.
- **Reproducibility:** the same data, weights, settings, seed, and numerical environment
  reproduce the paths. The local random generator does not alter global random state.
  Changing historical data can change results even with the same seed.
- **Scope:** fractional holdings and reinvested distributions, with no withdrawals,
  taxes, trading costs, or currency conversion. Unseen market regimes and parameter
  uncertainty are not modeled. The saved risk-free rate is not used to generate returns.
- **Resources:** default 5,000 paths over 10 years; UI limits of 10,000 paths and 40
  years, with a combined work limit that may require fewer paths for large portfolios.
  Only monthly wealth and final holdings are retained, not a full daily asset cube.

Changing any simulation setting or allocation clears its previous result immediately.
Portfolio edits also invalidate simulation results. Optimization-setting edits only
invalidate simulations using optimized allocations and reset that selection once.
A failed run never leaves old outcome charts displayed.

Reference: [time-series block bootstraps](https://bashtage.github.io/arch/doc/bootstrap/timeseries-bootstraps.html).

## Data Source

This application uses **yfinance** to fetch real-time and historical stock data from Yahoo Finance.
Price return uses split-adjusted close and excludes distributions. Total return uses Yahoo
Finance adjusted close as a proxy for reinvested dividends and other distributions; it is
not tax-aware and may differ from an investor's realized return.

## Project Structure

```
stock-performance-analyzer/
├── app.py                      # Main Streamlit application
├── pages/
│   ├── 1_📈_Historical_YTD.py   # Historical YTD comparison page
│   ├── 2_📊_Ticker_Comparison.py # Multi-ticker comparison page
│   └── 3_🔗_Correlation_Analysis.py # Static and rolling correlation analysis
├── utils/
│   ├── __init__.py
│   ├── correlation_analysis.py # Return-matrix and correlation calculations
│   ├── data_fetcher.py         # Data retrieval functions
│   └── chart_builder.py        # Plotting and visualization functions
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## Technologies Used

- **Streamlit**: Web application framework
- **yfinance**: Stock data retrieval
- **Plotly**: Interactive charting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

## Tests

Install development dependencies and run the offline regression suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic market data and mocked Yahoo Finance downloads, including
Streamlit page tests. They cover price and total returns, static and rolling correlation,
common-date alignment, completed-period filtering, and matching comparison endpoints.
Portfolio tests also cover weight validation and normalization, adjusted-price integrity,
currency checks, benchmark coverage, cross-page persistence, and stale-data invalidation.
Historical analysis tests use independently calculable portfolios to check wealth,
weight drift, period-end rebalancing, sample statistics, drawdowns, benchmark alignment,
negative risk contributions, and undefined statistics.
Optimization tests verify independently known two-asset optima and frontier points,
constraint compliance, infeasible caps, single-allocation cases, estimation conventions,
solver failures and invalid outputs, and unavailable maximum-Sharpe solutions.
Simulation tests cover seed reproducibility, shared block dates, end-of-month cash
flows, rebalancing schedules, loss/target boundaries, handoff, and result invalidation.
GitHub Actions also checks compatibility with pandas 2.0 and current dependencies.

The shared portfolio foundation lives in `utils/portfolio.py` (validated definitions),
`utils/portfolio_data.py` (prices, returns, and coverage), `utils/portfolio_state.py`
(session state and invalidation), and `utils/portfolio_ui.py` (the shared editor).
Calculation modules in subsequent stages can consume these definitions and aligned
inputs independently of Streamlit.
`utils/portfolio_analysis.py` implements historical accounting and statistics;
`utils/portfolio_charts.py` and `utils/portfolio_analysis_ui.py` display the results.
`utils/portfolio_optimization.py` implements estimation and constrained solvers;
`utils/portfolio_optimization_ui.py` provides constraints and allocation comparisons.
`utils/portfolio_simulation.py` implements block sampling and cash-flow accounting;
`utils/portfolio_simulation_ui.py` provides allocation handoff and outcome displays.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

Colin Paterson - [GitHub Profile](https://github.com/colinlpaterson)

## Acknowledgments

- Data provided by Yahoo Finance via yfinance
- Built with Streamlit

---

**Note**: This application is for educational and informational purposes only. It is not financial advice. Always consult with a qualified financial advisor before making investment decisions.
