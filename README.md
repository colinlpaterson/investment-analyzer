# Stock Performance Analyzer

A Streamlit web application for analyzing and comparing stock and ETF performance with interactive visualizations.

## Features

### 📈 Historical YTD Performance
- View price return or adjusted-close total return for any stock ticker
- Compare current year against historical years
- Available for any ticker supported by Yahoo Finance

### 📊 Multi-Ticker Comparison
- Compare price or total-return YTD performance across multiple tickers simultaneously
- View last full year's performance (dashed lines) alongside current YTD
- Color-coded visualization for easy comparison
- Available for any ticker supported by Yahoo Finance

### 🔗 Correlation Analysis
- Compare Pearson correlations across 2–10 stocks and ETFs
- Choose price or total returns, lookback period, and return frequency
- View an annotated heatmap and the highest and lowest correlation pairs
- Export the correlation matrix as CSV

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

**Historical YTD Analysis:**
1. Navigate to the "Historical YTD" page
2. Enter a stock ticker (e.g., SPY, AAPL, MSFT)
3. Select a start year (e.g., 2004)

**Multi-Ticker Comparison:**
1. Navigate to the "Ticker Comparison" page
2. Add 2–5 popular or custom ticker symbols
3. View comparative YTD performance and last year's full performance
4. Use the color-coded charts to identify relative performance

**Correlation Analysis:**
1. Navigate to the "Correlation Analysis" page
2. Select or enter 2–10 ticker symbols
3. Choose the return type, lookback period, and observation frequency
4. Calculate and explore the static correlation heatmap

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
│   └── 3_🔗_Correlation_Analysis.py # Static return-correlation heatmap
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
Streamlit page tests. They cover price and total returns, correlation calculations,
common-date alignment, completed-period filtering, and matching comparison endpoints.
GitHub Actions also checks compatibility with pandas 2.0 and current dependencies.

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
