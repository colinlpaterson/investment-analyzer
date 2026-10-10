"""
Stock Performance Analyzer - Main Application
A Streamlit app for analyzing and comparing stock and ETF performance.
"""
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Stock Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("📊 Stock Performance Analyzer")
st.markdown("---")

# Welcome message
st.markdown("""
### Welcome to the Stock Performance Analyzer!

This application provides powerful tools for analyzing stock and ETF performance using year-to-date (YTD) returns.

#### 📈 Available Analysis Tools

**1. Historical YTD Analysis**
- Compare current year performance against multiple historical years
- Identify best and worst performing years
- View historical averages and trends
- Analyze any stock or ETF ticker

**2. Multi-Ticker Comparison**
- Compare YTD performance across multiple tickers
- View current year vs. last year performance
- Analyze relative strength between different assets

**3. Correlation Analysis**
- Compare how closely the periodic returns of 2–10 securities move together
- Choose price or total returns and daily, weekly, or monthly frequency
- Identify the highest and lowest correlations in the selected group

### 🚀 Getting Started

1. Select an analysis tool from the sidebar
2. Configure your parameters
3. Click "Analyze" to generate interactive visualizations
4. Explore the results and download data as needed

### 📊 Data Source

All data is sourced from **Yahoo Finance** using the yfinance library. Data includes:
- Daily split-adjusted and dividend-adjusted closing prices
- Historical data back to the early 2000s (varies by ticker)
- Real-time updates for current year data

### ⚠️ Important Notes

- **Selectable Return Type**: Choose price return or adjusted-close total return
- **Analysis Frequency**: YTD analysis uses month-end data; correlation frequency is selectable
- **Total Return Assumption**: Total return uses Yahoo Finance adjusted close as a proxy for reinvested distributions
- **Not Financial Advice**: This tool is for informational and educational purposes only

### 🔗 Resources

- [Yahoo Finance](https://finance.yahoo.com/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

👈 **Select a tool from the sidebar to begin your analysis**
""")

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Use the pages above to access different analysis tools.")
    
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    **Version:** 1.0.0
    
    **Created by:** Colin Paterson
    
    **GitHub:** [stock-performance-analyzer](https://github.com/colinlpaterson/stock-performance-analyzer)
    """)
    
    st.markdown("---")
    
    st.markdown("### Feedback")
    st.markdown("Have suggestions or found a bug? Please open an issue on GitHub!")

# Footer
st.markdown("---")
st.caption("© 2026 Stock Performance Analyzer | Data provided by Yahoo Finance | For educational purposes only")
