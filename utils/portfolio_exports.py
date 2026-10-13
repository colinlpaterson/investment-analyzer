"""CSV tables with explicit units, dates, and assumptions; undefined values stay blank."""

import pandas as pd


METRIC_HELP = {
    "Total return": "Total gain or loss over the displayed period, including reinvested distributions; not annualized.",
    "CAGR": "The constant annual growth rate that would turn the starting value into the ending value. Actual yearly returns vary.",
    "Volatility": "How much daily returns vary, annualized using sample standard deviation × √252. Risk here means variability, not capital loss.",
    "Max drawdown": "The largest percentage decline from a previous portfolio peak in this period. A negative value records a loss from that peak.",
    "Sharpe ratio": "Annualized average daily return above the assumed risk-free rate per unit of volatility. Undefined when volatility is zero.",
    "Target weight": "The entered fraction invested in this holding at the start and after rebalancing.",
    "Ending weight": "The holding's share of portfolio value at the final close, after market movements.",
    "Risk contribution": "Share of target-weight portfolio volatility attributed to this holding, not its share of capital loss. Negative shares indicate diversification.",
    "Volatility contribution": "Contribution to annualized portfolio volatility in percentage points. A 25% risk share at 12% volatility contributes 3 pp.",
}


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    for column in frame:
        if (column in METRIC_HELP and column != "Sharpe ratio") or column in ("Estimated return", "Estimated volatility"):
            table[column] *= 100
            table = table.rename(columns={column: column + (" (pp)" if column == "Volatility contribution" else " (%)")})
    return table


def portfolio_metadata(data) -> dict:
    return {
        "Currency": data.currency,
        "Requested start": data.definition.start_date.strftime("%m/%d/%Y"),
        "Requested end": data.definition.end_date.strftime("%m/%d/%Y"),
        "History start": data.prices.index[0].strftime("%m/%d/%Y"),
        "History end": data.prices.index[-1].strftime("%m/%d/%Y"),
        "Rebalancing": data.definition.rebalancing,
        "Annual risk-free rate (%)": data.definition.risk_free_rate * 100,
        "Current weights (%)": "; ".join(f"{h.ticker}: {h.weight * 100:.12g}" for h in data.definition.holdings),
        "Return basis": "Adjusted close; reinvested distributions; no costs or taxes",
    }


def csv_bytes(frame: pd.DataFrame, metadata: dict) -> bytes:
    table = frame.copy()
    if table.index.name is None:
        table.index.name = "Date" if isinstance(table.index, pd.DatetimeIndex) else "Item"
    for name, value in metadata.items():
        table[f"Context: {name}"] = value
    return table.to_csv(date_format="%m/%d/%Y", na_rep="").encode("utf-8")
