import pandas as pd
import ipeadatapy as idp
from statsmodels.tsa.arima.model import ARIMA

def test_arima(code):
    print(f"\nTesting ARIMA for code: {code}")
    try:
        df = idp.timeseries(code)
        if df is None or df.empty:
            print("No data.")
            return
        
        val_cols = [c for c in df.columns if c.startswith("VALUE")]
        series = df[val_cols[0]].dropna()
        
        # Treatment of frequency
        s = series.copy()
        if s.index.freq is None:
            freq = pd.infer_freq(s.index)
            if freq is None:
                diffs = s.index.to_series().diff().dropna()
                time_delta = diffs.median()
                freq = pd.tseries.frequencies.to_offset(time_delta)
            s = s.asfreq(freq, method='ffill')
            
        print(f"Index freq: {s.index.freq}")
        model = ARIMA(s, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit()
        fc = fit.forecast(steps=6)
        print("Forecast succeeded:")
        print(fc)
    except Exception as e:
        print("ARIMA Failed:", e)

test_arima("PRECOS12_IPCA12")
test_arima("DEPIS_POP")
