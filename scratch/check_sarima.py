import pandas as pd
import ipeadatapy as idp
from statsmodels.tsa.statespace.sarimax import SARIMAX

def test_sarima(code):
    print(f"\nTesting SARIMA for code: {code}")
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
            
        freq_str = str(s.index.freqstr or s.index.freq).upper()
        print(f"Freq str: {freq_str}")
        
        if 'M' in freq_str:
            seasonal_order = (1, 1, 1, 12)
        elif 'Q' in freq_str:
            seasonal_order = (1, 1, 1, 4)
        else:
            seasonal_order = (0, 0, 0, 0)
            
        print(f"Seasonal order: {seasonal_order}")
        model = SARIMAX(s, order=(1, 1, 1), seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.forecast(steps=6)
        print("Forecast succeeded:")
        print(fc)
    except Exception as e:
        print("SARIMA Failed:", e)

test_sarima("PRECOS12_IPCA12")
