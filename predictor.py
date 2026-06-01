import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

# Silencia logs verbosos do Prophet e cmdstanpy
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def prever_arima(series, passos=6):
    """
    Realiza a previsão utilizando o modelo ARIMA(1, 1, 1).
    Garante que a série possui uma frequência definida antes do ajuste.
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    # Faz uma cópia da série para evitar efeitos colaterais
    s = series.copy()
    
    # Garante que o índice temporal possui frequência definida
    if s.index.freq is None:
        freq = pd.infer_freq(s.index)
        if freq is None:
            if len(s) > 1:
                diffs = s.index.to_series().diff().dropna()
                time_delta = diffs.median()
                freq = pd.tseries.frequencies.to_offset(time_delta)
            else:
                freq = 'D'
        s = s.asfreq(freq, method='ffill')
        
    # Ajusta o modelo ARIMA(1, 1, 1) com configurações robustas
    model = ARIMA(s, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    
    # Realiza previsão para os passos futuros
    forecast = model_fit.forecast(steps=passos)
    return forecast

def prever_sarima(series, passos=6):
    """
    Realiza a previsão utilizando o modelo SARIMA.
    Ajusta a ordem sazonal de acordo com a frequência da série.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    s = series.copy()
    
    # Garante frequência definida no índice
    if s.index.freq is None:
        freq = pd.infer_freq(s.index)
        if freq is None:
            if len(s) > 1:
                diffs = s.index.to_series().diff().dropna()
                time_delta = diffs.median()
                freq = pd.tseries.frequencies.to_offset(time_delta)
            else:
                freq = 'D'
        s = s.asfreq(freq, method='ffill')
        
    freq_str = str(s.index.freqstr or s.index.freq).upper()
    
    # Identifica a ordem sazonal (sazonalidade mensal = 12, trimestral = 4)
    if 'M' in freq_str:
        seasonal_order = (1, 1, 1, 12)
    elif 'Q' in freq_str:
        seasonal_order = (1, 1, 1, 4)
    else:
        seasonal_order = (0, 0, 0, 0)
        
    # Ajusta o modelo SARIMAX
    model = SARIMAX(
        s, 
        order=(1, 1, 1), 
        seasonal_order=seasonal_order, 
        enforce_stationarity=False, 
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # Realiza previsão para os passos futuros
    forecast = model_fit.forecast(steps=passos)
    return forecast

def prever_series(df, algoritmo="Regressão Linear", passos=6):
    """
    Aplica o algoritmo selecionado (Regressão Linear, XGBoost, Prophet, ARIMA ou SARIMA)
    para projetar os próximos passos de cada série ativa no DataFrame.
    Retorna um DataFrame com as projeções.
    """
    if df is None or df.empty or len(df.columns) < 1:
        raise ValueError("O DataFrame fornecido está vazio.")
        
    projecoes = {}
    datas_futuras = None
    
    # Determina as datas futuras antes do loop (comum para todos os modelos)
    last_date = df.index[-1]
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        datas_futuras = pd.date_range(start=last_date, periods=passos + 1, freq=inferred_freq)[1:]
    else:
        if len(df) > 1:
            diffs = df.index.to_series().diff().dropna()
            time_delta = diffs.median()
        else:
            time_delta = pd.Timedelta(days=30)
        datas_futuras = pd.DatetimeIndex([last_date + (i * time_delta) for i in range(1, passos + 1)])
        
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 4:
            raise ValueError(f"Dados insuficientes na série '{col}' para modelagem (mínimo 4 pontos).")
            
        if algoritmo == "ARIMA":
            pred_series = prever_arima(series, passos=passos)
            projecoes[col] = [float(x) for x in pred_series.values]
        elif algoritmo == "SARIMA":
            pred_series = prever_sarima(series, passos=passos)
            projecoes[col] = [float(x) for x in pred_series.values]
        elif algoritmo == "Prophet":
            from prophet import Prophet
            
            # Prepara o DataFrame no formato exigido pelo Prophet (ds, y)
            df_prophet = pd.DataFrame({
                'ds': series.index.tz_localize(None) if hasattr(series.index, 'tz_localize') else series.index,
                'y': series.values
            })
            
            # Ajusta o modelo Prophet
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)
            
            # Cria DataFrame com as datas futuras para previsão
            future = pd.DataFrame({
                'ds': datas_futuras.tz_localize(None) if hasattr(datas_futuras, 'tz_localize') else datas_futuras
            })
            
            # Realiza previsões
            forecast = model.predict(future)
            projecoes[col] = forecast['yhat'].values.tolist()
            
        else:
            # Modelos baseados em Lags (Regressão Linear e XGBoost)
            df_lags = pd.DataFrame(index=series.index)
            df_lags['y'] = series
            df_lags['y_lag1'] = series.shift(1)
            df_lags['y_lag2'] = series.shift(2)
            df_lags['y_lag3'] = series.shift(3)
            df_lags = df_lags.dropna()
            
            if df_lags.empty:
                raise ValueError(f"Dados insuficientes na série '{col}' após a criação dos lags.")
                
            X = df_lags[['y_lag1', 'y_lag2', 'y_lag3']].values
            y = df_lags['y'].values
            
            if algoritmo == "XGBoost":
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
            else:
                model = LinearRegression()
                
            model.fit(X, y)
            
            # Previsão recursiva utilizando os últimos 3 valores da série
            last_values = list(series.iloc[-3:])
            predictions = []
            for _ in range(passos):
                X_next = [[last_values[-1], last_values[-2], last_values[-3]]]
                pred = model.predict(X_next)[0]
                pred = float(pred)
                predictions.append(pred)
                last_values.append(pred)
                
            projecoes[col] = predictions
            
    df_proj = pd.DataFrame(projecoes, index=datas_futuras)
    df_proj.index.name = df.index.name or 'DATE'
    return df_proj
