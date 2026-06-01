import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import ccf

def analisar_relacoes(df):
    """
    Calcula a matriz de correlacao de Pearson, a matriz de covariancia,
    e a Funcao de Correlacao Cruzada (CCF) se houver exatamente duas series.
    """
    if df is None or df.empty or len(df.columns) < 1:
        return pd.DataFrame(), pd.DataFrame(), None
        
    # Matriz de correlacao de Pearson
    corr_matrix = df.corr(method='pearson')
    
    # Matriz de covariancia
    cov_matrix = df.cov()
    
    # CCF para exatamente duas series
    ccf_result = None
    if len(df.columns) == 2:
        col1, col2 = df.columns[0], df.columns[1]
        
        # Remove valores nulos caso existam
        df_clean = df[[col1, col2]].dropna()
        
        if len(df_clean) > 4:
            x = df_clean[col1].values
            y = df_clean[col2].values
            
            # Determina o numero maximo de lags
            n_lags = min(20, len(df_clean) // 2 - 1)
            if n_lags > 0:
                try:
                    # Lags positivos (x_t com y_{t+k})
                    ccf_pos = ccf(x, y)[:n_lags + 1]
                    # Lags negativos (x_t com y_{t-k}) - equivale a ccf(y, x) invertido
                    ccf_neg = ccf(y, x)[1:n_lags + 1]
                    
                    lags = np.arange(-n_lags, n_lags + 1)
                    ccf_vals = np.concatenate([ccf_neg[::-1], ccf_pos])
                    
                    ccf_result = {
                        'lags': lags,
                        'valores': ccf_vals,
                        'nomes': (col1, col2)
                    }
                except Exception:
                    ccf_result = None
                    
    return corr_matrix, cov_matrix, ccf_result

def gerar_insights_texto(matriz_corr):
    """
    Avalia os coeficientes de correlacao entre as series e gera um relatorio
    explicativo traduzindo os coeficientes em insights práticos.
    """
    if matriz_corr is None or matriz_corr.empty:
        return "Sem dados de correlação suficientes para gerar insights."
        
    colunas = matriz_corr.columns
    if len(colunas) < 2:
        return "Selecione pelo menos duas séries para analisar a relação e correlação entre elas."
        
    insights = []
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            col1 = colunas[i]
            col2 = colunas[j]
            r = matriz_corr.loc[col1, col2]
            
            if pd.isna(r):
                continue
                
            abs_r = abs(r)
            
            # Classificacao da intensidade
            if abs_r >= 0.7:
                forca = "Forte"
            elif abs_r >= 0.4:
                forca = "Moderada"
            elif abs_r >= 0.1:
                forca = "Fraca"
            else:
                forca = "Nula ou desprezível"
                
            # Classificacao da direcao
            if r > 0.1:
                direcao = "correlação positiva direta (as duas variáveis tendem a subir ou descer juntas)"
            elif r < -0.1:
                direcao = "correlação negativa inversa (quando uma variável sobe, a outra tende a descer)"
            else:
                direcao = "relação linear direta inexistente"
                
            # Construcao da explicacao
            if forca == "Nula ou desprezível" or abs_r < 0.1:
                insights.append(f"• **{col1}** e **{col2}**: **Sem relação linear aparente** (coeficiente de Pearson de {r:.2f}). Seus movimentos parecem ser estatisticamente independentes um do outro.")
            else:
                insights.append(f"• **{col1}** e **{col2}**: Apresentam uma **{forca.lower()}** {direcao}, com coeficiente de Pearson de **{r:.2f}**.")
                
    if not insights:
        return "Não foram detectados relacionamentos lineares válidos entre as séries selecionadas."
        
    return "\n".join(insights)
