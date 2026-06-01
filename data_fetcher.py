import ipeadatapy
import pandas as pd

def listar_series():
    """
    Lista todas as series disponiveis no IPEADATA, filtrando as series inativas.
    """
    df = ipeadatapy.list_series()
    if df is not None and not df.empty:
        if 'NAME' in df.columns:
            df = df[~df['NAME'].str.contains('inativa', case=False, na=False)]
    return df

def buscar_dados_multiplos(lista_codigos, inicio, fim):
    """
    Busca os dados historicos de multiplas series no intervalo especificado,
    faz o merge das series usando inner join no indice de datas e retorna
    o DataFrame consolidado e os metadados de cada serie.
    """
    dfs = []
    metadados = {}
    
    inicio_dt = pd.to_datetime(inicio)
    fim_dt = pd.to_datetime(fim)
    
    for cod in lista_codigos:
        try:
            # Obtem a serie temporal
            df_temp = ipeadatapy.timeseries(cod)
            if df_temp is not None and not df_temp.empty:
                # Filtra pelo intervalo de datas
                df_temp = df_temp[(df_temp.index >= inicio_dt) & (df_temp.index <= fim_dt)]
                
                if not df_temp.empty:
                    # Encontra a coluna de valor
                    val_cols = [c for c in df_temp.columns if c.startswith("VALUE")]
                    if val_cols:
                        val_col = val_cols[0]
                        # Mantem apenas a coluna de valor renomeada para o codigo da serie
                        df_s = df_temp[[val_col]].rename(columns={val_col: cod})
                        dfs.append(df_s)
                        
                        # Busca os metadados da serie
                        meta_df = ipeadatapy.metadata(cod)
                        if not meta_df.empty:
                            metadados[cod] = meta_df.iloc[0].to_dict()
                        else:
                            metadados[cod] = {"NAME": cod, "UNIT": "N/A", "COMMENT": ""}
        except Exception:
            # Continua para as proximas series caso ocorra erro em uma delas
            continue
            
    if not dfs:
        return pd.DataFrame(), {}
        
    # Realiza o merge usando inner join com base na data (index)
    df_consolidado = pd.concat(dfs, axis=1, join='inner')
    
    return df_consolidado, metadados

def buscar_valores(codigo, inicio, fim):
    """
    Busca valores de uma série temporal no período especificado.
    Suporta o mapeamento de códigos antigos ou incorretos para códigos ativos no IPEADATA.
    """
    # Mapeamento robusto para evitar quebras por códigos inativos ou renomeados
    map_codigos = {
        "PRECO12_IPCA12": "PRECOS12_IPCA12",
        "POPU_POPU": "DEPIS_POP"
    }
    cod_real = map_codigos.get(codigo, codigo)
    
    try:
        df = ipeadatapy.timeseries(cod_real)
        if df is not None and not df.empty:
            inicio_dt = pd.to_datetime(inicio)
            fim_dt = pd.to_datetime(fim)
            df = df[(df.index >= inicio_dt) & (df.index <= fim_dt)]
            return df
    except Exception as e:
        import sys
        print(f"Erro ao buscar valores para a série {codigo} (tentativa com {cod_real}): {e}", file=sys.stderr)
    return pd.DataFrame()
