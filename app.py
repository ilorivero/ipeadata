import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np
from sklearn.linear_model import LinearRegression

import data_fetcher
import predictor
import stats_analyzer

# Configuração da página do Streamlit
st.set_page_config(
    page_title="IPEADATA Multi-Series Predictor & Interpreter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada para um design premium
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        .title-container {
            padding: 1.5rem 0rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .title-text {
            font-family: 'Inter', sans-serif;
            color: #0f172a;
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .subtitle-text {
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# Função auxiliar para limpar e linkar URLs em metadados
def processar_comentarios(texto):
    if not isinstance(texto, str):
        return texto
    texto = re.sub(
        r'href=["\']\.\./([^"\']+)["\']', 
        r'href="http://www.ipeadata.gov.br/\1"', 
        texto
    )
    texto = re.sub(
        r'href=["\']/(?!/)([^"\']+)["\']', 
        r'href="http://www.ipeadata.gov.br/\1"', 
        texto
    )
    def adicionar_target(match):
        tag = match.group(0)
        if 'target=' not in tag:
            return tag.replace('>', ' target="_blank">')
        return tag
    texto = re.sub(r'<a\s+[^>]+>', adicionar_target, texto)
    
    partes = re.split(r'(<[^>]+>)', texto)
    for i in range(len(partes)):
        if not (partes[i].startswith('<') and partes[i].endswith('>')):
            def replace_url(match):
                url = match.group(1)
                pontuacao_final = ""
                while url and url[-1] in ['.', ',', ';', ':', '?', '!', ')', ']']:
                    pontuacao_final = url[-1] + pontuacao_final
                    url = url[:-1]
                return f'<a href="{url}" target="_blank">{url}</a>{pontuacao_final}'
                
            partes[i] = re.sub(r'\b(https?://[^\s<>\'"]+)', replace_url, partes[i])
            
    return "".join(partes)

@st.cache_data(show_spinner="Carregando dados didáticos...")
def obter_dados_didaticos():
    try:
        df_ipca = data_fetcher.buscar_valores("PRECO12_IPCA12", "2000-01-01", "2024-12-31")
    except Exception:
        df_ipca = pd.DataFrame()
    try:
        df_pop = data_fetcher.buscar_valores("POPU_POPU", "2000-01-01", "2024-12-31")
    except Exception:
        df_pop = pd.DataFrame()
    return df_ipca, df_pop

# Cache para obter a lista de séries
@st.cache_data(show_spinner="Carregando lista de séries do IPEADATA...")
def obter_todas_series():
    try:
        return data_fetcher.listar_series()
    except Exception:
        fallback_data = {
            'CODE': ['GM366_ERC366', 'PRECOS12_IPCA12', 'IGP12_IGPM12'],
            'NAME': [
                'Taxa de câmbio - R$ / US$ - comercial - venda - média',
                'Inflação - IPCA - geral - índice (dez. 1993 = 100)',
                'Inflação - IGP-M - geral - índice (ago. 1994 = 100)'
            ]
        }
        return pd.DataFrame(fallback_data)

# Título da aplicação
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">📊 IPEADATA Multi-Series Hub</h1>
        <p class="subtitle-text">Explore múltiplas séries temporais, analise correlações cruzadas e realize previsões com Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

# Inicializa os estados na sessão
if "selected_codes" not in st.session_state:
    st.session_state.selected_codes = ['PRECOS12_IPCA12', 'IGP12_IGPM12']
if "multi_previsoes" not in st.session_state:
    st.session_state.multi_previsoes = None

# Carrega todas as séries disponíveis
df_todas_series = obter_todas_series()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.markdown("### Filtros Globais")

# Configuração de datas globais
hoje = datetime.date.today()
ano_atras = hoje - datetime.timedelta(days=365)
data_inicio = st.sidebar.date_input("Data de Início", ano_atras)
data_fim = st.sidebar.date_input("Data de Fim", hoje)

if data_inicio > data_fim:
    st.sidebar.error("A data de início deve ser anterior à data de fim.")
    st.stop()

# Filtro/busca por palavra-chave para o multiselect
termo_busca = st.sidebar.text_input("Filtrar opções de busca (Ex: Inflação, Câmbio)", "")

if termo_busca:
    df_filtrado = df_todas_series[
        df_todas_series['CODE'].str.contains(termo_busca, case=False, na=False) |
        df_todas_series['NAME'].str.contains(termo_busca, case=False, na=False)
    ]
else:
    df_filtrado = df_todas_series.head(100)

# Garante que os códigos já selecionados permaneçam nas opções do selectbox
selected_options = df_todas_series[df_todas_series['CODE'].isin(st.session_state.selected_codes)]
df_combined = pd.concat([selected_options, df_filtrado]).drop_duplicates(subset=['CODE'])

opcoes_series = {}
for _, linha in df_combined.head(500).iterrows():
    opcoes_series[linha['CODE']] = linha['NAME']

st.session_state.selected_codes = st.sidebar.multiselect(
    "Selecione as Séries Temporais",
    options=list(opcoes_series.keys()),
    default=st.session_state.selected_codes,
    format_func=lambda x: opcoes_series[x]
)

# Seleção do algoritmo de previsão
st.sidebar.markdown("---")
st.sidebar.markdown("### Configurações de Previsão")
algoritmo_selecionado = st.sidebar.selectbox(
    "Algoritmo de ML",
    options=["Regressão Linear", "XGBoost", "Prophet"],
    index=0,
    help="Escolha o modelo de Machine Learning a ser utilizado nas projeções."
)

# Reseta as previsões caso haja alteração de parâmetros
chave_params = f"{st.session_state.selected_codes}_{data_inicio}_{data_fim}_{algoritmo_selecionado}"
if "params_anteriores" not in st.session_state or st.session_state.params_anteriores != chave_params:
    st.session_state.multi_previsoes = None
    st.session_state.params_anteriores = chave_params

# --- PAINEL PRINCIPAL ---

tab_dashboard, tab_didatico = st.tabs(["📊 Dashboard Interativo", "🎓 Exemplos Didáticos de Classificação"])

with tab_dashboard:
    if not st.session_state.selected_codes:
        st.warning("⚠️ Selecione pelo menos uma série temporal na barra lateral para começar a exploração.")
    else:
        # Busca dados de todas as séries selecionadas
        with st.spinner("Carregando e integrando dados das séries..."):
            df_consolidado, metadados = data_fetcher.buscar_dados_multiplos(
                st.session_state.selected_codes, data_inicio, data_fim
            )

        if df_consolidado.empty:
            st.error("❌ Não foi possível carregar dados no período selecionado ou as séries não possuem intersecção de datas (Inner Join vazio).")
        else:
            # --- ÁREA DE DOWNLOADS ---
            with st.expander("📥 Baixar Dados Ativos"):
                st.markdown("Exporte os dados individuais das séries carregadas (após o join temporal):")
                for cod in st.session_state.selected_codes:
                    if cod in df_consolidado.columns:
                        nome_meta = str(metadados.get(cod, {}).get('NAME') or cod)
                        df_dl = df_consolidado[[cod]].dropna()
                        csv_data = df_dl.to_csv().encode('utf-8')
                        st.download_button(
                            label=f"Download: {nome_meta[:50]}...",
                            data=csv_data,
                            file_name=f"{cod}_dados.csv",
                            mime="text/csv",
                            key=f"btn_dl_{cod}"
                        )

            # --- VISUALIZAÇÃO E TRATAMENTO DE ESCALAS ---
            st.markdown("### 📊 Gráfico de Séries Temporais Integradas")

            palette = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f43f5e']
            n_series = len(df_consolidado.columns)

            if n_series == 1:
                # --- CENÁRIO 1 SÉRIE: Gráfico de linha simples ---
                col = df_consolidado.columns[0]
                nome = str(metadados.get(col, {}).get('NAME') or col)
                unidade = str(metadados.get(col, {}).get('UNIT') or 'N/A')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_consolidado.index,
                    y=df_consolidado[col],
                    mode='lines',
                    name=nome,
                    line=dict(color=palette[0], width=2.5),
                    hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>Valor: %{{y:.4f}} ({unidade})<extra></extra>"
                ))
                
                if st.session_state.multi_previsoes is not None:
                    df_proj = st.session_state.multi_previsoes
                    if col in df_proj.columns:
                        pred_x = [df_consolidado.index[-1]] + list(df_proj.index)
                        pred_y = [df_consolidado[col].iloc[-1]] + list(df_proj[col])
                        fig.add_trace(go.Scatter(
                            x=pred_x,
                            y=pred_y,
                            mode='lines',
                            name=f"Previsão ({algoritmo_selecionado}): {nome}",
                            line=dict(color=palette[0], width=2.5, dash='dash'),
                            hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>Previsão: %{{y:.4f}} ({unidade})<extra></extra>"
                        ))
                        
                fig.update_layout(
                    yaxis_title=f"Valor ({unidade})",
                    template="plotly_white",
                    xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )

            elif n_series == 2:
                # --- CENÁRIO 2 SÉRIES: Dois eixos Y independentes ---
                col1, col2 = df_consolidado.columns[0], df_consolidado.columns[1]
                nome1 = str(metadados.get(col1, {}).get('NAME') or col1)
                unidade1 = str(metadados.get(col1, {}).get('UNIT') or 'N/A')
                nome2 = str(metadados.get(col2, {}).get('NAME') or col2)
                unidade2 = str(metadados.get(col2, {}).get('UNIT') or 'N/A')
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Série 1 (Eixo Y Esquerdo)
                fig.add_trace(go.Scatter(
                    x=df_consolidado.index,
                    y=df_consolidado[col1],
                    mode='lines',
                    name=nome1,
                    line=dict(color=palette[0], width=2),
                    hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{nome1[:30]}...: %{{y:.4f}} ({unidade1})<extra></extra>"
                ), secondary_y=False)
                
                # Série 2 (Eixo Y Direito)
                fig.add_trace(go.Scatter(
                    x=df_consolidado.index,
                    y=df_consolidado[col2],
                    mode='lines',
                    name=nome2,
                    line=dict(color=palette[1], width=2),
                    hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>{nome2[:30]}...: %{{y:.4f}} ({unidade2})<extra></extra>"
                ), secondary_y=True)
                
                # Previsões
                if st.session_state.multi_previsoes is not None:
                    df_proj = st.session_state.multi_previsoes
                    if col1 in df_proj.columns:
                        pred1_x = [df_consolidado.index[-1]] + list(df_proj.index)
                        pred1_y = [df_consolidado[col1].iloc[-1]] + list(df_proj[col1])
                        fig.add_trace(go.Scatter(
                            x=pred1_x,
                            y=pred1_y,
                            mode='lines',
                            name=f"Previsão ({algoritmo_selecionado}): {nome1}",
                            line=dict(color=palette[0], width=2, dash='dash'),
                            hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>Previsão: %{{y:.4f}} ({unidade1})<extra></extra>"
                        ), secondary_y=False)
                        
                    if col2 in df_proj.columns:
                        pred2_x = [df_consolidado.index[-1]] + list(df_proj.index)
                        pred2_y = [df_consolidado[col2].iloc[-1]] + list(df_proj[col2])
                        fig.add_trace(go.Scatter(
                            x=pred2_x,
                            y=pred2_y,
                            mode='lines',
                            name=f"Previsão ({algoritmo_selecionado}): {nome2}",
                            line=dict(color=palette[1], width=2, dash='dash'),
                            hovertemplate=f"<b>%{{x|%d/%m/%Y}}</b><br>Previsão: %{{y:.4f}} ({unidade2})<extra></extra>"
                        ), secondary_y=True)
                        
                fig.update_layout(
                    template="plotly_white",
                    yaxis_title=f"{nome1[:30]}... ({unidade1})",
                    yaxis2_title=f"{nome2[:30]}... ({unidade2})",
                    xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )

            else:
                # --- CENÁRIO 3 OU MAIS SÉRIES: Checkbox opcional de normalização ---
                normalizar = st.checkbox("Normalizar escalas (Min-Max)", value=False)
                
                fig = go.Figure()
                df_plot = df_consolidado.copy()
                min_max_info = {}
                
                if normalizar:
                    for col in df_plot.columns:
                        min_v = df_plot[col].min()
                        max_v = df_plot[col].max()
                        min_max_info[col] = (min_v, max_v)
                        if max_v != min_v:
                            df_plot[col] = (df_plot[col] - min_v) / (max_v - min_v)
                        else:
                            df_plot[col] = 0.5
                            
                for idx, col in enumerate(df_consolidado.columns):
                    cor = palette[idx % len(palette)]
                    nome = str(metadados.get(col, {}).get('NAME') or col)
                    unidade = str(metadados.get(col, {}).get('UNIT') or 'N/A')
                    valores_originais = df_consolidado[col].values
                    hover_texts = [f"Valor Original: {v:.4f} ({unidade})" for v in valores_originais]
                    
                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot[col],
                        mode='lines',
                        name=nome,
                        line=dict(color=cor, width=2),
                        text=hover_texts,
                        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{text}<br>Valor no Gráfico: %{y:.4f}<extra></extra>"
                    ))
                    
                if st.session_state.multi_previsoes is not None:
                    df_proj = st.session_state.multi_previsoes
                    df_proj_plot = df_proj.copy()
                    
                    for idx, col in enumerate(df_consolidado.columns):
                        cor = palette[idx % len(palette)]
                        nome = str(metadados.get(col, {}).get('NAME') or col)
                        unidade = str(metadados.get(col, {}).get('UNIT') or 'N/A')
                        
                        ultimo_x = [df_plot.index[-1]]
                        ultimo_y = [df_plot[col].iloc[-1]]
                        
                        if normalizar and col in min_max_info:
                            min_v, max_v = min_max_info[col]
                            if max_v != min_v:
                                df_proj_plot[col] = (df_proj[col] - min_v) / (max_v - min_v)
                            else:
                                df_proj_plot[col] = 0.5
                                
                        pred_x = ultimo_x + list(df_proj_plot[col].index)
                        pred_y = ultimo_y + list(df_proj_plot[col].values)
                        
                        hover_texts_pred = [f"Previsão Original: {v:.4f} ({unidade})" for v in [df_consolidado[col].iloc[-1]] + list(df_proj[col].values)]
                        
                        fig.add_trace(go.Scatter(
                            x=pred_x,
                            y=pred_y,
                            mode='lines',
                            name=f"Previsão ({algoritmo_selecionado}): {nome}",
                            line=dict(color=cor, width=2, dash='dash'),
                            text=hover_texts_pred,
                            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{text}<br>Projeção no Gráfico: %{y:.4f}<extra></extra>"
                        ))
                        
                fig.update_layout(
                    template="plotly_white",
                    yaxis_title="Escala Normalizada (0-1)" if normalizar else "Valores Históricos",
                    xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )

            # Renderiza o gráfico
            fig.update_layout(
                xaxis_title="Data",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            col_btn, col_empty = st.columns([1.5, 4])
            with col_btn:
                if st.button("Executar Previsão de ML", use_container_width=True):
                    try:
                        with st.spinner(f"Projetando séries com {algoritmo_selecionado}..."):
                            df_proj = predictor.prever_series(df_consolidado, algoritmo=algoritmo_selecionado, passos=6)
                            st.session_state.multi_previsoes = df_proj
                            st.toast("Previsões calculadas!", icon="🤖")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao calcular as previsões: {e}")

            # Tabela das projeções futuras calculadas
            if st.session_state.multi_previsoes is not None:
                st.markdown("#### 🔮 Projeções Futuras (Dados Originais)")
                df_tabela = st.session_state.multi_previsoes.copy()
                df_tabela.index = df_tabela.index.strftime('%d/%m/%Y')
                df_tabela.index.name = "Data Projetada"
                st.dataframe(df_tabela, use_container_width=True)

            # --- PAINEL ESTATÍSTICO E INTERPRETADOR ---
            st.markdown("---")
            st.markdown("### 📈 Painel Estatístico & Interpretador")

            corr_df, cov_df, ccf_res = stats_analyzer.analisar_relacoes(df_consolidado)

            col_corr, col_cov = st.columns(2)
            with col_corr:
                st.markdown("#### Matriz de Correlação de Pearson")
                if not corr_df.empty:
                    nomes_colunas = {col: str(metadados.get(col, {}).get('NAME') or col)[:40] + "..." for col in corr_df.columns}
                    st.dataframe(corr_df.rename(columns=nomes_colunas, index=nomes_colunas), use_container_width=True)
                else:
                    st.info("Matriz indisponível.")

            with col_cov:
                st.markdown("#### Matriz de Covariância")
                if not cov_df.empty:
                    nomes_colunas = {col: str(metadados.get(col, {}).get('NAME') or col)[:40] + "..." for col in cov_df.columns}
                    st.dataframe(cov_df.rename(columns=nomes_colunas, index=nomes_colunas), use_container_width=True)
                else:
                    st.info("Matriz indisponível.")

            if not corr_df.empty and len(corr_df.columns) >= 2:
                st.markdown("#### 💡 Insights Interpretativos")
                nomes_colunas_completos = {col: str(metadados.get(col, {}).get('NAME') or col) for col in corr_df.columns}
                corr_df_nomes = corr_df.rename(columns=nomes_colunas_completos, index=nomes_colunas_completos)
                insights_texto = stats_analyzer.gerar_insights_texto(corr_df_nomes)
                st.info(insights_texto)

            if ccf_res is not None:
                st.markdown("#### Função de Correlação Cruzada (CCF)")
                
                nome_x = str(metadados.get(ccf_res['nomes'][0], {}).get('NAME') or ccf_res['nomes'][0])
                nome_y = str(metadados.get(ccf_res['nomes'][1], {}).get('NAME') or ccf_res['nomes'][1])
                
                fig_ccf = go.Figure()
                fig_ccf.add_trace(go.Bar(
                    x=ccf_res['lags'],
                    y=ccf_res['valores'],
                    marker_color='#3b82f6',
                    name='CCF',
                    hovertemplate="Lag: %{x}<br>Correlação: %{y:.4f}<extra></extra>"
                ))
                
                limite_ic = 2.0 / (len(df_consolidado) ** 0.5)
                fig_ccf.add_hline(y=limite_ic, line_dash="dash", line_color="#ef4444", 
                                  annotation_text=f"+IC 95% ({limite_ic:.3f})", annotation_position="top left")
                fig_ccf.add_hline(y=-limite_ic, line_dash="dash", line_color="#ef4444", 
                                  annotation_text=f"-IC 95% ({-limite_ic:.3f})", annotation_position="bottom left")
                
                fig_ccf.update_layout(
                    title=f"Correlação Cruzada entre:<br>X: {nome_x[:60]}... vs Y: {nome_y[:60]}...",
                    xaxis_title="Lags (Lags negativos: X antecede Y | Lags positivos: Y antecede X)",
                    yaxis_title="Coeficiente de Correlação",
                    template="plotly_white",
                    yaxis=dict(range=[-1.05, 1.05], showgrid=True, gridcolor='#f1f5f9'),
                    xaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )
                st.plotly_chart(fig_ccf, use_container_width=True)

            # --- METADADOS INFORMATIVOS ---
            st.markdown("---")
            st.markdown("### 📋 Informações das Séries Ativas")
            for cod in st.session_state.selected_codes:
                if cod in metadados:
                    meta = metadados[cod]
                    nome = str(meta.get('NAME') or cod)
                    unidade = str(meta.get('UNIT') or 'N/A')
                    comentario = meta.get('COMMENT', 'Sem comentários adicionais.')
                    fonte = meta.get('SOURCE', 'N/A')
                    freq = meta.get('FREQUENCY', 'N/A')
                    status = meta.get('SERIES STATUS', 'Ativo')
                    
                    with st.expander(f"ℹ️ {nome} ({cod})"):
                        comentario_processado = processar_comentarios(comentario)
                        info_html = f"""
                        <div style="font-size: 0.95rem; line-height: 1.6; color: var(--text-color);">
                            <b>Código:</b> <code>{cod}</code> | <b>Unidade:</b> {unidade} | <b>Periodicidade Original:</b> {freq} | <b>Status:</b> {status}<br>
                            <b>Fonte:</b> {fonte}<br><br>
                            <b>Comentários:</b><br>{comentario_processado}
                        </div>
                        """
                        st.markdown(info_html, unsafe_allow_html=True)

with tab_didatico:
    st.markdown("### 🎓 Exemplos Didáticos de Classificação de Séries Temporais")
    st.markdown("Explore conceitos fundamentais da teoria de séries temporais comparando o **IPCA** e a **População Residente**.")
    
    df_ipca_edu, df_pop_edu = obter_dados_didaticos()
    
    if df_ipca_edu.empty or df_pop_edu.empty:
        st.error("Erro ao carregar dados didáticos da API do Ipeadata.")
    else:
        col_ipca = [c for c in df_ipca_edu.columns if c.startswith("VALUE")][0]
        col_pop = [c for c in df_pop_edu.columns if c.startswith("VALUE")][0]
        
        tab_concept1, tab_concept2, tab_concept3, tab_concept4 = st.tabs([
            "1. Univariada vs Multivariada",
            "2. Discreta vs Contínua",
            "3. Estacionária vs Não Estacionária",
            "4. Determinística vs Estocástica"
        ])
        
        with tab_concept1:
            st.markdown("#### Univariada vs Multivariada")
            st.markdown("Séries **univariadas** registram a evolução de uma única variável ao longo do tempo. Séries **multivariadas** acompanham múltiplos fenômenos simultaneamente para analisar covariâncias e causalidades.")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_univar = st.button("Plota apenas IPCA", key="btn_uni")
            with col_btn2:
                btn_multivar = st.button("Plota IPCA + População com eixos Y duplos", key="btn_multi")
                
            if "edu_univar_state" not in st.session_state:
                st.session_state.edu_univar_state = "univariada"
                
            if btn_univar:
                st.session_state.edu_univar_state = "univariada"
            elif btn_multivar:
                st.session_state.edu_univar_state = "multivariada"
                
            if st.session_state.edu_univar_state == "univariada":
                fig_uni = go.Figure()
                fig_uni.add_trace(go.Scatter(
                    x=df_ipca_edu.index, y=df_ipca_edu[col_ipca],
                    mode='lines', name='IPCA', line=dict(color='#2563eb', width=2.5)
                ))
                fig_uni.update_layout(title="IPCA - Série Univariada", yaxis_title="Índice IPCA", template="plotly_white")
                st.plotly_chart(fig_uni, use_container_width=True)
                st.caption(r"**Conceito Matemático (Univariada):** Representa o comportamento isolado de uma única variável. Na modelagem univariada clássica, assume-se que os valores futuros dependem apenas de termos históricos anteriores e de ruídos passados da própria variável: $Y_t = c + \sum_{i=1}^p \phi_i Y_{t-i} + \epsilon_t$.")
            else:
                fig_mul = make_subplots(specs=[[{"secondary_y": True}]])
                fig_mul.add_trace(go.Scatter(
                    x=df_ipca_edu.index, y=df_ipca_edu[col_ipca],
                    mode='lines', name='IPCA', line=dict(color='#2563eb', width=2)
                ), secondary_y=False)
                fig_mul.add_trace(go.Scatter(
                    x=df_pop_edu.index, y=df_pop_edu[col_pop],
                    mode='lines', name='População', line=dict(color='#10b981', width=2)
                ), secondary_y=True)
                fig_mul.update_layout(title="IPCA e População - Série Multivariada com Eixos Y Duplos", template="plotly_white")
                fig_mul.update_yaxes(title_text="Índice IPCA", secondary_y=False)
                fig_mul.update_yaxes(title_text="População (Habitantes)", secondary_y=True)
                st.plotly_chart(fig_mul, use_container_width=True)
                st.caption(r"**Conceito Matemático (Multivariada):** Apresenta múltiplos fenômenos interdependentes plotados em escalas diferentes com eixos Y duplos. Esse cenário permite estudar relações conjuntas como a correlação cruzada e a causalidade de Granger, modeladas por sistemas matriciais de equações simultâneas: $\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{\Phi}_i \mathbf{Y}_{t-i} + \boldsymbol{\epsilon}_t$.")

        with tab_concept2:
            st.markdown("#### Discreta vs Contínua")
            st.markdown("Séries **discretas** representam medições pontuais feitas em intervalos de tempo específicos (como taxas mensais). Séries **contínuas** representam processos que fluem ininterruptamente no tempo, mesmo que sejam medidos anualmente (como a população total).")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_discreta = st.button("Plota IPCA (Discreta)", key="btn_disc")
            with col_btn2:
                btn_continua = st.button("Plota População (Contínua)", key="btn_cont")
                
            if "edu_discreta_state" not in st.session_state:
                st.session_state.edu_discreta_state = "discreta"
                
            if btn_discreta:
                st.session_state.edu_discreta_state = "discreta"
            elif btn_continua:
                st.session_state.edu_discreta_state = "continua"
                
            if st.session_state.edu_discreta_state == "discreta":
                fig_disc = go.Figure()
                df_ipca_recent = df_ipca_edu.tail(36)
                fig_disc.add_trace(go.Scatter(
                    x=df_ipca_recent.index, y=df_ipca_recent[col_ipca],
                    mode='markers', name='IPCA (Discreta)', marker=dict(size=8, color='#ef4444')
                ))
                fig_disc.update_layout(title="IPCA - Medições Discretas (Últimos 36 meses)", yaxis_title="Índice IPCA", template="plotly_white")
                st.plotly_chart(fig_disc, use_container_width=True)
                st.caption(r"**Conceito Matemático (Discreta):** O gráfico plota os dados de inflação usando marcadores/pontos isolados (`mode='markers'`), demonstrando que medições de séries discretas ocorrem em tempos pontuais específicos $t \in \{t_1, t_2, \dots, t_n\}$. Não existem valores observados da série entre as medições periódicas.")
            else:
                fig_cont = go.Figure()
                fig_cont.add_trace(go.Scatter(
                    x=df_pop_edu.index, y=df_pop_edu[col_pop],
                    mode='lines', line_shape='spline', name='População (Contínua)', line=dict(color='#10b981', width=2.5)
                ))
                fig_cont.update_layout(title="População - Processo Contínuo (Interpolação Suave)", yaxis_title="Habitantes", template="plotly_white")
                st.plotly_chart(fig_cont, use_container_width=True)
                st.caption(r"**Conceito Matemático (Contínua):** Representa a evolução da População com uma linha suavizada contínua (`line_shape='spline'`). Embora medida anualmente, a variável populacional representa um fluxo ininterrupto de crescimento acumulado no tempo, em que a evolução é modelada matematicamente por uma função contínua $Y(t)$ onde $t \in \mathbb{R}$.")

        with tab_concept3:
            st.markdown("#### Estacionária vs Não Estacionária")
            st.markdown("Séries **estacionárias** oscilam em torno de uma média fixa, com variância estável. Séries **não estacionárias** exibem tendências ou variabilidade que se deslocam ao longo do tempo.")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_estac = st.button("Plota variação mensal do IPCA (Estacionária)", key="btn_est")
            with col_btn2:
                btn_nao_estac = st.button("Plota População (Não Estacionária)", key="btn_nest")
                
            if "edu_estac_state" not in st.session_state:
                st.session_state.edu_estac_state = "estacionaria"
                
            if btn_estac:
                st.session_state.edu_estac_state = "estacionaria"
            elif btn_nao_estac:
                st.session_state.edu_estac_state = "nao_estacionaria"
                
            if st.session_state.edu_estac_state == "estacionaria":
                df_est = df_ipca_edu.copy()
                df_est['VAR_PERCENT'] = df_est[col_ipca].pct_change() * 100
                df_est = df_est.dropna()
                
                fig_est = go.Figure()
                fig_est.add_trace(go.Scatter(
                    x=df_est.index, y=df_est['VAR_PERCENT'],
                    mode='lines', name='Variação IPCA', line=dict(color='#8b5cf6', width=2)
                ))
                fig_est.add_hline(y=df_est['VAR_PERCENT'].mean(), line_dash="dash", line_color="black", annotation_text="Média")
                fig_est.update_layout(title="Variação Mensal do IPCA - Comportamento Estacionário", yaxis_title="Variação Percentual (%)", template="plotly_white")
                st.plotly_chart(fig_est, use_container_width=True)
                st.caption(r"**Conceito Matemático (Estacionária):** A variação percentual do IPCA oscila constantemente em torno de sua média. Um processo estocástico é fracamente estacionário se sua média for constante ($E[Y_t] = \mu$) e sua função de autocovariância depender apenas do distanciamento temporal $\tau$ entre as observações: $Cov(Y_t, Y_{t-\tau}) = \gamma(\tau)$.")
            else:
                fig_nest = go.Figure()
                fig_nest.add_trace(go.Scatter(
                    x=df_pop_edu.index, y=df_pop_edu[col_pop],
                    mode='lines', name='População', line=dict(color='#10b981', width=2.5)
                ))
                fig_nest.update_layout(title="População - Série Não Estacionária com Tendência de Alta", yaxis_title="Habitantes", template="plotly_white")
                st.plotly_chart(fig_nest, use_container_width=True)
                st.caption("**Conceito Matemático (Não Estacionária):** A série da População apresenta uma tendência contínua de crescimento. A média do processo depende do tempo, $E[Y_t] = f(t)$, o que viola o princípio de estacionariedade. Modelos clássicos exigem a remoção da tendência (ex: por diferenciação) para que a modelagem preditiva seja estatisticamente válida.")

        with tab_concept4:
            st.markdown("#### Determinística vs Estocástica")
            st.markdown("Uma série ou componente **determinístico** pode ser previsto de forma exata por uma equação matemática de tempo. Processos **estocásticos** incluem volatilidades e choques aleatórios imprevisíveis.")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_deter = st.button("Plota linha de tendência central da População ajustada por regressão pura (sem ruído)", key="btn_det")
            with col_btn2:
                btn_estoc = st.button("Plota o IPCA real evidenciando as flutuações e choques aleatórios", key="btn_estoc")
                
            if "edu_deter_state" not in st.session_state:
                st.session_state.edu_deter_state = "deterministica"
                
            if btn_deter:
                st.session_state.edu_deter_state = "deterministica"
            elif btn_estoc:
                st.session_state.edu_deter_state = "estocastica"
                
            if st.session_state.edu_deter_state == "deterministica":
                df_pop_clean = df_pop_edu.copy().dropna()
                X_arr = np.arange(len(df_pop_clean)).reshape(-1, 1)
                y_arr = df_pop_clean[col_pop].values
                reg_model = LinearRegression().fit(X_arr, y_arr)
                df_pop_clean['TENDENCIA'] = reg_model.predict(X_arr)
                
                fig_det = go.Figure()
                fig_det.add_trace(go.Scatter(
                    x=df_pop_clean.index, y=df_pop_clean['TENDENCIA'],
                    mode='lines', name='Tendência Determinística', line=dict(color='#f59e0b', width=2.5)
                ))
                fig_det.update_layout(title="Tendência de Crescimento da População - Modelo Determinístico", yaxis_title="Habitantes", template="plotly_white")
                st.plotly_chart(fig_det, use_container_width=True)
                st.caption("**Conceito Matemático (Determinística):** Plota a linha de tendência central da População ajustada por regressão linear pura. Em um modelo determinístico, não há termo de erro aleatório; os valores da série dependem estritamente do tempo: $Y_t = \beta_0 + \beta_1 t$. Qualquer ponto futuro pode ser calculado sem incerteza.")
            else:
                fig_estoc = go.Figure()
                fig_estoc.add_trace(go.Scatter(
                    x=df_ipca_edu.index, y=df_ipca_edu[col_ipca],
                    mode='lines', name='IPCA Real (Estocástico)', line=dict(color='#2563eb', width=2)
                ))
                fig_estoc.update_layout(title="IPCA Real - Processo Estocástico", yaxis_title="Índice IPCA", template="plotly_white")
                st.plotly_chart(fig_estoc, use_container_width=True)
                st.caption(r"**Conceito Matemático (Estocástica):** Plota o IPCA real, ilustrando flutuações e choques econômicos inesperados. Séries reais contêm um termo estocástico (aleatório) $\epsilon_t \sim N(0, \sigma^2)$ de modo que $Y_t = f(Y_{t-1}) + \epsilon_t$, impossibilitando previsões perfeitas e exigindo estimativas de probabilidade.")
