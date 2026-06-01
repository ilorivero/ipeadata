# 📊 IPEADATA Multi-Series Hub

Este é um aplicativo interativo desenvolvido em **Python** e **Streamlit** para visualização, previsão e análise interpretativa de múltiplas séries temporais da base de dados do **IPEADATA** (Instituto de Pesquisa Econômica Aplicada). O aplicativo permite realizar previsões com algoritmos modernos de Machine Learning, analisar relações estatísticas complexas e explorar conceitos didáticos de séries temporais.

---

## 🚀 Como Instalar e Executar

Siga os passos abaixo para configurar o ambiente e executar o aplicativo em sua máquina local:

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.10** ou superior instalado em seu computador.

### 2. Clonar ou Acessar a Pasta do Projeto
Abra o seu terminal (PowerShell ou Bash) na pasta raiz do projeto.

### 3. Criar e Ativar um Ambiente Virtual
Recomenda-se o uso de um ambiente virtual (`.venv`) para isolar as dependências:

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**No macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias declaradas no arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Executar o Streamlit
Inicialize o servidor de desenvolvimento do Streamlit:
```bash
python -m streamlit run app.py --server.port 8506
```
Após a inicialização, o terminal exibirá as URLs de acesso local. Abra `http://localhost:8506` em seu navegador de preferência.

---

## 🔌 Conexão com o IPEADATA

O aplicativo consome os dados diretamente do IPEADATA em tempo real através da biblioteca open-source `ipeadatapy`.

### Integração de Dados e Tratamentos:
1. **Listagem e Filtragem de Séries:** A aplicação busca a lista de todas as séries disponíveis via `ipeadatapy.list_series()`. Para otimizar a experiência do usuário e evitar quebras de requisição, o sistema filtra e remove automaticamente todas as séries temporais que contenham a palavra **"INATIVA"** em seu nome.
2. **Download Multissérie e Alinhamento Temporal (Inner Join):** Ao selecionar múltiplas séries, o sistema realiza o download individual de cada uma delas via `ipeadatapy.timeseries()` e aplica um **Inner Join** combinando-as em um único DataFrame com base no índice de datas. Isso garante que a matriz de correlação, covariância e os gráficos de projeção trabalhem apenas com períodos congruentes.
3. **Resolução de Códigos Alternativos (Fallback robusto):** Para garantir o funcionamento das demonstrações didáticas mesmo se o usuário utilizar códigos legados ou simplificados, a função `buscar_valores()` no arquivo `data_fetcher.py` realiza um mapeamento automático sob o capô:
   - `PRECO12_IPCA12` $\rightarrow$ `PRECOS12_IPCA12` (Índice de Preços ao Consumidor Amplo)
   - `POPU_POPU` $\rightarrow$ `DEPIS_POP` (População Residente Nacional)

---

## 🧠 Algoritmos de Previsão de Machine Learning

Na aba principal de exploração, o usuário pode selecionar o algoritmo a ser utilizado para projetar os próximos 6 passos futuros da série. Os algoritmos implementados no arquivo `predictor.py` são:

### 1. Regressão Linear Autoregressiva (Autoregressive Linear Regression)
* **Biblioteca:** `scikit-learn` (`sklearn.linear_model.LinearRegression`).
* **Abordagem:** O modelo cria três características de lags temporais ($Y_{t-1}$, $Y_{t-2}$ e $Y_{t-3}$) para cada ponto histórico. O modelo aprende a relação linear:
  $$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \phi_3 Y_{t-3} + \epsilon_t$$
* **Projeção Futura:** É realizada de forma **recursiva** (multi-step forecast). O valor previsto para o passo $T+1$ é realimentado como entrada ($Y_{t-1}$) para calcular a previsão de $T+2$, e assim sucessivamente.

### 2. XGBoost Regressor
* **Biblioteca:** `xgboost` (`xgboost.XGBRegressor`).
* **Abordagem:** Utiliza a mesma estrutura de 3 lags autoregressivos ($Y_{t-1}$, $Y_{t-2}$, $Y_{t-3}$), porém alimentados em um regressor baseado em árvores de decisão impulsionadas por gradiente (*Gradient Boosted Decision Trees*).
* **Vantagem:** Consegue capturar padrões autoregressivos não lineares e dinâmicas complexas nas séries que a regressão linear falharia em modelar.

### 3. Prophet
* **Biblioteca:** `prophet` (`prophet.Prophet`).
* **Abordagem:** Desenvolvido pela Meta (Facebook), é um modelo de previsão aditivo projetado para séries temporais de negócios com forte componente sazonal. A série é decomposta como:
  $$Y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
  Onde $g(t)$ é a tendência linear ou logística, $s(t)$ representa efeitos sazonais periódicos (anuais/semanais) e $h(t)$ representa feriados.
* **Integração:** Os dados históricos de data e valor são estruturados no padrão exigido pelo Prophet (colunas `ds` e `y`). A previsão é calculada de forma direta sobre a linha do tempo futura gerada pelo app.

### 4. ARIMA(1, 1, 1) (Autoregressive Integrated Moving Average)
* **Biblioteca:** `statsmodels` (`statsmodels.tsa.arima.model.ARIMA`).
* **Abordagem:** É um dos modelos estatísticos lineares clássicos mais consolidados para previsão de séries temporais. O modelo é parametrizado com ordem $(1, 1, 1)$ contemplando:
  - **AR(1):** Parte autoregressiva, relacionando o valor corrente com seu lag anterior.
  - **I(1):** Nível de diferenciação para estabilizar a média e tornar a série estacionária.
  - **MA(1):** Parte de média móvel, relacionando o valor com erros estocásticos de previsões passadas.
* **Tratamento de Frequência:** O app analisa o índice de datas e garante a definição de uma frequência explícita (`asfreq`), aplicando preenchimento por propagação de valor (`ffill`) caso restem descontinuidades.

### 5. SARIMA (Seasonal Autoregressive Integrated Moving Average)
* **Biblioteca:** `statsmodels` (`statsmodels.tsa.statespace.sarimax.SARIMAX`).
* **Abordagem:** Extensão do ARIMA projetada para séries com flutuações sazonais sistemáticas. Adiciona parâmetros sazonais específicos ao modelo:
  - **Identificação de Frequência Sazonal:** Se o índice for detectado como mensal ('M'), define-se a ordem sazonal como `seasonal_order=(1, 1, 1, 12)`. Se for detectado como trimestral ('Q'), adota-se `seasonal_order=(1, 1, 1, 4)`.
  - **Vantagem:** Essencial para modelar ciclos de negócios e sazonalidades anuais comuns em indicadores econômicos de inflação e consumo.

---

## 📊 Análises Estatísticas Realizadas

O aplicativo fornece um painel analítico avançado no arquivo `stats_analyzer.py` para entender as relações entre as séries temporais carregadas:

### 1. Matriz de Correlação de Pearson
Calcula o grau de associação linear entre os pares de séries históricas integradas:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
Os coeficientes variam entre $-1$ (correlação negativa perfeita) e $+1$ (correlação positiva perfeita).

### 2. Matriz de Covariância
Mede a variabilidade conjunta de duas variáveis aleatórias. Indica a direção do relacionamento linear (se covariam positivamente ou negativamente), mantendo a escala original das variáveis.

### 3. Função de Correlação Cruzada (CCF - Cross-Correlation Function)
Quando **exatamente duas séries** são selecionadas, o app computa a correlação cruzada utilizando a biblioteca `statsmodels`. 
* **O que faz:** Mede a relação entre $X_t$ e $Y_{t+k}$ para diferentes defasagens (lags) temporais $k$.
* **Interpretação:**
  - **Lags negativos ($k < 0$):** Indicam se a série $X$ antecede/lidera a série $Y$ (sinalizador antecedente).
  - **Lags positivos ($k > 0$):** Indicam se a série $Y$ antecede/lidera a série $X$.
* **Intervalo de Confiança (IC):** O gráfico exibe linhas tracejadas a $\pm 1.96 / \sqrt{N}$ representando o limite de significância de 95%. Valores de correlação cruzada que ultrapassam essas linhas são estatisticamente relevantes.

### 4. Interpretador de Insights Automatizado
Traduz coeficientes de Pearson abstratos em relatórios simples em português (ex: classificando a força da correlação como *Forte*, *Moderada*, *Fraca* ou *Nula* e explicando o sentido da relação).

---

## 🎓 Exemplos Didáticos de Classificação

O aplicativo conta com uma seção dedicada à educação sobre séries temporais, ilustrando 4 distinções conceituais fundamentais utilizando o **IPCA** e a **População Residente**:

1. **Univariada vs Multivariada:**
   - *Univariada:* Plota exclusivamente a série histórica do IPCA. A previsão é feita com base apenas nos dados passados de inflação.
   - *Multivariada:* Plota o IPCA e a População em eixos Y independentes simultâneos (esquerdo/direito), ilustrando a covariância mútua.
2. **Discreta vs Contínua:**
   - *Discreta:* Plota o IPCA em intervalos mensais usando marcadores pontuais (`mode='markers'`), demonstrando que medições discretas $t \in \mathbb{Z}$ não revelam flutuações intermediárias.
   - *Contínua:* Plota a População com uma linha suavizada (`line_shape='spline'`), ilustrando processos contínuos $t \in \mathbb{R}$ onde a variável evolui ininterruptamente.
3. **Estacionária vs Não Estacionária:**
   - *Estacionária:* Plota a variação mensal do IPCA. Mostra que o processo oscila de forma estável ao redor de sua média aritmética (com variância e covariância constantes).
   - *Não Estacionária:* Plota a População Residente evidenciando uma tendência clara de alta, demonstrando que a média do processo depende do tempo ($E[Y_t] = f(t)$).
4. **Determinística vs Estocástica:**
   - *Determinística:* Ajusta e plota a reta de tendência da População obtida por regressão linear pura. Livre de ruídos, o futuro é deduzido exatamente por $Y_t = \beta_0 + \beta_1 t$.
   - *Estocástica:* Plota o IPCA real, que contém ruídos do mercado, choques econômicos e volatilidade imprevisível $\epsilon_t \sim N(0, \sigma^2)$, exigindo modelagem probabilística.

---

## 📁 Estrutura de Arquivos do Projeto

* `app.py`: Arquivo principal da interface do Streamlit, renderização do dashboard, abas e lógica de interação.
* `data_fetcher.py`: Comunicação com a API `ipeadatapy`, filtragem de séries inativas, alinhamento multissérie (inner join) e fallback de códigos de séries.
* `predictor.py`: Preparação e execução dos pipelines preditivos (Regressão Linear, XGBoost, Prophet).
* `stats_analyzer.py`: Lógica para cálculo de correlação, covariância, CCF e geração textual de insights.
* `requirements.txt`: Lista de dependências Python.
* `chat_context_summary.md`: Documentação de backup contendo o histórico de desenvolvimento do chat para portabilidade entre computadores.
