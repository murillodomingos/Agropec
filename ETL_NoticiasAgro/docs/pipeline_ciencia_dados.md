# Pipeline de Ciência de Dados - Análise de Cotações

Este documento descreve o pipeline completo para usar os dados de cotações em projetos de ciência de dados.

## 🏗️ Arquitetura do Pipeline

```
Dados Parquet (ETL) → Análise Exploratória → Feature Engineering → Modelagem → Deploy
     ↓                        ↓                      ↓              ↓          ↓
   Storage              Jupyter Notebooks        Transformações   ML Models   API/Dashboard
```

## 📊 1. CARREGAMENTO E EXPLORAÇÃO DE DADOS

### Estrutura de Projeto Recomendada:
```
projeto-ciencia-dados/
├── data/
│   ├── raw/                    # Link simbólico para os dados Parquet
│   ├── processed/              # Dados processados
│   └── features/               # Features engineered
├── notebooks/
│   ├── 01-exploracao.ipynb    # Análise exploratória
│   ├── 02-feature-eng.ipynb   # Feature engineering
│   ├── 03-modelagem.ipynb     # Modelos ML
│   └── 04-validacao.ipynb     # Validação e testes
├── src/
│   ├── data/
│   │   ├── loader.py          # Carregamento de dados
│   │   └── preprocessor.py    # Pré-processamento
│   ├── features/
│   │   ├── builder.py         # Construção de features
│   │   └── selector.py        # Seleção de features
│   ├── models/
│   │   ├── trainer.py         # Treinamento
│   │   ├── predictor.py       # Predições
│   │   └── evaluator.py       # Avaliação
│   └── visualization/
│       └── plots.py           # Visualizações
├── models/                     # Modelos salvos
├── reports/                    # Relatórios gerados
└── requirements.txt           # Dependências
```

## 🚀 2. QUICK START

### Instalação:
```bash
# Instalar dependências básicas
pip install -r requirements.txt

# Instalar dependências completas para DS
pip install -r requirements-datascience.txt

# Iniciar Jupyter
jupyter notebook
```

### Carregamento Rápido:
```python
from src.data_science_loader import DataScience_Loader

# Inicializar
loader = DataScience_Loader("output/parquet")

# Carregar dados para análise
data = loader.load_time_series(
    start_date="2024-01-01",
    end_date="2025-08-27",
    table_type="indicadores_estados",
    states=["São Paulo", "Minas Gerais"]  # Estados específicos
)

print(f"Dados carregados: {len(data)} registros")
```

## 📈 3. CASOS DE USO TÍPICOS

### A. **🔮 Previsão de Preços (Forecasting)**

**Objetivo**: Prever preços futuros do boi gordo  
**Horizonte**: 1-30 dias  
**Modelos**: ARIMA, Prophet, LSTM, XGBoost

```python
# Exemplo de setup para forecasting
from src.data_science_loader import load_for_forecasting

# Carregar dados otimizados
ts_data = load_for_forecasting(
    parquet_path="output/parquet",
    target_state="São Paulo",
    lookback_days=365
)

# Usar com Prophet
from prophet import Prophet
model = Prophet()
model.fit(ts_data.rename(columns={'date': 'ds', 'price_brl': 'y'}))
```

**Métricas**: MAE, MAPE, RMSE  
**Validação**: Time series cross-validation

### B. **📊 Análise de Volatilidade**

**Objetivo**: Modelar e prever volatilidade  
**Modelos**: GARCH, EGARCH, Stochastic Volatility

```python
# Exemplo de análise de volatilidade
import pandas as pd
from arch import arch_model

# Calcular retornos
returns = data['price_brl'].pct_change().dropna() * 100

# Modelo GARCH
model = arch_model(returns, vol='GARCH', p=1, q=1)
result = model.fit()
```

### C. **🗺️ Clustering de Estados**

**Objetivo**: Agrupar estados por comportamento de preços  
**Algoritmos**: K-Means, Hierarchical Clustering

```python
# Matriz de correlação entre estados
pivot_data = loader.create_pivot_table(
    data, value_col='price_brl', 
    columns_col='state'
)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data.T)
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(scaled_data)
```

### D. **⚖️ Análise de Spread**

**Objetivo**: Identificar oportunidades de arbitragem  
**Métrica**: Diferenças de preços entre regiões

```python
# Calcular spreads
sp_prices = data[data['state'] == 'São Paulo']['price_brl']
mg_prices = data[data['state'] == 'Minas Gerais']['price_brl']
spread = sp_prices - mg_prices

# Detectar oportunidades
opportunities = spread[abs(spread) > spread.std() * 2]
```

### E. **🚨 Detecção de Anomalias**

**Objetivo**: Identificar preços anômalos automaticamente  
**Algoritmos**: Isolation Forest, DBSCAN, Statistical tests

```python
from sklearn.ensemble import IsolationForest

# Detectar anomalias
iso_forest = IsolationForest(contamination=0.1)
anomalies = iso_forest.fit_predict(data[['price_brl']])
```

## 🎯 4. FEATURES PARA MACHINE LEARNING

### Features Temporais:
```python
# Features automáticas com o loader
features_df = loader.create_features_dataframe(
    data, price_col='price_brl'
)
```

- ✅ **Lags**: price_t-1, price_t-7, price_t-30
- ✅ **Médias Móveis**: MA_7, MA_30, MA_90
- ✅ **Volatilidade**: Vol_7, Vol_30 (rolling std)
- ✅ **Retornos**: Return_1d, Return_7d
- ✅ **Sazonalidade**: month, quarter, day_of_week

### Features Técnicas Avançadas:
```python
def create_technical_indicators(data):
    # RSI
    data['RSI'] = calculate_rsi(data['price_brl'], window=14)
    
    # Bollinger Bands
    data['BB_upper'] = data['price_brl'].rolling(20).mean() + 2*data['price_brl'].rolling(20).std()
    data['BB_lower'] = data['price_brl'].rolling(20).mean() - 2*data['price_brl'].rolling(20).std()
    
    # MACD
    ema12 = data['price_brl'].ewm(span=12).mean()
    ema26 = data['price_brl'].ewm(span=26).mean()
    data['MACD'] = ema12 - ema26
    
    return data
```

### Features Econômicas (externas):
- 💱 **Taxa de câmbio**: USD/BRL
- 📊 **Commodities**: Milho, Soja (correlacionadas)
- 📈 **Índices**: Bovespa, CDI, IPCA
- 🌍 **Exportações**: Volume de carne bovina

## 🔧 5. STACK TECNOLÓGICA

### 🏗️ **Infraestrutura Base**:
- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas  
- **pyarrow**: Leitura eficiente de Parquet
- **jupyter**: Análise interativa

### 📊 **Visualização**:
- **matplotlib**: Gráficos estáticos
- **seaborn**: Análises estatísticas
- **plotly**: Gráficos interativos
- **streamlit**: Dashboards web

### 🤖 **Machine Learning**:
- **scikit-learn**: ML tradicional
- **xgboost**: Gradient boosting
- **tensorflow**: Deep learning
- **optuna**: Hyperparameter tuning

### ⏰ **Séries Temporais**:
- **prophet**: Forecasting robusto
- **statsmodels**: Modelos estatísticos
- **arch**: Modelos GARCH
- **tsfresh**: Feature extraction automática

### ⚡ **Performance**:
- **dask**: Processamento paralelo
- **polars**: DataFrame ultra-rápido
- **numba**: Compilação JIT

## 📊 6. PERFORMANCE E OTIMIZAÇÃO

### Para datasets grandes (10 anos):

#### Otimizações de Carregamento:
```python
# ✅ Carregar apenas colunas necessárias
data = loader.load_data_range(
    start_date="2020-01-01",
    end_date="2025-08-27",
    columns=["date", "state", "price_brl"],  # Apenas essenciais
    parallel=True  # Processamento paralelo
)

# ✅ Filtros pushdown (usar particionamento)
# ✅ Chunking para modelos grandes
# ✅ Cache de resultados intermediários
```

#### Memory Management:
```python
# Para datasets muito grandes
import dask.dataframe as dd

# Usar Dask para processamento out-of-core
ddf = dd.read_parquet("output/parquet/*/")
result = ddf.groupby("state").price_brl.mean().compute()
```

### Benchmarks Estimados:

| Operação | 1 ano | 5 anos | 10 anos |
|----------|--------|--------|---------|
| Carregamento | 2s | 8s | 15s |
| EDA Completa | 30s | 2min | 5min |
| Feature Engineering | 10s | 45s | 2min |
| Modelo XGBoost | 5s | 30s | 1min |
| Forecasting Prophet | 15s | 1min | 3min |

**Hardware recomendado**:
- **RAM**: 16GB+ (32GB para 10 anos)
- **CPU**: 8+ cores
- **Storage**: SSD recomendado

## 🎯 7. WORKFLOWS TÍPICOS

### Workflow 1: **Análise Exploratória Rápida**
```python
# 1. Carregar dados
data = loader.load_time_series("2024-01-01", "2025-08-27", "indicadores_estados")

# 2. Estatísticas básicas
print(data.describe())

# 3. Visualização temporal
data.groupby('date')['price_brl'].mean().plot(figsize=(12,6))

# 4. Análise por estado
data.groupby('state')['price_brl'].mean().sort_values().plot(kind='bar')
```

### Workflow 2: **Modelo de Previsão Completo**
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Preparar features
features_df = loader.create_features_dataframe(data)
features_df = features_df.dropna()

# 2. Split temporal
X = features_df.select_dtypes(include=[np.number]).drop(['price_brl'], axis=1)
y = features_df['price_brl']

tscv = TimeSeriesSplit(n_splits=5)

# 3. Treinar modelo
scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    scores.append(mean_absolute_error(y_val, pred))

print(f"MAE médio: {np.mean(scores):.2f}")
```

### Workflow 3: **Dashboard Automatizado**
```python
import streamlit as st

# Dashboard com Streamlit
st.title("🐂 Dashboard de Cotações")

# Sidebar para filtros
states = st.sidebar.multiselect("Estados", data['state'].unique())
date_range = st.sidebar.date_input("Período", [start_date, end_date])

# Filtrar dados
filtered_data = data[
    (data['state'].isin(states)) & 
    (data.index >= date_range[0]) & 
    (data.index <= date_range[1])
]

# Plotar
st.line_chart(filtered_data.groupby('date')['price_brl'].mean())
```

## 🎨 8. TEMPLATES DE ANÁLISE

### Template: **Análise de Sazonalidade**
```python
def analyze_seasonality(data):
    # Decomposição sazonal
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    monthly_data = data.groupby(data.index.to_period('M'))['price_brl'].mean()
    decomposition = seasonal_decompose(monthly_data, model='additive')
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Tendência')
    decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    return decomposition
```

### Template: **Backtesting de Estratégias**
```python
def backtest_strategy(data, strategy_func, initial_capital=100000):
    """
    Template para backtesting de estratégias
    """
    positions = strategy_func(data)  # 1 = comprar, -1 = vender, 0 = neutro
    returns = data['price_brl'].pct_change()
    strategy_returns = positions.shift(1) * returns
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    final_value = initial_capital * cumulative_returns.iloc[-1]
    
    metrics = {
        'return': (final_value / initial_capital - 1) * 100,
        'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
        'max_drawdown': (cumulative_returns / cumulative_returns.expanding().max() - 1).min() * 100
    }
    
    return metrics, cumulative_returns
```

## 🚀 9. DEPLOY E PRODUÇÃO

### Opção 1: **API REST com FastAPI**
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('models/price_predictor.pkl')

@app.post("/predict")
async def predict_price(data: dict):
    prediction = model.predict([list(data.values())])
    return {"predicted_price": prediction[0]}
```

### Opção 2: **Streamlit Dashboard**
```bash
# Comando para deploy
streamlit run dashboard.py --server.port 8501
```

### Opção 3: **Jupyter Notebooks como Reports**
```python
# Usando papermill para automatizar notebooks
import papermill as pm

pm.execute_notebook(
    'template.ipynb',
    f'reports/report_{datetime.now().strftime("%Y-%m-%d")}.ipynb',
    parameters={'start_date': '2025-01-01', 'end_date': '2025-08-27'}
)
```

## 📋 10. CHECKLIST DE PROJETO

### ✅ **Setup Inicial**:
- [ ] Dados ETL coletados e validados
- [ ] Ambiente virtual configurado
- [ ] Dependências instaladas
- [ ] Jupyter notebook funcionando

### ✅ **Análise Exploratória**:
- [ ] Estatísticas descritivas
- [ ] Visualizações temporais
- [ ] Análise por estados/regiões
- [ ] Identificação de outliers
- [ ] Padrões sazonais

### ✅ **Feature Engineering**:
- [ ] Features temporais (lags, médias móveis)
- [ ] Indicadores técnicos
- [ ] Features categóricas (estado, mês)
- [ ] Normalização/escalonamento
- [ ] Seleção de features

### ✅ **Modelagem**:
- [ ] Baseline simples (média, último valor)
- [ ] Modelos tradicionais (linear, tree-based)
- [ ] Séries temporais (ARIMA, Prophet)
- [ ] Deep Learning (LSTM, GRU)
- [ ] Ensemble methods

### ✅ **Validação**:
- [ ] Split temporal correto
- [ ] Cross-validation adequada
- [ ] Métricas de negócio
- [ ] Backtesting para estratégias
- [ ] Análise de resíduos

### ✅ **Deploy**:
- [ ] Modelo final selecionado
- [ ] Pipeline de inference
- [ ] API ou dashboard
- [ ] Monitoramento de performance
- [ ] Documentação

## 💡 11. DICAS E BOAS PRÁTICAS

### 🎯 **Análise de Dados**:
1. **Sempre validar com dados out-of-time** (never future data)
2. **Considerar feriados e fins de semana** (dados podem não existir)
3. **Documentar hipóteses e descobertas** em markdown
4. **Usar visualizações interativas** para exploração
5. **Salvar datasets intermediários** para reuso

### ⚡ **Performance**:
1. **Carregar apenas dados necessários** (período + colunas)
2. **Usar processamento paralelo** quando possível
3. **Cache resultados custosos** (feature engineering)
4. **Monitor memory usage** em datasets grandes
5. **Considerar downsampling** para visualizações

### 🔬 **Modelagem**:
1. **Começar sempre com baseline simples**
2. **Feature importance** para interpretabilidade  
3. **Usar métricas de negócio** além de técnicas
4. **Validação cruzada temporal** obrigatória
5. **Documentar hyperparameters** e resultados

### 🚀 **Deploy**:
1. **Versionamento de modelos** (MLflow, DVC)
2. **Monitoring de drift** nos dados
3. **Alertas para anomalias** de performance
4. **Backup de modelos** anteriores
5. **Logs detalhados** de predições

## 📞 12. SUPORTE E TROUBLESHOOTING

### ❓ **Problemas Comuns**:

**"Dados não carregam"**:
- ✅ Verificar se ETL foi executado
- ✅ Validar caminhos dos arquivos
- ✅ Conferir formato de datas

**"Notebook muito lento"**:
- ✅ Reduzir período de análise
- ✅ Carregar apenas colunas necessárias  
- ✅ Usar `parallel=True`
- ✅ Considerar usar Dask

**"Modelos com performance ruim"**:
- ✅ Verificar data leakage
- ✅ Validar split temporal
- ✅ Adicionar mais features
- ✅ Tunar hyperparameters

**"Memória insuficiente"**:
- ✅ Processar dados em chunks
- ✅ Usar dtypes otimizados
- ✅ Clear variables desnecessárias
- ✅ Considerar Dask/Polars

### 🆘 **Getting Help**:
1. Verificar logs de erro detalhadamente
2. Conferir versões das bibliotecas
3. Testar com dataset menor primeiro
4. Consultar documentação das libs
5. Procurar no Stack Overflow

---

## 🎉 CONCLUSÃO

Este pipeline oferece uma base sólida para análise de dados de cotações, desde exploração básica até modelos avançados em produção. 

**Próximos passos recomendados**:
1. Executar notebook de exploração
2. Implementar seu primeiro modelo
3. Criar dashboard personalizado  
4. Automatizar análises regulares
5. Expandir para outras commodities

**Lembre-se**: Comece simples, valide constantemente e documente tudo! 🚀
