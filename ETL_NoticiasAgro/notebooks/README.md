# Ciência de Dados - Exemplos de Análise

Este diretório contém notebooks Jupyter com exemplos práticos de análise dos dados de cotações.

## 📓 Notebooks Disponíveis

### 1. `01_exploracao_dados.ipynb`
- **Análise Exploratória de Dados (EDA)**
- Estatísticas descritivas
- Visualizações temporais
- Análise de correlações
- Identificação de padrões sazonais

### 2. `02_feature_engineering.ipynb`
- **Engenharia de Features**
- Criação de indicadores técnicos
- Lags e médias móveis
- Features temporais
- Normalização e escalonamento

### 3. `03_forecasting.ipynb`
- **Previsão de Preços**
- Modelos ARIMA
- Prophet (Facebook)
- LSTM para séries temporais
- Validação cruzada temporal

### 4. `04_machine_learning.ipynb`
- **Modelos de Machine Learning**
- Classificação (direção do preço)
- Regressão (valor do preço)
- XGBoost, Random Forest
- Hyperparameter tuning

### 5. `05_analise_regional.ipynb`
- **Análise Regional**
- Clustering de estados
- Análise de spread entre regiões
- Mapas coropléticos
- Correlações regionais

### 6. `06_deteccao_anomalias.ipynb`
- **Detecção de Anomalias**
- Outliers em preços
- Change point detection
- Alertas automáticos
- Análise de volatilidade

## 🚀 Como Usar

1. **Instalar dependências**:
```bash
pip install jupyter notebook matplotlib seaborn plotly scikit-learn
pip install prophet xgboost lightgbm statsmodels
```

2. **Iniciar Jupyter**:
```bash
jupyter notebook
```

3. **Abrir notebooks na ordem**:
   - Começar com `01_exploracao_dados.ipynb`
   - Seguir sequência numérica

## 📊 Dados Necessários

Os notebooks assumem que você já coletou dados usando o sistema ETL:

```python
# Exemplo de coleta de dados históricos
from src.data_science_loader import DataScience_Loader

loader = DataScience_Loader("../output/parquet")
data = loader.load_time_series(
    start_date="2024-01-01",
    end_date="2025-08-27",
    table_type="indicadores_estados"
)
```

## 🎯 Casos de Uso por Notebook

### Forecasting (`03_forecasting.ipynb`)
- Previsão de preços 1-30 dias
- Comparação de modelos
- Intervalos de confiança
- Métricas de avaliação

### Machine Learning (`04_machine_learning.ipynb`)
- Prever direção do preço (sobe/desce)
- Features importance
- Cross-validation
- Modelo em produção

### Análise Regional (`05_analise_regional.ipynb`)
- Identificar regiões similares
- Análise de arbitragem
- Mapas interativos
- Dashboard regional

## 📈 Performance Esperada

Para **2-3 anos de dados**:
- Carregamento: 5-15 segundos
- EDA completa: 2-5 minutos
- Treinamento ML: 30-60 segundos
- Forecasting: 10-30 segundos

## 🔧 Configurações

### Memória Recomendada:
- **Mínimo**: 8GB RAM
- **Recomendado**: 16GB+ RAM
- **Para 10 anos**: 32GB RAM

### Otimizações:
- Usar `parallel=True` no carregamento
- Carregar apenas colunas necessárias
- Filtrar por período relevante
- Cache de resultados intermediários

## 📋 Checklist de Análise

### ✅ Exploração Inicial:
- [ ] Carregar e visualizar dados
- [ ] Verificar missing values
- [ ] Análise temporal básica
- [ ] Estatísticas por região

### ✅ Feature Engineering:
- [ ] Criar lags e médias móveis
- [ ] Indicadores técnicos
- [ ] Features sazonais
- [ ] Normalização

### ✅ Modelagem:
- [ ] Baseline simples
- [ ] Modelos tradicionais
- [ ] Deep Learning
- [ ] Ensemble methods

### ✅ Validação:
- [ ] Train/validation/test split
- [ ] Cross-validation temporal
- [ ] Métricas de negócio
- [ ] Backtesting

### ✅ Deploy:
- [ ] Modelo final selecionado
- [ ] Pipeline de inference
- [ ] Monitoramento
- [ ] Atualizações automáticas

## 💡 Dicas Importantes

1. **Sempre validar com dados out-of-time**
2. **Cuidado com data leakage**
3. **Considerar sazonalidade**
4. **Monitorar drift nos dados**
5. **Documentar hipóteses e descobertas**

## 🔍 Próximos Passos

Após completar os notebooks básicos:

1. **Dashboard Interativo** (Streamlit/Dash)
2. **API de Predição** (FastAPI)
3. **Alertas Automáticos** (Email/Slack)
4. **Integração com Trading** (se aplicável)
5. **Relatórios Automatizados** (PDFs)

## 📞 Suporte

Para dúvidas sobre os notebooks:
1. Verificar se dados estão carregados
2. Conferir dependências instaladas
3. Validar caminhos dos arquivos
4. Checar versões das bibliotecas

---

**Nota**: Os notebooks são templates e devem ser adaptados para suas necessidades específicas de análise.
