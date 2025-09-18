# ETL Notícias Agro - Sistema de Cotações

Sistema otimizado para coleta e armazenamento de dados de cotações de boi gordo do portal Notícias Agrícolas.

## 🚀 Características

- **Coleta automatizada** de cotações diárias
- **Armazenamento eficiente** em formato Parquet
- **Particionamento inteligente** por ano/mês
- **Processamento paralelo** para grandes volumes
- **Schema padronizado** para diferentes tipos de dados
- **Consultas otimizadas** para análise de dados
- **Suporte a múltiplas datas** com backfill automático

## 📁 Estrutura do Projeto

```
ETL_NoticiasAgro/
├── src/
│   ├── scraper.py           # Coleta de dados do site
│   ├── data_processor.py    # Processamento e armazenamento
│   └── __init__.py
├── output/
│   └── parquet/            # Dados em formato Parquet
│       ├── indicadores_simples/
│       ├── indicadores_estados/
│       ├── contratos_futuros/
│       ├── reposicao/
│       └── mercados_externos/
├── main.py                 # Script principal CLI
├── examples.py            # Exemplos de uso
├── requirements.txt       # Dependências
└── README.md             # Este arquivo
```

## 🛠️ Instalação

1. **Criar ambiente virtual**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Tipos de Dados Coletados

O sistema organiza automaticamente os dados em 5 categorias:

### 1. **Indicadores Simples** (`indicadores_simples`)
- Indicador do Boi Gordo Esalq/B3
- Boi Gordo - Média SP a prazo
- Bezerro Esalq/BM&F Bovespa - MS

**Schema**: `date`, `indicator_name`, `price_brl`, `variation_pct`, `price_usd`

### 2. **Indicadores por Estado** (`indicadores_estados`)
- Indicador da Novilha
- Indicador da Vaca  
- Indicador do Boi
- Média de Boi Gordo - Indicador AgroBrazil

**Schema**: `date`, `indicator_name`, `state`, `price_brl`, `variation_pct`, `price_usd`

### 3. **Contratos Futuros** (`contratos_futuros`)
- Boi Gordo - B3 (Pregão Regular)

**Schema**: `date`, `contract_month`, `price`, `variation`, `indicator_name`

### 4. **Reposição** (`reposicao`)
- Reposição Nelore - Fêmea
- Reposição Nelore - Macho

**Schema**: `date`, `state`, `category`, `desmama`, `bezerra_bezerro`, `novilha_garrote`, `vaca_boi_magro`, `indicator_name`

### 5. **Mercados Externos** (`mercados_externos`)
- Chicago (CME)
- Brasil (B3)
- New York (NYBOT)

**Schema**: `date`, `market`, `contract`, `price`, `variation`

## 🔧 Uso do Sistema

### Comando Principal (main.py)

#### Coletar uma data específica:
```bash
python main.py single --date 2025-08-20
```

#### Coletar um intervalo de datas:
```bash
python main.py range --start-date 2025-08-01 --end-date 2025-08-20
```

#### Backfill automático (últimos 30 dias):
```bash
python main.py backfill
```

#### Backfill personalizado:
```bash
python main.py backfill --start-date 2024-01-01 --end-date 2025-08-27
```

#### Ver estatísticas de armazenamento:
```bash
python main.py stats
```

#### Opções adicionais:
```bash
# Usar mais workers para processamento paralelo
python main.py range --start-date 2025-08-01 --end-date 2025-08-20 --workers 5

# Reprocessar arquivos existentes
python main.py range --start-date 2025-08-01 --end-date 2025-08-20 --force

# Diretório de saída personalizado
python main.py single --date 2025-08-20 --output-dir /path/to/custom/output
```

### Uso Programático

```python
from src.scraper import scrap_ox_data
from src.data_processor import CotacaoDataProcessor

# Coletar dados de uma data
raw_data = scrap_ox_data("2025-08-20")

# Processar e salvar
processor = CotacaoDataProcessor("output")
processed_dfs = processor.process_raw_data(raw_data)
processor.save_to_parquet(processed_dfs, "2025-08-20")

# Consultar dados armazenados
data = processor.load_date_range("2025-08-01", "2025-08-20", ["indicadores_estados"])
df_estados = data["indicadores_estados"]
print(f"Dados carregados: {len(df_estados)} registros")
```

## 📈 Análise de Dados

### Exemplos de Consultas

```python
import pandas as pd
from src.data_processor import CotacaoDataProcessor

processor = CotacaoDataProcessor("output")

# Carregar dados de indicadores por estado
data = processor.load_date_range("2025-08-01", "2025-08-20", ["indicadores_estados"])
df = data["indicadores_estados"]

# Preço médio por estado
precos_por_estado = df.groupby('state')['price_brl'].mean().sort_values(ascending=False)
print(precos_por_estado)

# Variação diária de preços
variacao_diaria = df.groupby('date')['price_brl'].mean()
print(variacao_diaria)

# Top 5 estados com maior preço
top_estados = df.groupby('state')['price_brl'].mean().nlargest(5)
print(top_estados)
```

### Exportação

```python
# Exportar para CSV
df.to_csv("cotacoes_estados.csv", index=False, encoding='utf-8')

# Exportar para Excel
df.to_excel("cotacoes_estados.xlsx", index=False)

# Criar resumo por estado
resumo = df.groupby(['state', 'date']).agg({
    'price_brl': ['mean', 'min', 'max'],
    'variation_pct': 'mean'
}).round(2)
resumo.to_csv("resumo_cotacoes.csv")
```

## 🎯 Vantagens do Sistema

### **1. Armazenamento Eficiente**
- **Formato Parquet**: Compressão até 80% menor que CSV
- **Particionamento**: Consultas rápidas por período
- **Schema tipado**: Validação automática de dados

### **2. Escalabilidade**
- **Processamento paralelo**: Múltiplas datas simultâneas
- **Incremental**: Processa apenas datas novas
- **Particionamento**: Suporta décadas de dados

### **3. Análise Otimizada**
- **Consultas rápidas**: Por data, estado ou indicador
- **Agregações eficientes**: GroupBy otimizado
- **Integração**: Pandas, Jupyter, BI tools

### **4. Robustez**
- **Retry automático**: Em caso de falhas de rede
- **Logs detalhados**: Para monitoramento
- **Validação**: Schema consistente

## 📊 Performance Estimada

Para **10 anos de dados diários**:

| Métrica | Estimativa |
|---------|-----------|
| Total de arquivos | ~18.250 arquivos |
| Tamanho aproximado | ~2-5 GB |
| Consulta 1 ano | < 2 segundos |
| Coleta 1 dia | 3-5 segundos |
| Processamento paralelo | 100 dias/minuto |

## 🔍 Monitoramento

### Ver estatísticas:
```bash
python main.py stats
```

### Verificar logs:
```bash
cat output/etl_logs.txt
```

### Estrutura de arquivos:
```
output/parquet/
├── indicadores_estados/
│   ├── 2024/
│   │   ├── 01/cotacoes_2024-01-15.parquet
│   │   ├── 02/cotacoes_2024-02-10.parquet
│   │   └── ...
│   └── 2025/
│       ├── 08/cotacoes_2025-08-20.parquet
│       └── ...
└── ...
```

## 🔄 Automação

### Crontab (Linux/Mac):
```bash
# Coleta diária às 18h
0 18 * * * cd /path/to/project && python main.py single --date $(date +%Y-%m-%d)

# Backfill semanal aos domingos
0 2 * * 0 cd /path/to/project && python main.py backfill
```

### Task Scheduler (Windows):
```powershell
# Criar tarefa diária
schtasks /create /tn "Cotacoes_Diaria" /tr "python main.py single --date $(Get-Date -f 'yyyy-MM-dd')" /sc daily /st 18:00
```

## 🚨 Troubleshooting

### **Erro de rede**
```python
# Dados não encontrados ou erro 404
# Verificar se a data é um dia útil (sem fins de semana/feriados)
```

### **Problema de dependências**
```bash
# Reinstalar dependências
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### **Arquivos corrompidos**
```bash
# Reprocessar com --force
python main.py single --date 2025-08-20 --force
```

## 📞 Suporte

Para dúvidas ou problemas:
1. Verificar logs em `output/etl_logs.txt`
2. Executar `python examples.py` para teste
3. Verificar se o site está acessível
4. Validar formato de datas (YYYY-MM-DD)

---

**Nota**: Este sistema foi desenvolvido para fins educacionais e de pesquisa. Respeite os termos de uso do site Notícias Agrícolas e implemente delays adequados entre requisições.
