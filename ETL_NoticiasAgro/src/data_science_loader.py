"""
Módulo otimizado para carregamento de dados em projetos de ciência de dados
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataScience_Loader:
    """
    Carregador otimizado de dados Parquet para projetos de ciência de dados
    """
    
    def __init__(self, parquet_base_path: str):
        """
        Inicializa o carregador
        
        Args:
            parquet_base_path: Caminho base dos arquivos Parquet
        """
        self.base_path = Path(parquet_base_path)
        self.available_tables = self._discover_tables()
        
    def _discover_tables(self) -> List[str]:
        """Descobre tabelas disponíveis"""
        tables = []
        if self.base_path.exists():
            for table_dir in self.base_path.iterdir():
                if table_dir.is_dir():
                    tables.append(table_dir.name)
        return sorted(tables)
    
    def load_time_series(
        self,
        start_date: str,
        end_date: str,
        table_type: str = "indicadores_estados",
        states: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        resample_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carrega dados como série temporal otimizada para análise
        
        Args:
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            table_type: Tipo de tabela
            states: Lista de estados para filtrar
            indicators: Lista de indicadores para filtrar
            columns: Colunas específicas para carregar
            resample_freq: Frequência para reamostragem ('D', 'W', 'M')
        
        Returns:
            DataFrame otimizado para análise de séries temporais
        """
        
        # Carregar dados base
        df = self.load_data_range(
            start_date=start_date,
            end_date=end_date,
            table_types=[table_type],
            columns=columns
        )
        
        if df.empty or table_type not in df:
            return pd.DataFrame()
        
        data = df[table_type].copy()
        
        # Converter data para datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Filtros
        if states and 'state' in data.columns:
            data = data[data['state'].isin(states)]
        
        if indicators and 'indicator_name' in data.columns:
            data = data[data['indicator_name'].isin(indicators)]
        
        # Configurar índice temporal
        data = data.set_index('date').sort_index()
        
        # Reamostragem se solicitada
        if resample_freq and 'price_brl' in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data = data.groupby(['state', 'indicator_name'])[numeric_cols].resample(resample_freq).mean()
            data = data.reset_index()
        
        return data
    
    def load_data_range(
        self,
        start_date: str,
        end_date: str,
        table_types: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Carregamento otimizado para intervalos grandes de datas
        
        Args:
            start_date: Data inicial
            end_date: Data final  
            table_types: Tipos de tabelas para carregar
            columns: Colunas específicas
            parallel: Usar processamento paralelo
        
        Returns:
            Dicionário com DataFrames por tipo de tabela
        """
        
        if table_types is None:
            table_types = self.available_tables
        
        all_data = {}
        
        for table_type in table_types:
            if table_type not in self.available_tables:
                warnings.warn(f"Tabela {table_type} não encontrada")
                continue
            
            # Descobrir arquivos disponíveis no período
            files = self._discover_files_in_range(table_type, start_date, end_date)
            
            if not files:
                continue
            
            # Carregar arquivos
            if parallel and len(files) > 1:
                dfs = self._load_files_parallel(files, columns)
            else:
                dfs = self._load_files_sequential(files, columns)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                # Filtrar datas exatas
                combined = self._filter_date_range(combined, start_date, end_date)
                all_data[table_type] = combined
        
        return all_data
    
    def _discover_files_in_range(self, table_type: str, start_date: str, end_date: str) -> List[Path]:
        """Descobre arquivos no intervalo de datas"""
        files = []
        table_path = self.base_path / table_type
        
        if not table_path.exists():
            return files
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Iterar por anos e meses
        current = start.replace(day=1)  # Primeiro dia do mês
        while current <= end:
            year_dir = table_path / current.strftime('%Y')
            month_dir = year_dir / current.strftime('%m')
            
            if month_dir.exists():
                # Buscar arquivos parquet no mês
                for file_path in month_dir.glob('*.parquet'):
                    files.append(file_path)
            
            # Próximo mês
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return sorted(files)
    
    def _load_files_parallel(self, files: List[Path], columns: Optional[List[str]]) -> List[pd.DataFrame]:
        """Carrega arquivos em paralelo"""
        dfs = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self._load_single_file, file_path, columns): file_path 
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                df = future.result()
                if df is not None and not df.empty:
                    dfs.append(df)
        
        return dfs
    
    def _load_files_sequential(self, files: List[Path], columns: Optional[List[str]]) -> List[pd.DataFrame]:
        """Carrega arquivos sequencialmente"""
        dfs = []
        for file_path in files:
            df = self._load_single_file(file_path, columns)
            if df is not None and not df.empty:
                dfs.append(df)
        return dfs
    
    def _load_single_file(self, file_path: Path, columns: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """Carrega um único arquivo Parquet"""
        try:
            if columns:
                # Verificar se todas as colunas existem
                available_columns = pd.read_parquet(file_path, nrows=0).columns.tolist()
                valid_columns = [col for col in columns if col in available_columns]
                if valid_columns:
                    return pd.read_parquet(file_path, columns=valid_columns)
                else:
                    return pd.read_parquet(file_path)
            else:
                return pd.read_parquet(file_path)
        except Exception as e:
            warnings.warn(f"Erro ao carregar {file_path}: {e}")
            return None
    
    def _filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filtra DataFrame por intervalo de datas"""
        if 'date' not in df.columns:
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        return df[(df['date'] >= start) & (df['date'] <= end)]
    
    def create_pivot_table(
        self,
        data: pd.DataFrame,
        value_col: str = 'price_brl',
        index_col: str = 'date',
        columns_col: str = 'state'
    ) -> pd.DataFrame:
        """
        Cria tabela pivot otimizada para análise
        
        Args:
            data: DataFrame de entrada
            value_col: Coluna de valores
            index_col: Coluna para índice (geralmente data)
            columns_col: Coluna para colunas da pivot
        
        Returns:
            DataFrame em formato pivot
        """
        
        if not all(col in data.columns for col in [value_col, index_col, columns_col]):
            raise ValueError("Colunas especificadas não encontradas no DataFrame")
        
        # Converter data se necessário
        if index_col == 'date':
            data[index_col] = pd.to_datetime(data[index_col])
        
        # Criar pivot
        pivot = data.pivot_table(
            values=value_col,
            index=index_col,
            columns=columns_col,
            aggfunc='mean'  # Média caso haja duplicatas
        )
        
        # Limpar nomes
        pivot.columns.name = None
        
        return pivot
    
    def get_data_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os dados disponíveis"""
        info = {
            'available_tables': self.available_tables,
            'base_path': str(self.base_path),
            'table_details': {}
        }
        
        for table_type in self.available_tables:
            table_path = self.base_path / table_type
            file_count = len(list(table_path.rglob('*.parquet')))
            
            # Tentar carregar um arquivo de amostra para ver schema
            sample_files = list(table_path.rglob('*.parquet'))
            if sample_files:
                try:
                    sample = pd.read_parquet(sample_files[0], nrows=1)
                    columns = sample.columns.tolist()
                    dtypes = sample.dtypes.to_dict()
                except:
                    columns = []
                    dtypes = {}
            else:
                columns = []
                dtypes = {}
            
            info['table_details'][table_type] = {
                'file_count': file_count,
                'columns': columns,
                'dtypes': {str(k): str(v) for k, v in dtypes.items()}
            }
        
        return info
    
    def create_features_dataframe(
        self,
        data: pd.DataFrame,
        price_col: str = 'price_brl',
        date_col: str = 'date',
        group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Cria features básicas para machine learning
        
        Args:
            data: DataFrame de entrada
            price_col: Coluna de preços
            date_col: Coluna de data
            group_cols: Colunas para agrupamento
        
        Returns:
            DataFrame com features adicionais
        """
        
        if group_cols is None:
            group_cols = ['state', 'indicator_name'] if 'state' in data.columns else []
        
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Ordenar por data
        df = df.sort_values([*group_cols, date_col])
        
        # Features por grupo
        for group in group_cols if group_cols else [None]:
            if group:
                grouped = df.groupby(group)[price_col]
            else:
                grouped = df[price_col]
            
            # Lags
            df[f'{price_col}_lag_1'] = grouped.shift(1)
            df[f'{price_col}_lag_7'] = grouped.shift(7)
            
            # Médias móveis
            df[f'{price_col}_ma_7'] = grouped.rolling(7).mean()
            df[f'{price_col}_ma_30'] = grouped.rolling(30).mean()
            
            # Volatilidade
            df[f'{price_col}_vol_7'] = grouped.rolling(7).std()
            df[f'{price_col}_vol_30'] = grouped.rolling(30).std()
            
            # Retornos
            df[f'{price_col}_ret_1'] = grouped.pct_change(1)
            df[f'{price_col}_ret_7'] = grouped.pct_change(7)
        
        # Features temporais
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        
        return df


# Funções de conveniência
def load_for_forecasting(
    parquet_path: str,
    target_state: str = 'São Paulo',
    target_indicator: str = 'Indicador do Boi',
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Carrega dados otimizados para forecasting
    
    Args:
        parquet_path: Caminho dos dados Parquet
        target_state: Estado alvo
        target_indicator: Indicador alvo
        lookback_days: Dias para trás a partir de hoje
    
    Returns:
        DataFrame pronto para forecasting
    """
    
    loader = DataScience_Loader(parquet_path)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Carregar dados
    ts_data = loader.load_time_series(
        start_date=start_date,
        end_date=end_date,
        table_type="indicadores_estados",
        states=[target_state],
        indicators=[target_indicator]
    )
    
    if ts_data.empty:
        return pd.DataFrame()
    
    # Criar pivot simples (data x preço)
    if 'state' in ts_data.columns:
        ts_data = ts_data[ts_data['state'] == target_state]
    
    # Simplificar para série temporal univariada
    result = ts_data[['price_brl']].copy()
    result = result.groupby(result.index).mean()  # Média se houver duplicatas
    
    return result


def load_for_ml_classification(
    parquet_path: str,
    target_col: str = 'price_direction',
    lookback_days: int = 730
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega e prepara dados para classificação ML
    
    Args:
        parquet_path: Caminho dos dados
        target_col: Nome da coluna target a ser criada
        lookback_days: Dias históricos
    
    Returns:
        Tuple (features_df, target_series)
    """
    
    loader = DataScience_Loader(parquet_path)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # Carregar dados principais
    data = loader.load_data_range(
        start_date=start_date,
        end_date=end_date,
        table_types=["indicadores_estados"]
    )
    
    if 'indicadores_estados' not in data:
        return pd.DataFrame(), pd.Series()
    
    df = data['indicadores_estados']
    
    # Criar features
    features_df = loader.create_features_dataframe(df)
    
    # Criar target (direção do preço: sobe=1, desce=0)
    features_df = features_df.sort_values(['state', 'indicator_name', 'date'])
    features_df[target_col] = (
        features_df.groupby(['state', 'indicator_name'])['price_brl']
        .shift(-1) > features_df['price_brl']
    ).astype(int)
    
    # Remover linhas com NaN
    clean_df = features_df.dropna()
    
    if clean_df.empty:
        return pd.DataFrame(), pd.Series()
    
    # Separar features e target
    feature_cols = [col for col in clean_df.columns if col not in ['date', target_col, 'state', 'indicator_name']]
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    
    return X, y


# Exemplo de uso
if __name__ == "__main__":
    # Inicializar loader
    loader = DataScience_Loader("../output/parquet")
    
    # Ver informações disponíveis
    info = loader.get_data_info()
    print("Dados disponíveis:")
    for table, details in info['table_details'].items():
        print(f"  {table}: {details['file_count']} arquivos")
    
    # Carregar dados de SP para forecasting
    sp_data = load_for_forecasting(
        parquet_path="../output/parquet",
        target_state="São Paulo",
        lookback_days=90
    )
    
    if not sp_data.empty:
        print(f"\nDados SP carregados: {len(sp_data)} registros")
        print(f"Período: {sp_data.index.min()} até {sp_data.index.max()}")
        print(f"Preço médio: R$ {sp_data['price_brl'].mean():.2f}")
