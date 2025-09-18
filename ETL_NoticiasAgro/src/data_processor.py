import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re

class CotacaoDataProcessor:
    """
    Processador otimizado para dados de cotações com armazenamento eficiente
    """
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Criar estrutura de diretórios
        self.parquet_dir = self.base_output_dir / "parquet"
        self.parquet_dir.mkdir(exist_ok=True)
        
        # Schemas padronizados para cada tipo de tabela
        self.schemas = {
            'indicadores_simples': ['date', 'indicator_name', 'price_brl', 'variation_pct', 'price_usd'],
            'indicadores_estados': ['date', 'indicator_name', 'state', 'price_brl', 'variation_pct', 'price_usd'],
            'contratos_futuros': ['date', 'contract_month', 'price', 'variation', 'indicator_name'],
            'reposicao': ['date', 'state', 'category', 'desmama', 'bezerra_bezerro', 'novilha_garrote', 'vaca_boi_magro', 'indicator_name'],
            'mercados_externos': ['date', 'market', 'contract', 'price', 'variation']
        }
    
    def clean_numeric_value(self, value: str) -> Optional[float]:
        """Limpa e converte valores numéricos"""
        if not value or value == '' or 'Ver histórico' in value or 'Atualizado' in value:
            return None
        
        # Converter para string
        clean_value = str(value).strip()
        
        # Remover sinais de + no início
        clean_value = clean_value.lstrip('+')
        
        # Para números brasileiros (ex: 1.246,72), trocar vírgula por ponto
        # e remover pontos de milhares se houver vírgula
        if ',' in clean_value:
            # Se tem vírgula, os pontos são separadores de milhares
            clean_value = clean_value.replace('.', '').replace(',', '.')
        
        # Remover caracteres não numéricos exceto ponto e sinal de menos
        clean_value = re.sub(r'[^\d.-]', '', clean_value)
        
        try:
            return float(clean_value)
        except (ValueError, TypeError):
            return None
    
    def classify_table(self, table_name: str, sample_data: List[Dict]) -> str:
        """Classifica o tipo de tabela baseado no nome e estrutura"""
        table_name_lower = table_name.lower()
        
        if 'reposição' in table_name_lower or 'reposicao' in table_name_lower:
            return 'reposicao'
        elif 'chicago' in table_name_lower or 'new york' in table_name_lower:
            return 'mercados_externos'
        elif 'pregão regular' in table_name_lower or ('pregão' in table_name_lower and 'b3' in table_name_lower):
            return 'contratos_futuros'
        elif (any('estado' in str(row.get('Estado', '')).lower() for row in sample_data if row.get('Estado')) or
              any('município' in str(row.get('column_0', '')).lower() or 'municipio' in str(row.get('column_0', '')).lower() or 'munípicio' in str(row.get('column_0', '')).lower() for row in sample_data if row.get('column_0'))):
            return 'indicadores_estados'
        else:
            # Indicadores simples (inclui "Indicador do Boi Gordo Esalq / B3")
            return 'indicadores_simples'
    
    def process_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Processa dados brutos em DataFrames estruturados"""
        processed_dfs = {}
        date = raw_data.get('date', '')
        
        for table_name, table_data in raw_data.get('tables', {}).items():
            if not table_data or not isinstance(table_data, list):
                continue
            
            # Filtrar linhas válidas
            valid_rows = [row for row in table_data if self._is_valid_row(row)]
            if not valid_rows:
                continue
            
            table_type = self.classify_table(table_name, valid_rows)
            df = self._process_by_type(table_name, valid_rows, date, table_type)
            
            if df is not None and not df.empty:
                processed_dfs[f"{table_type}_{table_name}"] = df
        
        return processed_dfs
    
    def _is_valid_row(self, row: Dict) -> bool:
        """Verifica se a linha contém dados válidos"""
        if not isinstance(row, dict):
            return False
        
        # Filtrar linhas com informações de atualização
        text_content = str(row)
        invalid_indicators = [
            'Ver histórico', 'Atualizado em:', 'Última atualização',
            'ponderada considerando', 'Desmama:', 'Peso'
        ]
        
        return not any(indicator in text_content for indicator in invalid_indicators)
    
    def _process_by_type(self, table_name: str, data: List[Dict], date: str, table_type: str) -> Optional[pd.DataFrame]:
        """Processa dados baseado no tipo da tabela"""
        
        if table_type == 'indicadores_simples':
            return self._process_indicadores_simples(table_name, data, date)
        elif table_type == 'indicadores_estados':
            return self._process_indicadores_estados(table_name, data, date)
        elif table_type == 'contratos_futuros':
            return self._process_contratos_futuros(table_name, data, date)
        elif table_type == 'reposicao':
            return self._process_reposicao(table_name, data, date)
        elif table_type == 'mercados_externos':
            return self._process_mercados_externos(table_name, data, date)
        
        return None
    
    def _process_indicadores_simples(self, table_name: str, data: List[Dict], date: str) -> pd.DataFrame:
        """Processa indicadores simples (sem estado)"""
        rows = []
        
        for row in data:
            price_brl = None
            variation_pct = None
            price_usd = None
            
            # Extrair valores baseado nas chaves disponíveis
            for key, value in row.items():
                key_lower = key.lower()
                if ('r$' in key_lower or 'vista' in key_lower or 'prazo' in key_lower or 
                    'valor' in key_lower or 'preço' in key_lower or 'preco' in key_lower):
                    price_brl = self.clean_numeric_value(value)
                elif 'variação' in key_lower or 'variacao' in key_lower or '(%)' in key:
                    variation_pct = self.clean_numeric_value(value)
                elif 'u$' in key_lower or 'us$' in key_lower:
                    price_usd = self.clean_numeric_value(value)
            
            if price_brl is not None:
                rows.append({
                    'date': date,
                    'indicator_name': table_name,
                    'price_brl': price_brl,
                    'variation_pct': variation_pct,
                    'price_usd': price_usd
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.schemas['indicadores_simples'])
    
    def _process_indicadores_estados(self, table_name: str, data: List[Dict], date: str) -> pd.DataFrame:
        """Processa indicadores por estado/município"""
        rows = []
        
        for row in data:
            # Tentar pegar Estado ou Município (column_0 para IMEA)
            state = row.get('Estado', '') or row.get('column_0', '')
            if not state or not isinstance(state, str) or state.lower() in ['município', 'municipio', 'munípicio']:
                continue
            
            price_brl = None
            variation_pct = None
            price_usd = None
            
            for key, value in row.items():
                key_lower = key.lower()
                if ('r$' in key_lower or 'preço' in key_lower or 'preco' in key_lower or 
                    'média' in key_lower or 'media' in key_lower or 'column_1' in key_lower):
                    price_brl = self.clean_numeric_value(value)
                elif ('variação' in key_lower or 'variacao' in key_lower or 'column_2' in key_lower):
                    variation_pct = self.clean_numeric_value(value)
                elif 'u$' in key_lower or 'us$' in key_lower:
                    price_usd = self.clean_numeric_value(value)
            
            if price_brl is not None:
                rows.append({
                    'date': date,
                    'indicator_name': table_name,
                    'state': state,
                    'price_brl': price_brl,
                    'variation_pct': variation_pct,
                    'price_usd': price_usd
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.schemas['indicadores_estados'])
    
    def _process_contratos_futuros(self, table_name: str, data: List[Dict], date: str) -> pd.DataFrame:
        """Processa contratos futuros"""
        rows = []
        
        for row in data:
            contract_month = None
            price = None
            variation = None
            
            # Identificar coluna do contrato/mês
            for key, value in row.items():
                if 'column_0' in key or 'Contrato' in key:
                    if '/' in str(value):  # Formato Mês/Ano
                        contract_month = value
                elif 'column_1' in key or 'Fechamento' in key or 'R$' in key:
                    price = self.clean_numeric_value(value)
                elif 'column_2' in key or 'Variação' in key:
                    variation = self.clean_numeric_value(value)
            
            if contract_month and price is not None:
                rows.append({
                    'date': date,
                    'contract_month': contract_month,
                    'price': price,
                    'variation': variation,
                    'indicator_name': table_name
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.schemas['contratos_futuros'])
    
    def _process_reposicao(self, table_name: str, data: List[Dict], date: str) -> pd.DataFrame:
        """Processa dados de reposição"""
        rows = []
        
        for row in data:
            state = row.get('Estado', '')
            if not state or not isinstance(state, str):
                continue
            
            category = 'Fêmea' if 'Fêmea' in table_name else 'Macho'
            
            desmama = self.clean_numeric_value(row.get('Desmama', ''))
            bezerra_bezerro = self.clean_numeric_value(row.get('Bezerra', row.get('Bezerro', '')))
            novilha_garrote = self.clean_numeric_value(row.get('Novilha', row.get('Garrote', '')))
            vaca_boi_magro = self.clean_numeric_value(row.get('Vaca Magra', row.get('Boi Magro', '')))
            
            rows.append({
                'date': date,
                'state': state,
                'category': category,
                'desmama': desmama,
                'bezerra_bezerro': bezerra_bezerro,
                'novilha_garrote': novilha_garrote,
                'vaca_boi_magro': vaca_boi_magro,
                'indicator_name': table_name
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.schemas['reposicao'])
    
    def _process_mercados_externos(self, table_name: str, data: List[Dict], date: str) -> pd.DataFrame:
        """Processa dados de mercados externos"""
        rows = []
        
        for row in data:
            contract = None
            price = None
            variation = None
            
            for key, value in row.items():
                if 'CONTRATO' in key or 'column_0' in key:
                    if not any(word in str(value).lower() for word in ['última', 'atualização']):
                        contract = value
                elif 'PREÇO' in key or 'column_1' in key or 'US$' in key:
                    price = self.clean_numeric_value(value)
                elif 'VAR' in key or 'column_2' in key:
                    variation = self.clean_numeric_value(value)
            
            if contract and price is not None:
                rows.append({
                    'date': date,
                    'market': table_name,
                    'contract': contract,
                    'price': price,
                    'variation': variation
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.schemas['mercados_externos'])
    
    def clean_table_name_for_filename(self, table_name: str) -> str:
        """Limpa o nome da tabela para usar como nome de arquivo"""
        import re
        
        # Remover caracteres especiais e substituir por underscore
        clean_name = re.sub(r'[^\w\s-]', '', table_name)  # Remove caracteres especiais exceto espaços e hífens
        clean_name = re.sub(r'[\s-]+', '_', clean_name)   # Substitui espaços e hífens por underscore
        clean_name = clean_name.strip('_')                # Remove underscores no início e fim
        clean_name = clean_name.lower()                   # Converte para minúsculas
        
        # Limitar o tamanho do nome para evitar problemas no sistema de arquivos
        if len(clean_name) > 50:
            clean_name = clean_name[:50].rstrip('_')
        
        return clean_name

    def save_to_parquet(self, processed_dfs: Dict[str, pd.DataFrame], date: str, verbose: bool = False):
        """Salva DataFrames em arquivos Parquet particionados"""
        year = date[:4]
        month = date[5:7]
        
        for table_key, df in processed_dfs.items():
            if df.empty:
                continue
            
            # Extrair nome da tabela original do table_key
            # table_key formato: "tipo_NomeDaTabela"
            table_name_original = '_'.join(table_key.split('_')[2:]) if len(table_key.split('_')) > 2 else table_key.split('_')[1]
            
            # Limpar nome da tabela para usar como nome de pasta
            clean_table_name = self.clean_table_name_for_filename(table_name_original)
            
            # Criar estrutura de diretórios: tipo/ano/mês/nome_da_tabela
            table_type = table_key.split('_')[0]
            output_dir = self.parquet_dir / table_type / year / month / clean_table_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo simples com apenas a data
            filename = f"cotacoes_{date}.parquet"
            filepath = output_dir / filename
            
            # Salvar como Parquet
            df.to_parquet(filepath, index=False)
            if verbose:
                print(f"Salvo: {filepath} ({len(df)} registros)")
    
    def load_date_range(self, start_date: str, end_date: str, table_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Carrega dados de um intervalo de datas"""
        if table_types is None:
            table_types = list(self.schemas.keys())
        
        all_dfs = {}
        
        for table_type in table_types:
            type_dir = self.parquet_dir / table_type
            if not type_dir.exists():
                continue
            
            dfs = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date <= end_date_dt:
                year = current_date.strftime('%Y')
                month = current_date.strftime('%m')
                date_str = current_date.strftime('%Y-%m-%d')
                
                file_pattern = type_dir / year / month / f"cotacoes_{date_str}.parquet"
                
                if file_pattern.exists():
                    df = pd.read_parquet(file_pattern)
                    dfs.append(df)
                
                current_date += timedelta(days=1)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                all_dfs[table_type] = combined_df
        
        return all_dfs
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de armazenamento"""
        stats = {}
        
        # Mapear tipos de diretório para schemas
        directory_types = {
            'indicadores': ['indicadores_simples', 'indicadores_estados'],
            'contratos': ['contratos_futuros'],
            'reposicao': ['reposicao'],
            'mercados': ['mercados_externos']
        }
        
        # Buscar diretórios reais
        for dir_name in ['indicadores', 'contratos', 'reposicao', 'mercados']:
            type_dir = self.parquet_dir / dir_name
            if not type_dir.exists():
                continue
            
            file_count = 0
            total_size = 0
            
            for parquet_file in type_dir.rglob("*.parquet"):
                file_count += 1
                total_size += parquet_file.stat().st_size
            
            if file_count > 0:
                stats[dir_name] = {
                    'files': file_count,
                    'size_mb': round(total_size / (1024 * 1024), 2)
                }
        
        return stats

# Exemplo de uso
if __name__ == "__main__":
    # Teste com dados de exemplo
    processor = CotacaoDataProcessor("../output")
    
    # Simular processamento de um dia
    with open("example_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    processed_dfs = processor.process_raw_data(raw_data)
    processor.save_to_parquet(processed_dfs, raw_data['date'])
    
    print(f"\nEstatísticas de armazenamento:")
    stats = processor.get_storage_stats()
    for table_type, info in stats.items():
        print(f"{table_type}: {info['files']} arquivos, {info['size_mb']} MB")
