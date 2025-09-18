from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import random
from typing import Dict, List, Any

NA_SCRAPER_URL = "https://www.noticiasagricolas.com.br/cotacoes/boi-gordo/{}"

def get_headers():
    """Retorna headers realistas para evitar bloqueios"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
        'Accept-Encoding': 'identity',  # Evitar compressão para debug
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }

def is_business_day(date_str: str) -> bool:
    """
    Verifica se a data é um dia útil (seg-sex, excluindo feriados brasileiros comuns)
    
    Args:
        date_str (str): Data no formato YYYY-MM-DD
    
    Returns:
        bool: True se for dia útil
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Verificar se é fim de semana (5=sábado, 6=domingo)
        if date_obj.weekday() >= 5:
            return False
        
        # Feriados brasileiros fixos mais comuns
        fixed_holidays = [
            (1, 1),   # Ano Novo
            (4, 21),  # Tiradentes
            (5, 1),   # Dia do Trabalhador
            (9, 7),   # Independência
            (10, 12), # Nossa Senhora Aparecida
            (11, 2),  # Finados
            (11, 15), # Proclamação da República
            (12, 25), # Natal
        ]
        
        month_day = (date_obj.month, date_obj.day)
        if month_day in fixed_holidays:
            return False
        
        return True
        
    except ValueError:
        return True  # Se não conseguir parsear, considera como dia útil

def has_potential_data(date_str: str) -> bool:
    """
    Verifica se a data tem potencial de ter dados (não muito antiga)
    
    Args:
        date_str (str): Data no formato YYYY-MM-DD
    
    Returns:
        bool: True se a data pode ter dados
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Dados de cotação geralmente disponíveis a partir de 2010
        min_date = datetime(2010, 1, 1)
        # Não processar datas futuras
        max_date = datetime.now() + timedelta(days=1)
        
        return min_date <= date_obj <= max_date
    except ValueError:
        return True

def scrap_ox_data(date_str: str, skip_non_business_days: bool = True) -> Dict[str, Any]:
    """
    Get all ox tables for a specific date

    Args:
        date_str (str): YYYY-MM-DD (ex: '2025-08-20')
        skip_non_business_days (bool): Se True, pula fins de semana e feriados
    
    Returns:
        Dict[str, Any]: Data dictionary with all extracted tables
    """
    try:
        # Verificação rápida de data inválida
        if not has_potential_data(date_str):
            return {
                'date': date_str,
                'error': 'Data muito antiga ou futura - sem dados esperados',
                'collected_at': datetime.now().isoformat(),
                'skipped': True
            }
        
        # Verificação de dia útil
        if skip_non_business_days and not is_business_day(date_str):
            return {
                'date': date_str,
                'error': 'Fim de semana ou feriado - sem dados esperados',
                'collected_at': datetime.now().isoformat(),
                'skipped': True
            }
        
        # Make request with proper headers
        url = NA_SCRAPER_URL.format(date_str)
        
        # Create session for better connection handling
        session = requests.Session()
        session.headers.update(get_headers())
        
        response = session.get(url, timeout=15)  # Reduzido timeout de 30 para 15
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # Dictionary to store all extracted tables
        data = {
            'date': date_str,
            'url': url,
            'collected_at': datetime.now().isoformat(),
            'tables': {}
        }
        
        # Search for divs with 'cotacao' class - this is where the data is
        cotacao_divs = soup.find_all('div', class_='cotacao')
        
        # Early exit se não encontrar divs de cotação
        if not cotacao_divs:
            # Delay mínimo apenas se não encontrar dados
            time.sleep(random.uniform(0.5, 1.0))
            return {
                'date': date_str,
                'error': 'Nenhuma div de cotação encontrada - sem dados disponíveis',
                'collected_at': datetime.now().isoformat(),
                'empty': True
            }
        
        found_data = False
        for div in cotacao_divs:
            # Get the title from the div content
            div_text = div.get_text(strip=True)
            # Extract title (first line usually contains the title)
            lines = [line.strip() for line in div_text.split('\n') if line.strip()]
            
            # Try to find a clean title - look for the first line that looks like a title
            section_title = 'Cotação'
            for line in lines[:3]:  # Check first 3 lines
                if any(keyword in line.lower() for keyword in ['boi', 'bezerro', 'vaca', 'novilha', 'indicador', 'reposição']):
                    # Clean up the title
                    section_title = line.split('Fonte:')[0].strip()
                    break
            
            # Find table within this div
            table = div.find('table')
            if not table:
                continue
            
            # Extract table data
            table_data = extract_table_data(table, section_title)
            if table_data:
                data['tables'][section_title] = table_data
                found_data = True
        
        # Delay apenas se encontrou dados válidos
        if found_data:
            time.sleep(random.uniform(1, 2))  # Reduzido de 1-3 para 1-2
        else:
            time.sleep(random.uniform(0.3, 0.7))  # Delay mínimo para dados vazios
        
        return data
        
    except Exception as e:
        return {
            'date': date_str,
            'error': str(e),
            'collected_at': datetime.now().isoformat()
        }

def extract_table_data(table, section_title: str) -> List[Dict[str, Any]]:
    """
    Extrai dados de uma tabela HTML
    
    Args:
        table: Elemento BeautifulSoup da tabela
        section_title (str): Título da seção
    
    Returns:
        List[Dict[str, Any]]: Lista de dicionários com os dados da tabela
    """
    try:
        rows = table.find_all('tr')
        if not rows:
            return []
        
        data = []
        headers = []
        
        for i, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            if not cells:
                continue
            
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            # Primeira linha não vazia pode ser cabeçalho
            if i == 0 or not headers:
                # Verificar se parece com cabeçalho
                if any(text.lower() in ['data', 'preço', 'variação', 'estado', 'região'] for text in cell_texts):
                    headers = cell_texts
                    continue
            
            # Pular linhas vazias
            if not any(cell_texts):
                continue
            
            # Criar dicionário da linha
            if headers:
                row_dict = {}
                for j, value in enumerate(cell_texts):
                    header = headers[j] if j < len(headers) else f'column_{j}'
                    row_dict[header] = value
            else:
                # Sem cabeçalho identificado, usar índices
                row_dict = {f'column_{j}': value for j, value in enumerate(cell_texts)}
            
            row_dict['section'] = section_title
            data.append(row_dict)
        
        return data
        
    except Exception as e:
        return [{'error': str(e), 'section': section_title}]

def scrap_multiple_dates(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Coleta dados para múltiplas datas
    
    Args:
        start_date (str): Data inicial no formato YYYY-MM-DD
        end_date (str): Data final no formato YYYY-MM-DD
    
    Returns:
        List[Dict[str, Any]]: Lista com dados de todas as datas
    """
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current = start
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        print(f"Coletando dados para {date_str}...")
        
        data = scrap_ox_data(date_str)
        all_data.append(data)
        
        current += timedelta(days=1)
    
    return all_data

def save_data_to_csv(data: List[Dict[str, Any]], output_file: str):
    """
    Salva os dados coletados em arquivo CSV
    
    Args:
        data (List[Dict[str, Any]]): Dados coletados
        output_file (str): Caminho do arquivo de saída
    """
    all_rows = []
    
    for day_data in data:
        if 'error' in day_data:
            continue
            
        date = day_data.get('date', '')
        
        for section_name, table_data in day_data.get('tables', {}).items():
            for row in table_data:
                row['date'] = date
                row['section_name'] = section_name
                all_rows.append(row)
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Dados salvos em: {output_file}")
    else:
        print("Nenhum dado encontrado para salvar.")

# Exemplo de uso
if __name__ == "__main__":
    # Coletar dados de uma data específica
    single_date_data = scrap_ox_data("2025-08-20")
    print(json.dumps(single_date_data, indent=2, ensure_ascii=False))
    
    # Ou coletar dados de múltiplas datas
    # multiple_dates_data = scrap_multiple_dates("2025-08-15", "2025-08-20")
    # save_data_to_csv(multiple_dates_data, "cotacoes_boi_gordo.csv")

