#!/usr/bin/env python3
"""
Script principal para coleta e processamento de dados de cotações
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import scrap_ox_data, CotacaoDataProcessor
from src.ui_utils import ProgressReporter, format_error_message

class ETLPipeline:
    """Pipeline ETL para dados de cotações"""
    
    def __init__(self, output_dir: str = "output", max_workers: int = 3, stop_on_error: bool = True):
        self.processor = CotacaoDataProcessor(output_dir)
        self.max_workers = max_workers
        self.stop_on_error = stop_on_error
        self.reporter = ProgressReporter()
        self.logs = []
        self.total_records_processed = 0
    
    def _collect_single_date_silent(self, date_str: str) -> tuple[bool, int, str]:
        """Coleta dados de uma data sem interface visual (para spinner único)"""
        try:
            # Coletar dados com otimizações
            raw_data = scrap_ox_data(date_str, skip_non_business_days=True)
            
            # Verificar se foi pulado por ser fim de semana/feriado
            if raw_data.get('skipped'):
                self.logs.append(f"PULADO {date_str}: {raw_data['error']}")
                return True, 0, "Pulado (fim de semana/feriado)"
            
            # Verificar se está vazio mas é válido
            if raw_data.get('empty'):
                self.logs.append(f"VAZIO {date_str}: Sem dados disponíveis")
                return True, 0, "Sem dados disponíveis"
            
            if 'error' in raw_data and not raw_data.get('skipped') and not raw_data.get('empty'):
                error_msg = format_error_message(Exception(raw_data['error']))
                self.logs.append(f"ERRO {date_str}: {raw_data['error']}")
                return False, 0, error_msg
            
            # Processar e salvar apenas se houver dados
            if raw_data.get('tables'):
                processed_dfs = self.processor.process_raw_data(raw_data)
                
                # Verificar se realmente processou dados
                total_records = sum(len(df) for df in processed_dfs.values())
                
                if total_records > 0:
                    self.processor.save_to_parquet(processed_dfs, date_str)
                    self.total_records_processed += total_records
                    
                    success_msg = f"{len(processed_dfs)} tables processed"
                    self.logs.append(f"SUCESSO {date_str}: {len(processed_dfs)} tabelas, {total_records} registros")
                    
                    return True, total_records, success_msg
                else:
                    # Coletou tabelas mas não processou registros
                    print(f"⚠️ {date_str}: Tabelas encontradas mas 0 registros processados")
                    self.logs.append(f"PROBLEMA {date_str}: Tabelas encontradas mas 0 registros processados")
                    return True, 0, "Tabelas encontradas mas não processadas"
            else:
                self.logs.append(f"VAZIO {date_str}: Sem tabelas encontradas")
                return True, 0, "Sem tabelas encontradas"
            
        except Exception as e:
            error_msg = format_error_message(e)
            self.logs.append(f"ERRO {date_str}: {str(e)}")
            return False, 0, error_msg
    
    def collect_single_date(self, date_str: str, current: int = 1, total: int = 1) -> tuple[bool, int, str]:
        """Coleta e processa dados de uma data específica"""
        try:
            # Iniciar spinner para esta data
            self.reporter.start_date_processing(date_str, current, total)
            
            # Coletar dados com otimizações
            raw_data = scrap_ox_data(date_str, skip_non_business_days=True)
            
            # Verificar se foi pulado por ser fim de semana/feriado
            if raw_data.get('skipped'):
                self.logs.append(f"PULADO {date_str}: {raw_data['error']}")
                self.reporter.finish_date_processing(date_str, True, "Pulado (fim de semana/feriado)")
                return True, 0, "Pulado (fim de semana/feriado)"
            
            # Verificar se está vazio mas é válido
            if raw_data.get('empty'):
                self.logs.append(f"VAZIO {date_str}: Sem dados disponíveis")
                self.reporter.finish_date_processing(date_str, True, "Sem dados disponíveis")
                return True, 0, "Sem dados disponíveis"
            
            if 'error' in raw_data and not raw_data.get('skipped') and not raw_data.get('empty'):
                error_msg = format_error_message(Exception(raw_data['error']))
                self.logs.append(f"ERRO {date_str}: {raw_data['error']}")
                self.reporter.finish_date_processing(date_str, False, error_msg)
                return False, 0, error_msg
            
            # Processar e salvar apenas se houver dados
            if raw_data.get('tables'):
                processed_dfs = self.processor.process_raw_data(raw_data)
                
                # Verificar se realmente processou dados
                total_records = sum(len(df) for df in processed_dfs.values())
                
                if total_records > 0:
                    self.processor.save_to_parquet(processed_dfs, date_str)
                    self.total_records_processed += total_records
                    
                    success_msg = f"{len(processed_dfs)} tables processed"
                    self.logs.append(f"SUCESSO {date_str}: {len(processed_dfs)} tabelas, {total_records} registros")
                    self.reporter.finish_date_processing(date_str, True, success_msg, total_records)
                    
                    return True, total_records, success_msg
                else:
                    # Coletou tabelas mas não processou registros
                    problem_msg = "Tabelas encontradas mas não processadas"
                    print(f"⚠️ {date_str}: {problem_msg}")
                    self.logs.append(f"PROBLEMA {date_str}: {problem_msg}")
                    self.reporter.finish_date_processing(date_str, True, problem_msg)
                    return True, 0, problem_msg
            else:
                self.logs.append(f"VAZIO {date_str}: Sem tabelas encontradas")
                self.reporter.finish_date_processing(date_str, True, "Sem tabelas encontradas")
                return True, 0, "Sem tabelas encontradas"
            
        except Exception as e:
            error_msg = format_error_message(e)
            self.logs.append(f"ERRO {date_str}: {str(e)}")
            self.reporter.finish_date_processing(date_str, False, error_msg)
            return False, 0, error_msg
    
    def collect_date_range(self, start_date: str, end_date: str, skip_existing: bool = True) -> None:
        """Coleta dados para um intervalo de datas com processamento sequencial e stop-on-error"""
        
        # Importar função de validação
        from src.scraper import is_business_day, has_potential_data
        
        # Gerar lista de datas otimizada
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Pular datas que não têm potencial de dados
            if not has_potential_data(date_str):
                current += timedelta(days=1)
                continue
            
            # Verificar se já existe
            if skip_existing and self._file_exists(date_str):
                current += timedelta(days=1)
                continue
            
            dates.append(date_str)
            current += timedelta(days=1)
        
        if not dates:
            print("🎯 No new dates to process.")
            return
        
        # Filtrar apenas dias úteis se configurado
        business_days = [date for date in dates if is_business_day(date)]
        non_business_days = len(dates) - len(business_days)
        
        if non_business_days > 0:
            print(f"📅 Pulando {non_business_days} fins de semana/feriados automaticamente")
        
        # Usar apenas dias úteis para processamento
        dates_to_process = business_days
        
        if not dates_to_process:
            print("🎯 No business days to process.")
            return
        
        # Iniciar spinner único para todo o período
        self.reporter.start_range_execution(start_date, end_date, len(dates_to_process))
        
        # Processar sequencialmente com stop-on-error
        success_count = 0
        for i, date_str in enumerate(dates_to_process, 1):
            # Processar data silenciosamente (sem spinner individual)
            success, records, details = self._collect_single_date_silent(date_str)
            
            if success:
                success_count += 1
                self.reporter.update_progress(date_str, True, details)
            else:
                if self.stop_on_error:
                    # Parar imediatamente em caso de erro
                    self.reporter.report_error_and_stop(date_str, details)
                    self._save_logs()
                    return
                else:
                    self.reporter.update_progress(date_str, False, details)
        
        # Relatório final
        self.reporter.finish_execution(success_count, len(dates_to_process), self.total_records_processed)
        self._save_logs()
        self._print_stats()
        
    def collect_date_range_parallel(self, start_date: str, end_date: str, skip_existing: bool = True) -> None:
        """Coleta dados para um intervalo de datas com processamento paralelo otimizado"""
        
        # Importar função de validação
        from src.scraper import is_business_day, has_potential_data
        
        # Gerar lista de datas otimizada
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Pular datas que não têm potencial de dados
            if not has_potential_data(date_str):
                current += timedelta(days=1)
                continue
            
            # Verificar se já existe
            if skip_existing and self._file_exists(date_str):
                current += timedelta(days=1)
                continue
            
            dates.append(date_str)
            current += timedelta(days=1)
        
        if not dates:
            print("🎯 No new dates to process.")
            return
        
        # Filtrar apenas dias úteis
        business_days = [date for date in dates if is_business_day(date)]
        non_business_days = len(dates) - len(business_days)
        
        if non_business_days > 0:
            print(f"� Pulando {non_business_days} fins de semana/feriados automaticamente")
        
        dates_to_process = business_days
        
        if not dates_to_process:
            print("🎯 No business days to process.")
            return
        
        print(f"🚀 Processing {len(dates_to_process)} business days in parallel mode...")
        
        # Processar em paralelo com workers otimizados
        success_count = 0
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(dates_to_process))) as executor:
            # Submeter tarefas
            future_to_date = {
                executor.submit(self._collect_single_date_optimized, date): date 
                for date in dates_to_process
            }
            
            # Processar resultados
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    success, records, details = future.result()
                    if success:
                        success_count += 1
                        print(f"✅ {date}: {details}")
                    else:
                        print(f"❌ {date}: {details}")
                except Exception as e:
                    self.logs.append(f"ERRO {date}: {str(e)}")
                    print(f"❌ {date}: {str(e)}")
        
        # Relatório final
        print(f"\n📊 Processing completed:")
        print(f"- Success: {success_count}/{len(dates_to_process)}")
        print(f"- Errors: {len(dates_to_process) - success_count}")
        print(f"- Total records: {self.total_records_processed}")
        
        self._save_logs()
        self._print_stats()
    
    def _collect_single_date_optimized(self, date_str: str) -> tuple[bool, int, str]:
        """Versão otimizada para processamento paralelo"""
        try:
            # Coletar dados com otimizações
            raw_data = scrap_ox_data(date_str, skip_non_business_days=True)
            
            # Verificar se foi pulado por ser fim de semana/feriado
            if raw_data.get('skipped'):
                self.logs.append(f"PULADO {date_str}: {raw_data['error']}")
                return True, 0, "Pulado (fim de semana/feriado)"
            
            # Verificar se está vazio mas é válido
            if raw_data.get('empty'):
                self.logs.append(f"VAZIO {date_str}: Sem dados disponíveis")
                return True, 0, "Sem dados disponíveis"
            
            if 'error' in raw_data and not raw_data.get('skipped') and not raw_data.get('empty'):
                self.logs.append(f"ERRO {date_str}: {raw_data['error']}")
                return False, 0, raw_data['error']
            
            # Processar e salvar apenas se houver dados
            if raw_data.get('tables'):
                processed_dfs = self.processor.process_raw_data(raw_data)
                
                # Verificar se realmente processou dados
                total_records = sum(len(df) for df in processed_dfs.values())
                
                if total_records > 0:
                    self.processor.save_to_parquet(processed_dfs, date_str)
                    self.total_records_processed += total_records
                    
                    success_msg = f"{len(processed_dfs)} tabelas, {total_records} registros"
                    self.logs.append(f"SUCESSO {date_str}: {success_msg}")
                    
                    return True, total_records, success_msg
                else:
                    # Coletou tabelas mas não processou registros
                    problem_msg = "Tabelas encontradas mas não processadas"
                    print(f"⚠️ {date_str}: {problem_msg}")
                    self.logs.append(f"PROBLEMA {date_str}: {problem_msg}")
                    return True, 0, problem_msg
            else:
                self.logs.append(f"VAZIO {date_str}: Sem tabelas encontradas")
                return True, 0, "Sem tabelas encontradas"
            
        except Exception as e:
            self.logs.append(f"ERRO {date_str}: {str(e)}")
            return False, 0, str(e)
    
    def _collect_single_date_legacy(self, date_str: str) -> bool:
        """Versão legada para processamento paralelo (sem UI melhorada)"""
        try:
            print(f"Processando {date_str}...")
            
            # Coletar dados
            raw_data = scrap_ox_data(date_str)
            
            if 'error' in raw_data:
                self.logs.append(f"ERRO {date_str}: {raw_data['error']}")
                return False
            
            # Processar e salvar
            processed_dfs = self.processor.process_raw_data(raw_data)
            
            # Verificar se realmente processou dados
            total_records = sum(len(df) for df in processed_dfs.values())
            
            if total_records > 0:
                self.processor.save_to_parquet(processed_dfs, date_str)
                self.total_records_processed += total_records
                self.logs.append(f"SUCESSO {date_str}: {len(processed_dfs)} tabelas, {total_records} registros")
            else:
                # Coletou dados mas não processou registros
                print(f"⚠️ {date_str}: Dados coletados mas não processados")
                self.logs.append(f"PROBLEMA {date_str}: Dados coletados mas não processados")
            
            # Delay para não sobrecarregar o servidor
            time.sleep(random.uniform(1, 3))
            
            return True
            
        except Exception as e:
            self.logs.append(f"ERRO {date_str}: {str(e)}")
            return False
    
    def _file_exists(self, date_str: str) -> bool:
        """Verifica se já existem dados para uma data"""
        year = date_str[:4]
        month = date_str[5:7]
        
        # Verificar pelo menos uma tabela
        for table_type in self.processor.schemas.keys():
            file_path = self.processor.parquet_dir / table_type / year / month / f"cotacoes_{date_str}.parquet"
            if file_path.exists():
                return True
        return False
    
    def _save_logs(self):
        """Salva logs em arquivo"""
        log_file = self.processor.base_output_dir / "etl_logs.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== ETL Execution: {datetime.now()} ===\n")
            for log in self.logs:
                f.write(f"{log}\n")
    
    def _print_stats(self):
        """Imprime estatísticas de armazenamento"""
        stats = self.processor.get_storage_stats()
        print(f"\nEstatísticas de Armazenamento:")
        total_size = 0
        total_files = 0
        
        for table_type, info in stats.items():
            print(f"  {table_type}: {info['files']} arquivos, {info['size_mb']} MB")
            total_size += info['size_mb']
            total_files += info['files']
        
        print(f"\nTotal: {total_files} arquivos, {total_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='ETL Pipeline para cotações de boi gordo')
    parser.add_argument('command', choices=['single', 'range', 'range-parallel', 'range-fast', 'backfill', 'stats'], 
                       help='Comando a executar')
    
    parser.add_argument('--date', help='Data única (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Data inicial para range (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Data final para range (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='output', help='Diretório de saída')
    parser.add_argument('--workers', type=int, default=5, help='Número de workers paralelos (padrão: 5)')
    parser.add_argument('--force', action='store_true', help='Reprocessar arquivos existentes')
    parser.add_argument('--continue-on-error', action='store_true', help='Continuar processamento mesmo com erros')
    
    args = parser.parse_args()
    
    # Criar pipeline
    stop_on_error = not args.continue_on_error
    pipeline = ETLPipeline(args.output_dir, args.workers, stop_on_error)
    
    if args.command == 'single':
        if not args.date:
            print("❌ ERRO: --date é obrigatório para comando 'single'")
            return
        
        success, records, details = pipeline.collect_single_date(args.date)
        pipeline._save_logs()
        
        if success:
            print(f"✅ Processamento concluído com sucesso! ({records} registros)")
        else:
            print(f"❌ Processamento falhou: {details}")
        
        pipeline._print_stats()
    
    elif args.command == 'range':
        if not args.start_date or not args.end_date:
            print("❌ ERRO: --start-date e --end-date são obrigatórios para comando 'range'")
            return
        
        pipeline.collect_date_range(args.start_date, args.end_date, not args.force)
    
    elif args.command == 'range-parallel':
        if not args.start_date or not args.end_date:
            print("❌ ERRO: --start-date e --end-date são obrigatórios para comando 'range-parallel'")
            return
        
        pipeline.collect_date_range_parallel(args.start_date, args.end_date, not args.force)
    
    elif args.command == 'range-fast':
        # Modo rápido - paralelo otimizado com mais workers
        if not args.start_date or not args.end_date:
            print("❌ ERRO: --start-date e --end-date são obrigatórios para comando 'range-fast'")
            return
        
        print("🚀 Modo RÁPIDO ativado - processamento paralelo otimizado")
        
        # Aumentar workers se não especificado
        if args.workers == 5:  # Valor padrão
            args.workers = 8
        
        fast_pipeline = ETLPipeline(args.output_dir, args.workers, False)  # Não parar em erro
        fast_pipeline.collect_date_range_parallel(args.start_date, args.end_date, not args.force)
    
    elif args.command == 'backfill':
        # Backfill dos últimos 30 dias por padrão
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        if args.start_date:
            start_date = args.start_date
        if args.end_date:
            end_date = args.end_date
        
        print(f"📚 Backfill de {start_date} até {end_date}")
        
        # Usar modo paralelo para backfill
        pipeline.collect_date_range_parallel(start_date, end_date, not args.force)
    
    elif args.command == 'stats':
        stats = pipeline.processor.get_storage_stats()
        print("📊 Estatísticas de Armazenamento:")
        
        total_size = 0
        total_files = 0
        
        for table_type, info in stats.items():
            print(f"  {table_type}:")
            print(f"    Arquivos: {info['files']}")
            print(f"    Tamanho: {info['size_mb']} MB")
            total_size += info['size_mb']
            total_files += info['files']
        
        print(f"\n📈 Total: {total_files} arquivos, {total_size:.2f} MB")

if __name__ == "__main__":
    main()
