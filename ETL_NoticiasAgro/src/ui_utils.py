#!/usr/bin/env python3
"""
Utilitários para interface de usuário do ETL
"""

import sys
import time
import threading
from typing import Optional

class Spinner:
    """Spinner animado para mostrar progresso com percentual"""
    
    def __init__(self, message: str = "Processing", chars: str = "-\\|/-\\"):
        self.message = message
        self.chars = chars
        self.idx = 0
        self.stop_spinner = False
        self.spinner_thread: Optional[threading.Thread] = None
        self.current_progress = 0
        self.total_progress = 100
    
    def _spin(self):
        """Loop interno do spinner"""
        while not self.stop_spinner:
            if self.total_progress > 0:
                percentage = (self.current_progress / self.total_progress) * 100
                progress_msg = f'{self.message} {self.chars[self.idx % len(self.chars)]} {percentage:.1f}%'
            else:
                progress_msg = f'{self.message} {self.chars[self.idx % len(self.chars)]}'
            
            sys.stdout.write(f'\r{progress_msg}')
            sys.stdout.flush()
            self.idx += 1
            time.sleep(0.1)
    
    def update_progress(self, current: int, total: int):
        """Atualiza o progresso atual"""
        self.current_progress = current
        self.total_progress = total
    
    def start(self):
        """Inicia o spinner"""
        self.stop_spinner = False
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def stop(self, final_message: str = None):
        """Para o spinner"""
        self.stop_spinner = True
        if self.spinner_thread:
            self.spinner_thread.join()
        
        # Limpar linha e mostrar mensagem final
        line_length = len(self.message) + 20  # Espaço extra para percentual
        sys.stdout.write('\r' + ' ' * line_length + '\r')
        if final_message:
            print(final_message)
        sys.stdout.flush()

class ProgressReporter:
    """Reporter de progresso para o ETL"""
    
    def __init__(self):
        self.current_spinner: Optional[Spinner] = None
        self.processed_count = 0
        self.total_count = 0
    
    def start_range_execution(self, start_date: str, end_date: str, total_dates: int):
        """Inicia execução de um range de datas com spinner único"""
        print(f"🚀 Executing in range {start_date} to {end_date} ({total_dates} dates)")
        print("=" * 60)
        
        # Iniciar spinner único para todo o período
        self.total_count = total_dates
        self.processed_count = 0
        message = f"Processing {total_dates} dates"
        self.current_spinner = Spinner(message)
        self.current_spinner.start()
    
    def update_progress(self, date: str, success: bool, details: str = ""):
        """Atualiza progresso sem parar o spinner"""
        self.processed_count += 1
        
        # Atualizar progresso no spinner
        if self.current_spinner:
            self.current_spinner.update_progress(self.processed_count, self.total_count)
        
    def start_date_processing(self, date: str, current: int, total: int):
        """DEPRECATED - usar update_progress para spinner único"""
        pass
    
    def finish_date_processing(self, date: str, success: bool, details: str = "", records: int = 0):
        """DEPRECATED - usar update_progress para spinner único"""
        pass
    
    def report_error_and_stop(self, date: str, error: str):
        """Reporta erro crítico e para execução"""
        if self.current_spinner:
            self.current_spinner.stop(f"❌ STOPPED on {date}: {error}")
            self.current_spinner = None
        
        print(f"\n🛑 CRITICAL ERROR on {date}:")
        print(f"   Reason: {error}")
        print(f"   ETL execution stopped immediately.")
        print("=" * 60)
    
    def finish_execution(self, success_count: int, total_count: int, total_records: int):
        """Finaliza execução completa"""
        if self.current_spinner:
            percentage = (success_count / total_count) * 100 if total_count > 0 else 0
            self.current_spinner.stop(f"✅ Complete! {success_count}/{total_count} successful ({percentage:.1f}%)")
            self.current_spinner = None
        
        print("\n" + "=" * 60)
        print(f"🎯 ETL Execution Complete:")
        print(f"   • Successful: {success_count}/{total_count} dates")
        print(f"   • Failed: {total_count - success_count}/{total_count} dates")
        print(f"   • Total records processed: {total_records}")
        
        if success_count == total_count:
            print("🎉 All dates processed successfully!")
        elif success_count == 0:
            print("💥 No dates processed successfully.")
        else:
            print("⚠️  Partial success - check logs for details.")
        
        print("=" * 60)

def format_error_message(error: Exception, context: str = "") -> str:
    """Formata mensagem de erro de forma mais legível"""
    error_name = type(error).__name__
    error_msg = str(error)
    
    # Simplificar erros comuns
    if "403" in error_msg and "Forbidden" in error_msg:
        return "Website blocked request (403 Forbidden)"
    elif "pyarrow" in error_msg or "fastparquet" in error_msg:
        return "Missing Parquet libraries (install pyarrow/fastparquet)"
    elif "Connection" in error_msg or "timeout" in error_msg:
        return "Network connection issue"
    elif "KeyError" in error_name:
        return f"Data structure error: {error_msg}"
    else:
        return f"{error_name}: {error_msg}"

# Exemplo de uso
if __name__ == "__main__":
    # Teste do spinner
    reporter = ProgressReporter()
    
    reporter.start_range_execution("2025-01-01", "2025-01-03", 3)
    
    for i, date in enumerate(["2025-01-01", "2025-01-02", "2025-01-03"], 1):
        reporter.start_date_processing(date, i, 3)
        time.sleep(2)  # Simular processamento
        
        if date == "2025-01-02":
            reporter.report_error_and_stop(date, "Website blocked request (403 Forbidden)")
            break
        else:
            reporter.finish_date_processing(date, True, "2 tables processed", 15)
    
    reporter.finish_execution(1, 3, 15)