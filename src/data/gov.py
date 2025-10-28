from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from pathlib import Path
import os

import pandas as pd
from bcb import sgs

from src.data.utils import get_raw_dir



def _to_dt(x: Optional[str | datetime]) -> datetime:
    if x is None:
        return datetime.today()
    if isinstance(x, datetime):
        return x
    return datetime.strptime(x, "%Y-%m-%d")


def _iter_year_chunks(start: datetime, end: datetime, years_per_chunk: int = 10):
    """
    Yield inclusive (chunk_start, chunk_end) pairs covering [start, end],
    each chunk spanning at most `years_per_chunk` years.
    """
    cur_start = start
    while cur_start <= end:
        try:
            tentative_end = cur_start.replace(year=cur_start.year + years_per_chunk)
        except ValueError:
            tentative_end = cur_start + (datetime(cur_start.year + years_per_chunk, 3, 1) - datetime(cur_start.year, 3, 1))
        chunk_end = min(tentative_end - timedelta(days=1), end)
        yield cur_start, chunk_end
        cur_start = chunk_end + timedelta(days=1)


def _fetch_series_chunked(
    name: str,
    code: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch a (name, code) series from BCB in ≤10y chunks and concatenate the results.
    Returns a DataFrame with columns ['Date', name] (Date as datetime64[ns]).
    """
    frames: List[pd.DataFrame] = []
    for cstart, cend in _iter_year_chunks(start, end, years_per_chunk=10):
        try:
            df = sgs.get((name, code), start=cstart.strftime("%Y-%m-%d"), end=cend.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"Error fetching data for {name} ({code}) from {cstart.date()} to {cend.date()}: {e}")
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "Date"})  # Date in first column
        df = df[["Date", name]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", name])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out



@dataclass
class BRGovAPI:
    parquet_path: str

    def save_indexes_parquet(
        self,
        series_codes: List[Tuple[str, int]],
        start_date: Optional[str | datetime] = None,
        last_date: Optional[str | datetime] = None,
    ) -> None:
        """
        Example:
        series_codes = [
            ("selic", 432),
            ("ipca", 433),
        ]
        """
        os.makedirs(os.path.dirname(str(self.parquet_path)), exist_ok=True)

        start_dt = _to_dt(start_date) if start_date is not None else datetime(2010, 1, 1)
        end_dt = _to_dt(last_date) if last_date is not None else datetime.today()

        combined: Optional[pd.DataFrame] = None

        for name, code in series_codes:
            series_df = _fetch_series_chunked(name, code, start_dt, end_dt)
            if series_df.empty:
                print(f"No data fetched for {name} ({code}).")
                continue

            if combined is None:
                combined = series_df
            else:
                combined = combined.merge(series_df, on="Date", how="outer")

        if combined is None or combined.empty:
            print("No data was fetched. parquet file was not updated.")
            return

        combined = combined.sort_values("Date").reset_index(drop=True)

        parquet_path = Path(self.parquet_path)
        if parquet_path.exists():
            try:
                existing = pd.read_parquet(parquet_path)
                if "Date" in existing.columns:
                    existing["Date"] = pd.to_datetime(existing["Date"])
                else:
                    existing = pd.DataFrame(columns=["Date"])
                merged = existing.merge(combined, on="Date", how="outer")
                merged = merged.sort_values("Date").reset_index(drop=True)
                merged.to_parquet(parquet_path, index=False)
                print(f"Updated (merged) parquet: {parquet_path}")
            except Exception as e:
                print(f"Error merging with existing parquet; writing fresh file instead. Detail: {e}")
                combined.to_parquet(parquet_path, index=False)
                print(f"Saved: {parquet_path}")
        else:
            combined.to_parquet(parquet_path, index=False)
            print(f"Saved: {parquet_path}")



def main():
    parent = get_raw_dir()
    index_parquet = parent / "parquet" /"gov.parquet"

    start_date = "2010-01-01"
    last_date  = "2025-01-01"

    api = BRGovAPI(parquet_path=str(index_parquet))
    api.save_indexes_parquet(series_codes, start_date=start_date, last_date=last_date)



series_codes = [
    # Índices de Preços
    ("selic", 432),

    ("inpc", 188),
    ("ipca", 433),
    ("ipca_15", 7478),
    ("ipca_e", 10764),
    ("ipca_12m", 13522),
    ("igp_10", 7447),
    ("igp_di", 190),
    ("igp_m_1_dec", 7448),
    ("igp_m_2_dec", 7449),
    ("igp_m", 189),
    ("ipc_fipe_1_q", 7463),
    ("ipc_fipe_2_q", 272),
    ("ipc_fipe_3_q", 7464),
    ("ipc_fipe_mensal", 193),

    # Índices Gerais de Preços e Índices de Preços por Atacado
    ("igp_m", 189),
    ("ipa_m", 7450),
    ("igp_di", 190),
    ("ipa_di_geral", 225),
    ("ipa_di_industriais", 7459),
    ("ipa_di_agricolas", 7460),

    # Índices de Preços ao Consumidor
    ("ipc_di", 191),
    ("ipc_c1", 17680),
    ("ipc_3i", 17679),
    ("inpc", 188),
    
    # IPCA - Evolução dos Preços
    ("ipca_livres", 11428),
    ("ipca_comercializaveis", 4447),
    ("ipca_nao_comercializaveis", 4448),
    ("ipca_monitorados", 4449),

    # IPCA - Variações Percentuais Mensais
    ("ipca", 433),
    ("ipca_alimentacao_bebidas", 1635),
    ("ipca_habitacao", 1636),
    ("ipca_artigos_residencia", 1637),
    ("ipca_vestuario", 1638),
    ("ipca_transportes", 1639),
    ("ipca_saude_cuidados_pessoais", 1641),
    ("ipca_despesas_pessoais", 1642),
    ("ipca_educacao", 1643),
    ("ipca_comunicacao", 1640),

    # IPCA - Variações Percentuais nos Últimos 12 Meses
    ("ipca_12m", 433),  # Using the same code as IPCA

    # IPCA - Evolução dos Preços dos Bens
    ("ipca_duraveis", 10843),
    ("ipca_semiduraveis", 10842),
    ("ipca_nao_duraveis", 10841),
    ("ipca_servicos", 10844),
    ("ipca_monitorados", 4449),

    # IPCA - Núcleos
    ("ipca_ex1", 1621),
    ("ipca_ms", 4466),
    ("ipca_dp", 16122),

    # Tráfego de veículos pesados nas estradas pedagiadas
    ("trafego_veiculos_pesados", 28552),

    # Consultas ao Serasa
    ("consultas_serasa", 28547),

    # UCI FGV
    ("uci_fgv", 24352),

    # UCI CNI
    ("uci_cni", 24351),

    # Índices da Produção Industrial
    ("producao_industrial_geral", 21858),
    ("producao_bens_capital", 21863),
    ("producao_bens_intermediarios", 21864),
    ("producao_bens_consumo_geral", 21865),
    ("producao_bens_duraveis", 21866),
    ("producao_bens_nao_duraveis_semiduraveis", 21867),

    # Índices da Produção Industrial - Dessazonalizados
    ("producao_industrial_geral_dessaz", 28503),
    ("producao_bens_capital_dessaz", 28506),
    ("producao_bens_intermediarios_dessaz", 28507),
    ("producao_bens_consumo_geral_dessaz", 28508),
    ("producao_bens_duraveis_dessaz", 28509),
    ("producao_bens_nao_duraveis_semiduraveis_dessaz", 28510),


    # Produção de Autoveículos
    ("producao_autoveiculos_total", 1373),
    ("producao_automoveis_comerciais_leves", 1374),
    ("producao_caminhoes", 1375),

    # Índices de Expectativas do Consumidor e do Empresário Industrial
    ("icc_geral", 4393),
    ("icc_expectativas_futuras", 4394),
    ("icc_condicoes_economicas_atuais", 4395),
    ("icei_geral", 7341),
    ("icei_condicoes_atuais", 7342),
    ("icei_expectativas", 7343),


    # Índice de Volume de Vendas no Varejo - Dessazonalizados
    ("vendas_varejo_geral_dessaz", 28473),
    ("vendas_comercio_ampliado_dessaz", 28485),
    ("vendas_hiper_super_produtos_alimenticios_dessaz", 28475),
    ("vendas_moveis_eletrodomesticos_dessaz", 28478),
    ("vendas_automoveis_motocicletas_pecas_dessaz", 28479),
    ("vendas_material_construcao_dessaz", 28484),

    # Indicadores de Investimento
    ("prod_insumos_construcao_civil", 21868),
    ("producao_bens_capital", 21863),
    ("faturamento_real_bk_mecanicos", 7358),
    ("desembolso_bndes", 7415),

    # Indicadores de Investimento - Dessazonalizados
    ("exportacao_bens_capital", 28567),
    ("importacao_bens_capital", 28568),
    ("exportacao_bens_capital_dessaz", 28569),
    ("importacao_bens_capital_dessaz", 28570),

    # Índice do Nível de Emprego Formal
    ("emprego_formal_total", 25239),
    ("emprego_formal_transformacao", 25241),
    ("emprego_formal_comercio", 25256),
    ("emprego_formal_servicos", 25257),
    ("emprego_formal_construcao_civil", 25255),

    # Índice do Nível de Emprego Formal - Dessazonalizados
    ("emprego_formal_total_dessaz", 28512),
    ("emprego_formal_transformacao_dessaz", 28513),
    ("emprego_formal_comercio_dessaz", 28514),
    ("emprego_formal_servicos_dessaz", 28515),
    ("emprego_formal_construcao_civil_dessaz", 28516),

    # Força de Trabalho e População em Idade Ativa
    ("forca_trabalho_ocupadas", 24379),
    ("forca_trabalho_desocupadas", 24380),
    ("forca_trabalho_total", 24378),
    ("pessoas_idade_trabalhar", 24370),
    ("taxa_desocupacao", 24369),

    # Taxa de Desocupação
    ("taxa_desocupacao_media", 24369),
    ("taxa_desocupacao_norte", 28562),
    ("taxa_desocupacao_centro_oeste", 28563),
    ("taxa_desocupacao_nordeste", 28564),
    ("taxa_desocupacao_sudeste", 28565),
    ("taxa_desocupacao_sul", 28566),

    # Rendimento Médio Real Efetivo de Todos os Trabalhos
    ("remuneracao_media_deflac", 24381),
    ("remuneracao_media_nominal", 24382),
    ("pessoal_ocupado_rendimento", 28543),
    ("massa_salarial", 28544),

    # Produto Interno Bruto e Taxas Médias de Crescimento
    ("pib_preco_corrente", 1207),
    ("pib_em_rs_ultimo_ano", 1208),
    ("pib_em_usd", 7324),
    ("populacao", 21774),
    ("pib_per_capita_preco_corrente", 21775),
    ("pib_per_capita_em_rs_ultimo_ano", 21777),
    ("pib_per_capita_em_usd", 21776),

    # Contas Nacionais Trimestrais
    ("pib_trimestral", 22099),
    ("consumo_familias", 22100),
    ("consumo_governo", 22101),
    ("fbcf", 22102),
    ("exportacao", 22103),
    ("importacao", 22104),
    ("pib_dessaz", 22109),
    ("consumo_familias_dessaz", 22110),
    ("consumo_governo_dessaz", 22111),
    ("fbcf_dessaz", 22113),
    ("exportacao_dessaz", 22114),
    ("importacao_dessaz", 22115),

    # Produto Interno Bruto Trimestral
    ("pib_trimestral", 22099),
    ("pib_valor_adicionado", 22097),
    ("pib_agropecuaria", 22083),
    ("pib_industria", 22084),
    ("pib_servicos", 22089),
    ("pib_trimestral_dessaz", 22109),
    ("pib_valor_adicionado_dessaz", 22108),
    ("pib_agropecuaria_dessaz", 22105),
    ("pib_industria_dessaz", 22106),
    ("pib_servicos_dessaz", 22107)
]



if __name__ == "__main__":
    main()
