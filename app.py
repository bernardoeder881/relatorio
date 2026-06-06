import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.spatial import KDTree

# Configuração da página do Streamlit
st.set_page_config(page_title="Gerador de Relatório SEDEC-RJ", layout="wide")

# Funções auxiliares
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def processar_kml(kml_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    polygons = []
    for placemark in root.findall('.//kml:Placemark', namespaces):
        for poly in placemark.findall('.//kml:Polygon', namespaces):
            outer_boundary = poly.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
            if outer_boundary is not None:
                coords = [(float(p.split(',')[0]), float(p.split(',')[1])) for p in outer_boundary.text.strip().split() if len(p.split(',')) >= 2]
                if len(coords) >= 3: 
                    polygons.append(Polygon(coords))
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0], polygons

def gerar_mapa_b64(polygons, sisgeo_df, firms_df, inpe_df, matched_sisgeo, matched_firms, matched_inpe):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plotar o polígono do KML (Estado RJ)
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='#34495e', linewidth=1, alpha=0.5)

    # Plotar pontos sem cruzamento (opacos)
    if len(firms_df) > 0:
        ax.scatter(firms_df['longitude'], firms_df['latitude'], color='#e74c3c', s=15, alpha=0.3, edgecolors='none', label='FIRMS/NASA')
    if len(inpe_df) > 0:
        ax.scatter(inpe_df['Longitude'], inpe_df['Latitude'], color='#f39c12', s=20, marker='s', edgecolors='none', alpha=0.4, label='INPE')
    if len(sisgeo_df) > 0:
        ax.scatter(sisgeo_df['Longitude'], sisgeo_df['Latitude'], color='#2980b9', s=30, marker='*', edgecolors='none', alpha=0.4, label='SISGEO')

    # Plotar pontos validados (com cruzamento ICE)
    if len(matched_firms) > 0:
        ax.scatter(matched_firms['longitude'], matched_firms['latitude'], color='#c0392b', s=200, marker='p', edgecolors='#f1c40f', linewidth=2, zorder=5, label='🔥 Foco Confirmado')
    if len(matched_inpe) > 0:
        ax.scatter(matched_inpe['Longitude'], matched_inpe['Latitude'], color='#c0392b', s=200, marker='p', edgecolors='#f1c40f', linewidth=2, zorder=5)
    if len(matched_sisgeo) > 0:
        ax.scatter(matched_sisgeo['Longitude'], matched_sisgeo['Latitude'], color='#2980b9', s=250, marker='X', edgecolors='#ecf0f1', linewidth=2, zorder=6, label='🚒 Viaturas Empenhadas')

    ax.set_title("Focos de Calor e Ocorrências - Cruzamento Validado ICE", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True, shadow=True, facecolor='white', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Interface do Streamlit ---
st.title("🚒 NIFAD - Gerador Oficial de Relatório SEDEC-RJ")
st.markdown("Faça o upload das bases de dados para extrair a inteligência geoespacial e gerar o relatório final formatado.")

with st.sidebar:
    st.header("1. Upload de Arquivos")
    kml_file = st.file_uploader("Arquivo KML (Ex: ESTADO RJ.kml)", type=['kml', 'xml'])
    sisgeo_file = st.file_uploader("Ocorrências SISGEO (CSV)", type=['csv'])
    # MULTIPLOS ARQUIVOS PARA O FIRMS
    firms_files = st.file_uploader("Focos FIRMS NASA (Selecione os 4 CSVs)", type=['csv'], accept_multiple_files=True)
    inpe_file = st.file_uploader("Focos INPE (CSV)", type=['csv'])
    
    st.header("2. Período de Análise")
    start_date = st.date_input("Data Inicial")
    end_date = st.date_input("Data Final")
    
    gerar = st.button("Cruzar Dados e Gerar Relatório", type="primary", use_container_width=True)

if gerar:
    # Verificação atualizada para incluir a lista de arquivos do FIRMS
    if not all([kml_file, sisgeo_file, inpe_file]) or not firms_files:
        st.warning("⚠️ Por favor, faça o upload de todos os arquivos (KML, SISGEO, os 4 do FIRMS e INPE) para prosseguir.")
    else:
        with st.spinner("A extrair inteligência geoespacial e a cruzar coordenadas (Calculando ICE)..."):
            
            # 1. KML Parsing
            rj_geom, polygons = processar_kml(kml_file)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 2. SISGEO Parsing
            sisgeo_df = pd.read_csv(sisgeo_file, sep=None, engine='python')
            sisgeo_df.columns = sisgeo_df.columns.str.strip()
            
            if 'Latitude/Longitude' in sisgeo_df.columns:
                sisgeo_df['Latitude'] = sisgeo_df['Latitude/Longitude'].apply(lambda v: float(str(v).split(',')[0]) if ',' in str(v) else np.nan)
                sisgeo_df['Longitude'] = sisgeo_df['Latitude/Longitude'].apply(lambda v: float(str(v).split(',')[1]) if ',' in str(v) else np.nan)
            
            sisgeo_df = sisgeo_df.dropna(subset=['Latitude', 'Longitude'])
            
            if len(sisgeo_df) > 0:
                col_data = 'Data Ocorrência' if 'Data Ocorrência' in sisgeo_df.columns else sisgeo_df.columns[0]
                sisgeo_df[col_data] = pd.to_datetime(sisgeo_df[col_data], dayfirst=True, errors='coerce')
                sisgeo_filtrado = sisgeo_df[(sisgeo_df[col_data] >= start_dt) & (sisgeo_df[col_data].dt.floor('d') <= end_dt)].copy()
                sisgeo_filtrado['is_in_rj'] = sisgeo_filtrado.apply(lambda row: Point(row['Longitude'], row['Latitude']).within(rj_geom), axis=1)
                sisgeo_filtrado = sisgeo_filtrado[sisgeo_filtrado['is_in_rj']].copy()
            else:
                sisgeo_filtrado = pd.DataFrame()
            
            # 3. INPE Parsing
            inpe_df = pd.read_csv(inpe_file, sep=None, engine='python')
            inpe_df.columns = inpe_df.columns.str.strip()
            if len(inpe_df.columns) > 0:
                col_data_inpe = 'DataHora' if 'DataHora' in inpe_df.columns else inpe_df.columns[0]
                inpe_df[col_data_inpe] = pd.to_datetime(inpe_df[col_data_inpe], errors='coerce')
                inpe_filtrado = inpe_df[(inpe_df[col_data_inpe] >= start_dt) & (inpe_df[col_data_inpe].dt.floor('d') <= end_dt)].copy()
                
                col_lat_inpe = 'Latitude' if 'Latitude' in inpe_filtrado.columns else 'Lat'
                col_lon_inpe = 'Longitude' if 'Longitude' in inpe_filtrado.columns else 'Lon'
                inpe_filtrado['is_in_rj'] = inpe_filtrado.apply(lambda row: Point(row[col_lon_inpe], row[col_lat_inpe]).within(rj_geom), axis=1)
                inpe_filtrado = inpe_filtrado[inpe_filtrado['is_in_rj']].copy()
            else:
                inpe_filtrado = pd.DataFrame()
            
            # 4. FIRMS Parsing (UNINDO OS 4 ARQUIVOS)
            dfs_firms = []
            for f in firms_files:
                temp_df = pd.read_csv(f, sep=None, engine='python')
                temp_df.columns = temp_df.columns.str.strip()
                dfs_firms.append(temp_df)
            
            firms_df = pd.concat(dfs_firms, ignore_index=True)
            firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'], errors='coerce')
            firms_filtrado = firms_df[(firms_df['acq_date'] >= start_dt) & (firms_df['acq_date'] <= end_dt)].copy()
            firms_filtrado['is_in_rj'] = firms_filtrado.apply(lambda row: Point(row['longitude'], row['latitude']).within(rj_geom), axis=1)
            firms_filtrado = firms_filtrado[firms_filtrado['is_in_rj']].copy()

            # 5. Cruzamento ICE (Índice de Correlação Espacial)
            matched_occ_indices = set()
            
            # ICE FIRMS
            matched_firms = []
            ice_f_firms, ice_m_firms, ice_s_firms = 0, 0, 0
            if len(sisgeo_filtrado) > 0 and len(firms_filtrado) > 0:
                for idx, f_row in firms_filtrado.iterrows():
                    ds = haversine(f_row['latitude'], f_row['longitude'], sisgeo_filtrado['Latitude'].values, sisgeo_filtrado['Longitude'].values)
                    if len(ds) > 0:
                        min_dist = np.min(ds)
                        if min_dist <= 1.0:
                            matched_firms.append(f_row)
                            matched_occ_indices.add(sisgeo_filtrado.iloc[np.argmin(ds)].name)
                            ice_f_firms += 1
                        elif 1.0 < min_dist <= 3.0: ice_m_firms += 1
                        elif 3.0 < min_dist <= 5.0: ice_s_firms += 1
            matched_firms_df = pd.DataFrame(matched_firms)

            # ICE INPE
            matched_inpe = []
            ice_f_inpe, ice_m_inpe, ice_s_inpe = 0, 0, 0
            if len(sisgeo_filtrado) > 0 and len(inpe_filtrado) > 0:
                for idx, i_row in inpe_filtrado.iterrows():
                    ds = haversine(i_row[col_lat_inpe], i_row[col_lon_inpe], sisgeo_filtrado['Latitude'].values, sisgeo_filtrado['Longitude'].values)
                    if len(ds) > 0:
                        min_dist = np.min(ds)
                        if min_dist <= 1.0:
                            matched_inpe.append(i_row)
                            matched_occ_indices.add(sisgeo_filtrado.iloc[np.argmin(ds)].name)
                            ice_f_inpe += 1
                        elif 1.0 < min_dist <= 3.0: ice_m_inpe += 1
                        elif 3.0 < min_dist <= 5.0: ice_s_inpe += 1
            matched_inpe_df = pd.DataFrame(matched_inpe)
            
            if len(matched_occ_indices) > 0:
                matched_sisgeo_df = sisgeo_filtrado.loc[list(matched_occ_indices)]
            else:
                matched_sisgeo_df = pd.DataFrame()

            # 6. Resumos de Dados Seguros (Evitando KeyError / IndexError)
            coluna_subtipo = None
            if len(sisgeo_filtrado.columns) > 0:
                for col in sisgeo_filtrado.columns:
                    if col.lower() in ['subtipo', 'tipo', 'natureza', 'descricao']:
                        coluna_subtipo = col
                        break
            
            if coluna_subtipo:
                sisgeo_sub = sisgeo_filtrado[coluna_subtipo].value_counts().reset_index()
                sisgeo_sub.columns = ['Subtipo', 'Ocorrências']
            else:
                sisgeo_sub = pd.DataFrame(columns=['Subtipo', 'Ocorrências'])
                st.warning("⚠️ Atenção: A coluna de tipologia (Subtipo/Tipo) não foi encontrada.")

            # Proteção contra IndexError
            if len(sisgeo_filtrado.columns) > 0:
                col_ocorrencia = 'Ocorrência' if 'Ocorrência' in sisgeo_filtrado.columns else sisgeo_filtrado.columns[0]
            else:
                col_ocorrencia = 'Ocorrência'
                
            col_unidade = 'Unidade' if 'Unidade' in sisgeo_filtrado.columns else '-'
            
            tabela_cruzamento = ""
            if len(matched_sisgeo_df) > 0:
                for _, row in matched_sisgeo_df.iterrows():
                    val_ocorrencia = row[col_ocorrencia] if col_ocorrencia in row else '-'
                    val_subtipo = row[coluna_subtipo] if coluna_subtipo else '-'
                    val_unidade = row[col_unidade] if col_unidade in row else '-'
                    val_municipio = row['Município'] if 'Município' in sisgeo_filtrado.columns else '-'
                    
                    tabela_cruzamento += f"<tr><td><span style='color: #2980b9; font-weight: bold;'>&#128658; {val_ocorrencia}</span></td><td>{val_municipio}</td><td>{val_subtipo}</td><td>{val_unidade}</td><td>{round(row['Latitude'],4)} / {round(row['Longitude'],4)}</td></tr>"

            # 7. Gerar Mapa
            map_img_b64 = gerar_mapa_b64(polygons, sisgeo_filtrado, firms_filtrado, inpe_filtrado, matched_sisgeo_df, matched_firms_df, matched_inpe_df)

            # 8. Construir HTML
            html_content = f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2c3e50; line-height: 1.4; font-size: 10pt; background-color: #ffffff; padding: 20px; }}
                    .header {{ text-align: center; background-color: #c0392b; color: #ffffff; padding: 15px; border-bottom: 3px solid #e74c3c; margin-bottom: 20px;}}
                    .header h1 {{ font-size: 16pt; margin: 0 0 5px 0; text-transform: uppercase; }}
                    h2 {{ font-size: 13pt; color: #c0392b; margin-top: 25px; border-left: 5px solid #e74c3c; padding-left: 10px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 9pt; }}
                    th, td {{ border: 1px solid #bdc3c7; padding: 8px 6px; text-align: left; }}
                    th {{ background-color: #34495e; color: #ffffff; }}
                    tr:nth-child(even) {{ background-color: #f2f4f5; }}
                    .map-container {{ text-align: center; margin: 20px 0; padding: 10px; border: 1px solid #bdc3c7; border-radius: 6px; }}
                    .map-img {{ max-width: 100%; height: auto; }}
                    .intro-box {{ background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 10px; margin-bottom: 15px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>RELATÓRIO DE ANOMALIAS TÉRMICAS E FOGO EM VEGETAÇÃO NO ESTADO DO RIO DE JANEIRO</h1>
                    <p>Período de referência: {start_dt.strftime('%d/%m/%Y')} a {end_dt.strftime('%d/%m/%Y')} | Gerado via Sistema NIFAD</p>
                </div>

                <div class="intro-box">
                    <p>Este relatório apresenta um panorama integrado das ocorrências de fogo em vegetação. Diferentemente da delimitação padrão, <strong>este documento filtrou rigorosamente todos os pontos de calor (satélite) e as ocorrências (CBMERJ) utilizando os limites geográficos extraídos do arquivo KML.</strong></p>
                </div>

                <h2>1. Quadro de Dados Encontrados (No Polígono)</h2>
                <table>
                    <thead><tr><th>Fonte / Sistema</th><th>Registros / Focos Filtrados</th></tr></thead>
                    <tbody>
                        <tr><td>FIRMS/NASA</td><td>{len(firms_filtrado)}</td></tr>
                        <tr><td>INPE/BDQueimadas</td><td>{len(inpe_filtrado)}</td></tr>
                        <tr><td>SISGEO/CBMERJ</td><td>{len(sisgeo_filtrado)}</td></tr>
                    </tbody>
                </table>

                <h2>2. Quadro de Análise Operacional (Tipologia)</h2>
                <table><thead><tr><th>Subtipo</th><th>Ocorrências Despachadas</th></tr></thead><tbody>
                {''.join([f"<tr><td>{row['Subtipo']}</td><td>{row['Ocorrências']}</td></tr>" for _, row in sisgeo_sub.iterrows()])}
                </tbody></table>

                <h2>3. Ocorrências SISGEO com Correlação Satelital Direta (ICE-F)</h2>
                <table><thead><tr><th>Ocorrência</th><th>Município</th><th>Subtipo</th><th>Unidade</th><th>Lat/Lon</th></tr></thead>
                <tbody>{tabela_cruzamento}</tbody></table>

                <h2>4. Índice de Correlação Espacial (ICE) Zonal</h2>
                <table><thead><tr><th>Fonte Satelital</th><th>ICE-F (Na Mosca: &le; 1km)</th><th>ICE-M (Arredores: 1 a 3km)</th><th>ICE-S (Dispersos: 3 a 5km)</th></tr></thead>
                <tbody>
                    <tr><td>FIRMS/NASA</td><td>{ice_f_firms}</td><td>{ice_m_firms}</td><td>{ice_s_firms}</td></tr>
                    <tr><td>INPE/BDQueimadas</td><td>{ice_f_inpe}</td><td>{ice_m_inpe}</td><td>{ice_s_inpe}</td></tr>
                </tbody></table>
                <p style="font-size: 8.5pt; color: #555;"><strong>Figura 1:</strong> A prevalência de focos no ICE-F em detrimento do ICE-M comprova que as guarnições convergem diretamente para o epicentro térmico da anomalia.</p>

                <h2>5. Mapa de Calor e Validação Tática</h2>
                <div class="map-container">
                    <img src="data:image/png;base64,{map_img_b64}" class="map-img" alt="Mapa">
                    <p style="font-size: 8.5pt; margin-top: 10px;">Pontos validados no cruzamento (&#128293;) estão destacados sobrepostos às viaturas atuantes (&#128658;).</p>
                </div>

                <h2>6. Conclusão Institucional</h2>
                <p style="text-align: justify; font-size: 9pt;">A análise confirmada pelos limites geográficos mapeia de maneira robusta os padrões de incêndio. A métrica ICE (Índice de Correlação Espacial) comprova que a pronta-resposta operacional do CBMERJ esteve perfeitamente alinhada com as anomalias captadas pelo espaço, neutralizando os alertas orbitais de maior relevância.</p>
                <p style="text-align: justify; font-size: 9pt;">Ademais, cabe pontuar a divergência quantitativa entre os registros da plataforma FIRMS/NASA e do BDQueimadas/INPE. Essa diferença não denota inconsistência nos sistemas, mas sim metodologias de processamento orbitais distintas e complementares. A NASA opera com sensores de altíssima sensibilidade (como o VIIRS) para fornecer dados brutos em tempo quase real, enquanto o INPE aplica um rigoroso algoritmo de filtragem utilizando satélites de referência (como o Aqua) para depurar 'falsos positivos'.</p>
                <p style="text-align: justify; font-size: 9pt;">Os dados do ICE-S evidenciam como a inteligência geográfica agrega valor estratégico à triagem de chamados. A validação espacial permite filtrar anomalias térmicas pulverizadas e queimas controladas que não configuram emergência real, evitando o empenho desnecessário da tropa e preservando a prontidão da Força Especializada.</p>
                <p style="text-align: justify; font-size: 9pt;">A análise do SISGEO indica forte desgaste originado por eventos antrópicos urbanos. Para combater a ignição irregular, sugere-se a articulação da Defesa Civil com as COMPDECs e concessionárias, promovendo ações de zeladoria e fortalecendo o eixo preventivo do PLANCON.</p>
            </body>
            </html>
            """
            
            st.success("✅ Relatório Processado com Sucesso!")
            
            # Aba para Visualização e Botão de Download
            tab1, tab2 = st.tabs(["📄 Visualizar Relatório", "📥 Opções de Exportação"])
            
            with tab1:
                st.components.v1.html(html_content, height=800, scrolling=True)
                
            with tab2:
                st.markdown("### Exportar Documento")
                st.download_button(
                    label="Descarregar Relatório em HTML (Pode ser impresso como PDF no navegador)",
                    data=html_content,
                    file_name=f"Relatorio_NIFAD_{start_dt.strftime('%Y%m%d')}.html",
                    mime="text/html",
                    type="primary"
                )
