#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSC → Google Sheets | Queries por URL (lista en Sheet)
-----------------------------------------------------
Lee una lista de URLs desde un Google Sheet y obtiene las *queries* de Search Console
para cada URL (últimos 16 meses), escribiendo el resultado en dos pestañas:
  1) URL_Queries      → todas las queries por URL (ordenadas por Clicks desc)
  2) TopQueryPorURL   → solo la query con más Clicks por cada URL

Autenticación: **Service Account** (JSON en variables de entorno).
No necesita navegador. Ideal para servidores (Render, etc).

ENV requeridas (Render / local):
  - GOOGLE_CREDENTIALS_JSON   (pegar JSON completo)  **o**
  - GOOGLE_CREDENTIALS_BASE64 (JSON codificado en base64)
  - GSC_SITE_URL              (ej. 'sc-domain:prodesco.es' o 'https://prodesco.es/')
  - SOURCE_SPREADSHEET_ID     (ID del Sheet con la columna 'URL')
  - SOURCE_TAB_NAME           (opcional, si no se indica se usa la primera hoja)
  - TARGET_SPREADSHEET_ID     (opcional, por defecto se usa el mismo que SOURCE)
  - OUTPUT_TAB_ALL            (opcional, por defecto 'URL_Queries')
  - OUTPUT_TAB_TOP            (opcional, por defecto 'TopQueryPorURL')
  - GSC_SEARCH_TYPE           (opcional, por defecto 'web')
  - DATA_STATE                (opcional: 'final' o 'all'; por defecto 'final')

Notas sobre integridad de datos:
- 'DATA_STATE=final' devuelve datos estabilizados (sin parciales); 'all' incluye datos recientes
  que pueden estar incompletos. Véase doc oficial de 'dataState'. 
- La API puede ocultar *queries anónimas* por umbrales de privacidad; no se pueden recuperar.
- Para propiedades de Dominio, utilizar 'sc-domain:tudominio.com' garantiza cobertura de http/https y subdominios.
"""

import os
import sys
import json
import time
import base64
import datetime as dt
from typing import List, Dict, Tuple, Optional, Iterable
from collections import defaultdict

from dateutil.relativedelta import relativedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


ROW_LIMIT = 25000          # Máximo por página de la API de GSC
BATCH_WRITE_ROWS = 10000   # Lote de escritura a Sheets


# ---------------- Utils ----------------

def env_or_fail(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Falta variable de entorno: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def get_dates_last_16_months() -> Tuple[str, str]:
    """Devuelve (start_date, end_date) para últimos 16 meses.
    end_date = hoy - 2 días por el retraso normal de GSC.
    """
    today = dt.date.today()
    end_date = today - dt.timedelta(days=2)
    start_date = end_date - relativedelta(months=16)
    return start_date.isoformat(), end_date.isoformat()


def load_credentials_json_from_env() -> dict:
    """Carga el JSON de credenciales desde env (RAW o base64). Intenta arreglar \\n escapados."""
    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")

    data = None
    if b64:
        try:
            decoded = base64.b64decode(b64).decode("utf-8")
            data = json.loads(decoded)
        except Exception as e:
            print(f"[ERROR] GOOGLE_CREDENTIALS_BASE64 inválido: {e}", file=sys.stderr)
            sys.exit(3)

    if data is None and raw_json:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Intentar reemplazar literales '\\n' por saltos reales
            try:
                fixed = raw_json.replace('\\n', '\n')
                data = json.loads(fixed)
            except Exception as e:
                print(f"[ERROR] GOOGLE_CREDENTIALS_JSON inválido: {e}", file=sys.stderr)
                sys.exit(3)

    if data is None:
        print("[ERROR] Debes definir GOOGLE_CREDENTIALS_JSON o GOOGLE_CREDENTIALS_BASE64", file=sys.stderr)
        sys.exit(3)

    # Checks básicos
    for k in ("type", "client_email", "private_key"):
        if k not in data or not data[k]:
            print(f"[ERROR] Credenciales sin campo requerido: {k}", file=sys.stderr)
            sys.exit(3)

    return data


def build_credentials(scopes):
    info = load_credentials_json_from_env()
    return service_account.Credentials.from_service_account_info(info, scopes=scopes)


def build_services(creds):
    # Search Console: usa 'searchconsole' v1; cae a 'webmasters' v3 si no está
    try:
        gsc = build("searchconsole", "v1", credentials=creds, cache_discovery=False)
    except Exception:
        gsc = build("webmasters", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return gsc, sheets


def a1(row, col):
    """Convierte fila/col (1-index) a notación A1."""
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


# ---------------- Sheets helpers ----------------

def get_first_sheet_title(sheets, spreadsheet_id: str) -> str:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets_meta = meta.get("sheets", [])
    if not sheets_meta:
        print("[ERROR] El spreadsheet no tiene pestañas.", file=sys.stderr)
        sys.exit(4)
    return sheets_meta[0]["properties"]["title"]


def read_url_column(sheets, spreadsheet_id: str, source_tab_name: Optional[str]) -> List[str]:
    """Lee la columna 'URL'. Si no encuentra cabecera 'URL', usa columna A."""
    if not source_tab_name:
        source_tab_name = get_first_sheet_title(sheets, spreadsheet_id)

    # Leer primera fila para detectar cabeceras
    header_resp = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=f"{source_tab_name}!A1:Z1"
    ).execute()
    headers = header_resp.get("values", [[]])
    headers = headers[0] if headers else []

    url_col_idx = None
    for idx, name in enumerate(headers, start=1):
        if str(name).strip().lower() == "url":
            url_col_idx = idx
            break

    if url_col_idx is None:
        # Fallback: columna A completa
        rng = f"{source_tab_name}!A2:A"
    else:
        col_letter = a1(1, url_col_idx)[:-1]  # 'A', 'B', etc.
        rng = f"{source_tab_name}!{col_letter}2:{col_letter}"

    resp = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=rng
    ).execute()

    values = resp.get("values", [])
    urls = []
    for row in values:
        if not row:
            continue
        u = row[0].strip()
        if u:
            urls.append(u)
    # de-duplicar preservando orden
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def ensure_sheet_tab(sheets, spreadsheet_id: str, sheet_name: str) -> int:
    """Crea la pestaña si no existe y retorna sheetId."""
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        if props.get("title") == sheet_name:
            return props.get("sheetId")

    # crear
    reqs = [{"addSheet": {"properties": {"title": sheet_name}}}]
    resp = sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": reqs}).execute()
    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]


def ensure_grid_size(sheets, spreadsheet_id: str, sheet_id: int, needed_rows: int, needed_cols: int):
    """Amplía la cuadrícula de la pestaña si hace falta (filas/columnas)."""
    # Añadimos un margen para evitar redimensionar continuamente
    target_rows = max(needed_rows + 100, 1000)
    target_cols = max(needed_cols, 4)
    reqs = [{
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "rowCount": target_rows,
                    "columnCount": target_cols
                }
            },
            "fields": "gridProperties.rowCount,gridProperties.columnCount"
        }
    }]
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": reqs}).execute()


def clear_sheet(sheets, spreadsheet_id: str, sheet_name: str):
    """Limpia el rango A:Z de la pestaña (ignora si está vacía)."""
    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A:Z",
        body={}
    ).execute()


def write_values_batched(sheets, spreadsheet_id: str, sheet_name: str, values: List[List]):
    num_cols = len(values[0]) if values else 0
    start = 0
    row_index = 1
    while start < len(values):
        chunk = values[start:start + BATCH_WRITE_ROWS]
        end_row = row_index + len(chunk) - 1
        from_a1 = a1(row_index, 1)
        to_a1 = a1(end_row, num_cols)
        rng = f"{sheet_name}!{from_a1}:{to_a1}"
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=rng,
            valueInputOption="RAW",
            body={"values": chunk}
        ).execute()
        start += BATCH_WRITE_ROWS
        row_index = end_row + 1


# ---------------- GSC helpers ----------------

def fetch_queries_for_page(gsc, site_url: str, page_url: str, start_date: str, end_date: str,
                           search_type: str = "web", data_state: str = "final") -> List[Dict]:
    """Devuelve todas las filas (query) para una página concreta, paginando."""
    all_rows: List[Dict] = []
    start_row = 0
    backoff = 1.0

    while True:
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["query"],
            "dimensionFilterGroups": [{
                "groupType": "and",
                "filters": [{
                    "dimension": "page",
                    # operator 'equals' es el default según docs
                    "expression": page_url
                }]
            }],
            "rowLimit": ROW_LIMIT,
            "startRow": start_row,
            "searchType": search_type,
            "dataState": data_state  # 'final' (default) o 'all'
        }
        try:
            resp = gsc.searchanalytics().query(siteUrl=site_url, body=body).execute()
            backoff = 1.0
        except HttpError as e:
            if e.resp.status in (429, 500, 503):
                print(f"[WARN] {page_url} → HTTP {e.resp.status}. Reintento en {backoff:.1f}s...", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            # 400 por URL mal formada o no perteneciente a la propiedad
            print(f"[ERROR] {page_url} → {e}", file=sys.stderr)
            break

        rows = resp.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < ROW_LIMIT:
            break
        start_row += ROW_LIMIT

        if start_row > 5_000_000:
            print(f"[WARN] {page_url} → límite de paginación alcanzado.", file=sys.stderr)
            break

    return all_rows


# ---------------- Main ----------------

def main():
    # Configuración
    site_url = env_or_fail("GSC_SITE_URL").strip()          # ej. "sc-domain:prodesco.es"
    source_spreadsheet_id = env_or_fail("SOURCE_SPREADSHEET_ID").strip()
    source_tab_name = os.getenv("SOURCE_TAB_NAME", "").strip() or None

    target_spreadsheet_id = os.getenv("TARGET_SPREADSHEET_ID", "").strip() or source_spreadsheet_id
    output_tab_all = os.getenv("OUTPUT_TAB_ALL", "URL_Queries")
    output_tab_top = os.getenv("OUTPUT_TAB_TOP", "TopQueryPorURL")

    search_type = os.getenv("GSC_SEARCH_TYPE", "web")
    data_state = os.getenv("DATA_STATE", "final").lower()
    if data_state not in ("final", "all"):
        data_state = "final"

    start_date, end_date = get_dates_last_16_months()
    print(f"[INFO] Rango: {start_date} → {end_date} | site: {site_url} | searchType: {search_type} | dataState: {data_state}")

    # Autenticación y servicios
    scopes = [
        "https://www.googleapis.com/auth/webmasters.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = build_credentials(scopes)
    gsc, sheets = build_services(creds)

    # 1) Leer URLs fuente
    urls = read_url_column(sheets, source_spreadsheet_id, source_tab_name)
    if not urls:
        print("[ERROR] No se encontraron URLs en la hoja fuente.", file=sys.stderr)
        sys.exit(5)
    print(f"[INFO] URLs a procesar: {len(urls)}")

    # 2) Descargar queries por URL
    all_values = [["URL", "Clicks", "Impressions", "Query"]]
    top_values = [["URL", "Clicks", "Impressions", "Query"]]

    for i, u in enumerate(urls, start=1):
        print(f"[INFO] ({i}/{len(urls)}) Consultando: {u}")
        rows = fetch_queries_for_page(gsc, site_url, u, start_date, end_date, search_type=search_type, data_state=data_state)

        # Convertir a filas para Sheets
        per_url = []
        for r in rows:
            query = r["keys"][0] if r.get("keys") else ""
            clicks = int(round(r.get("clicks", 0)))
            impressions = int(round(r.get("impressions", 0)))
            per_url.append([u, clicks, impressions, query])

        # Ordenar por clicks desc y añadir a la salida global
        per_url.sort(key=lambda x: x[1], reverse=True)
        all_values.extend(per_url)

        if per_url:
            top_values.append(per_url[0])

    # 3) Escribir resultados
    # 3.1 Todas las queries
    sheet_id_all = ensure_sheet_tab(sheets, target_spreadsheet_id, output_tab_all)
    ensure_grid_size(sheets, target_spreadsheet_id, sheet_id_all, len(all_values), len(all_values[0]))
    clear_sheet(sheets, target_spreadsheet_id, output_tab_all)
    print(f"[INFO] Escribiendo {len(all_values)-1} filas en '{output_tab_all}'...")
    write_values_batched(sheets, target_spreadsheet_id, output_tab_all, all_values)

    # 3.2 Top query por URL
    sheet_id_top = ensure_sheet_tab(sheets, target_spreadsheet_id, output_tab_top)
    ensure_grid_size(sheets, target_spreadsheet_id, sheet_id_top, len(top_values), len(top_values[0]))
    clear_sheet(sheets, target_spreadsheet_id, output_tab_top)
    print(f"[INFO] Escribiendo {len(top_values)-1} filas en '{output_tab_top}'...")
    write_values_batched(sheets, target_spreadsheet_id, output_tab_top, top_values)

    print("[OK] Finalizado.")

if __name__ == "__main__":
    main()
