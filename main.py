#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSC → Google Sheets | Render cron | 16 meses | page + query | prodesco.es
-------------------------------------------------------------------------
• Auth: Service Account (JSON en env var GOOGLE_CREDENTIALS_JSON o GOOGLE_CREDENTIALS_BASE64)
• Search type: "web" (sin filtros de idioma/dispositivo)
• Salida (ordenado por Clicks desc): URL | Clicks | Impressions | Query

ENV requeridas en Render:
  - GOOGLE_CREDENTIALS_JSON   (pegar JSON completo)  **o**
  - GOOGLE_CREDENTIALS_BASE64 (JSON codificado en base64)
  - GSC_SITE_URL              (ej. 'sc-domain:prodesco.es' o 'https://prodesco.es/')
  - GS_SPREADSHEET_ID         (ID de tu Google Sheet)
  - GS_SHEET_NAME             (opcional, por defecto 'SC_16m')
  - GSC_SEARCH_TYPE           (opcional, por defecto 'web')

Dependencias: ver requirements.txt
"""
import os
import sys
import json
import time
import base64
import datetime as dt
from typing import List, Dict

from dateutil.relativedelta import relativedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


ROW_LIMIT = 25000          # Máximo por página de la API de GSC
BATCH_WRITE_ROWS = 10000   # Lote de escritura a Sheets


def env_or_fail(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Falta variable de entorno: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def get_dates_last_16_months():
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


def fetch_page_query_rows(gsc, site_url: str, start_date: str, end_date: str, search_type: str = "web") -> List[Dict]:
    """Descarga todas las filas (page, query) con paginación startRow."""
    all_rows: List[Dict] = []
    start_row = 0
    backoff = 1.0

    while True:
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["page", "query"],
            "rowLimit": ROW_LIMIT,
            "startRow": start_row,
            "searchType": search_type,
        }
        try:
            resp = gsc.searchanalytics().query(siteUrl=site_url, body=body).execute()
            backoff = 1.0  # reset si OK
        except HttpError as e:
            if e.resp.status in (429, 500, 503):
                print(f"[WARN] HTTP {e.resp.status}. Reintentando en {backoff:.1f}s...", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            raise

        rows = resp.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < ROW_LIMIT:
            # última página
            break
        start_row += ROW_LIMIT

        if start_row > 5_000_000:
            print("[WARN] Se alcanzó límite de seguridad de paginación.", file=sys.stderr)
            break

    return all_rows


def to_values(rows: List[Dict]) -> List[List]:
    """Convierte las filas a matriz para Sheets (orden por clicks desc)."""
    values = [["URL", "Clicks", "Impressions", "Query"]]
    for r in rows:
        keys = r.get("keys", [])
        if len(keys) != 2:
            continue
        page, query = keys
        clicks = int(round(r.get("clicks", 0)))
        impressions = int(round(r.get("impressions", 0)))
        values.append([page, clicks, impressions, query])

    header, data = values[0], values[1:]
    data.sort(key=lambda x: x[1], reverse=True)
    return [header] + data


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


def clear_sheet(sheets, spreadsheet_id: str, sheet_name: str):
    """Limpia el rango A:Z de la pestaña (ignora si está vacía)."""
    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A:Z",
        body={}
    ).execute()


def a1(row, col):
    """Convierte fila/col (1-index) a notación A1."""
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


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


def main():
    # Variables de entorno
    site_url = os.getenv("GSC_SITE_URL", "").strip()
    if not site_url:
        print("[ERROR] Debes definir GSC_SITE_URL (ej. 'sc-domain:prodesco.es')", file=sys.stderr)
        sys.exit(2)

    spreadsheet_id = os.getenv("GS_SPREADSHEET_ID", "").strip()
    if not spreadsheet_id:
        print("[ERROR] Debes definir GS_SPREADSHEET_ID", file=sys.stderr)
        sys.exit(2)

    sheet_name = os.getenv("GS_SHEET_NAME", "SC_16m")
    search_type = os.getenv("GSC_SEARCH_TYPE", "web")

    # Fechas
    start_date, end_date = get_dates_last_16_months()
    print(f"[INFO] Rango: {start_date} → {end_date}  |  site: {site_url}  |  searchType: {search_type}")

    # Credenciales y servicios
    scopes = [
        "https://www.googleapis.com/auth/webmasters.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = build_credentials(scopes)
    gsc, sheets = build_services(creds)

    # Descarga
    print("[INFO] Descargando datos de GSC...")
    rows = fetch_page_query_rows(gsc, site_url, start_date, end_date, search_type=search_type)
    print(f"[INFO] Filas descargadas: {len(rows)}")

    if not rows:
        print("[ERROR] Cero filas. Revisa permisos de la Service Account en Search Console y el SITE_URL.", file=sys.stderr)
        sys.exit(10)

    # Transformar
    values = to_values(rows)

    # Preparar hoja
    ensure_sheet_tab(sheets, spreadsheet_id, sheet_name)
    clear_sheet(sheets, spreadsheet_id, sheet_name)

    # Escribir
    print(f"[INFO] Escribiendo {len(values)-1} filas en '{sheet_name}'...")
    write_values_batched(sheets, spreadsheet_id, sheet_name, values)

    print("[OK] Finalizado.")

if __name__ == "__main__":
    main()
