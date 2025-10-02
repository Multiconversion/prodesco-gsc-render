#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WooCommerce Redirect Matcher + Search Console (v2, SEO-tuned)
=============================================================
Objetivo: máxima precisión para concordar **URLs antiguas** → **URLs nuevas** (WooCommerce)
usando señales mixtas que Google entiende bien:

1) Tokens del **path/slug** de la URL antigua (más peso en el último segmento).
2) **Consultas de Search Console por URL** (16 meses, dataState configurable):
   - Ponderación por **clics**, **impresiones** y **posición** (mejor pos ⇒ más peso).
   - Penalización **IDF** por ambigüedad (si una **misma query** aparece en muchas URLs antiguas).
   - Tokens + **n-gramas** opcionales para captar compuestos (zapatillas running).
3) Señales de producto/categoría de WooCommerce:
   - Nombre, slug, categorías, SKU y **path**.
4) Boosts y reglas SEO:
   - Coincidencia exacta del **slug final** (último segmento).
   - **SKU** presente en la URL antigua.
   - **Idioma** por subcarpeta (/es/, /en/) con pistas configurables.
   - **Tipo** (product vs category) estimado por el path antiguo.
5) Desambiguación final:
   - **Margen mínimo** entre el mejor y el segundo candidato (para evitar empates ~).
   - Umbral de **score mínimo**.
   - Notas con nº de queries y nº de queries ambiguas (para auditoría).

Salida (Google Sheets, mismo spreadsheet de origen):
- `Matches`     → old_url | new_url | type | score | name | slug | sku | id | notes
- `Unmatched`   → old_url | reason
- `Redirects_301` → líneas 'Redirect 301 /old-path https://tusitio.com/new-path'
- (opcional) `Debug` → top-3 candidatos con sus componentes de score

Además, genera `redirects_301.txt` con las reglas 301 para Apache.

ENV requeridas (Render/local):
  # Credenciales + Search Console
  - GOOGLE_CREDENTIALS_JSON    (o GOOGLE_CREDENTIALS_BASE64)
  - GSC_SITE_URL               (p.ej. 'sc-domain:prodesco.es' o 'https://prodesco.es/')
  - DATA_STATE                 (opcional: 'final' | 'all'; por defecto 'final')
  - GSC_SEARCH_TYPE            (opcional: 'web'; default)
  - DATE_RANGE_MONTHS          (opcional: por defecto 16)
  - END_DATE_OFFSET_DAYS       (opcional: por defecto 2)
  - MAX_QUERIES_PER_URL        (opcional: por defecto 200)
  - USE_PAGE_CONTAINS_FALLBACK (opcional: '1' para activar fallback 'contains')

  # Google Sheets
  - SOURCE_SPREADSHEET_ID      (ID del Sheet con columna 'URL')
  - SOURCE_TAB_NAME            (opcional: si no, primera pestaña)

  # WooCommerce
  - WOO_BASE_URL               (p.ej. 'https://prodesco.es')
  - WOO_CONSUMER_KEY
  - WOO_CONSUMER_SECRET

  # Scoring y control de calidad
  - MIN_SCORE                  (0-100; default 65)
  - MIN_MARGIN                 (0-100; diferencia mínima con 2º candidato; default 5)
  - URL_WEIGHT                 (0..1; default 0.35)
  - QUERY_WEIGHT               (0..1; default 0.65)  # URL_WEIGHT + QUERY_WEIGHT ≈ 1
  - LANG_HINTS                 ('es,en', etc.)
  - MAX_CANDIDATES             (default 400)
  - USE_NGRAMS                 (0, 2 o 3; default 2 → usa bigramas además de tokens)
  - TOKEN_SYNONYMS_JSON        (JSON opcional de sinónimos: {"zapatillas":["tenis","playeras"]})
  - WRITE_DEBUG_SHEET          ('1' para crear pestaña Debug con top-3)

Limitaciones:
- GSC oculta queries anónimas por privacidad (no recuperables).
- Si la URL antigua no pertenece a la propiedad de GSC, no habrá datos (se usa solo URL matching).
"""

import os
import sys
import re
import json
import time
import base64
import math
import unicodedata
import datetime as dt
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse

import requests
from dateutil.relativedelta import relativedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------- Config global ----------

ROW_LIMIT_WOO = 100   # Woo REST per_page máx = 100
TIMEOUT = 30

STOPWORDS = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el la los las en entre era erais eramos eran eres es esa esas ese esos esta estaba estado estaban estamos estan estar este estos para pero por porque que se sin sobre su sus tras un una uno y ya
the to of for in on at and or from as by with your our their is are was were this that these those it
producto productos categoria categorias tienda shop comprar oferta ofertas precio precios baratos barato nueva nuevo nuevas nuevos sale descuento""".split())

# ---------- Utilidades ----------

def env_or_fail(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] Falta variable de entorno: {name}", file=sys.stderr)
        sys.exit(2)
    return v

def load_credentials_json_from_env() -> dict:
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
            try:
                fixed = raw_json.replace('\\n', '\n')
                data = json.loads(fixed)
            except Exception as e:
                print(f"[ERROR] GOOGLE_CREDENTIALS_JSON inválido: {e}", file=sys.stderr)
                sys.exit(3)
    if data is None:
        print("[ERROR] Debes definir GOOGLE_CREDENTIALS_JSON o GOOGLE_CREDENTIALS_BASE64", file=sys.stderr)
        sys.exit(3)
    return data

def build_services(creds_scopes: List[str]):
    info = load_credentials_json_from_env()
    creds = service_account.Credentials.from_service_account_info(info, scopes=creds_scopes)
    try:
        gsc = build("searchconsole", "v1", credentials=creds, cache_discovery=False)
    except Exception:
        gsc = build("webmasters", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return gsc, sheets

def a1(row, col):
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text

def tokenize(s: str) -> List[str]:
    s = normalize(s)
    tokens = re.split(r"[^a-z0-9]+", s)
    return [t for t in tokens if t and t not in STOPWORDS]

def make_ngrams(tokens: List[str], n: int) -> List[str]:
    if n < 2:
        return []
    grams = []
    for i in range(len(tokens)-n+1):
        grams.append("_".join(tokens[i:i+n]))
    return grams

def tokens_from_url(url: str, use_ngrams: int = 2) -> List[str]:
    try:
        path = urlparse(url).path or ""
    except Exception:
        path = url
    toks = tokenize(path)
    if use_ngrams >= 2:
        toks += make_ngrams(toks, 2)
    if use_ngrams >= 3:
        toks += make_ngrams(toks, 3)
    return toks

def last_segment(url: str) -> str:
    try:
        path = urlparse(url).path or ""
        segs = [s for s in path.split("/") if s]
        return segs[-1] if segs else ""
    except Exception:
        return ""

def guess_language_code_from_path(path: str, lang_hints: List[str]) -> Optional[str]:
    m = re.search(r"/([a-z]{2})/", path + "/")
    if m:
        code = m.group(1)
        if code in lang_hints:
            return code
    return None

def guess_type_from_old_path(path: str) -> Optional[str]:
    path = path.lower()
    if any(x in path for x in ("/product-category", "/categoria", "/category")):
        return "category"
    if any(x in path for x in ("/product", "/producto", "/tienda/")):
        return "product"
    return None

# ---------- Sheets helpers ----------

def sheets_get_first_title(sheets, spreadsheet_id: str) -> str:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets_meta = meta.get("sheets", [])
    if not sheets_meta:
        print("[ERROR] El spreadsheet no tiene pestañas.", file=sys.stderr)
        sys.exit(4)
    return sheets_meta[0]["properties"]["title"]

def sheets_read_url_column(sheets, spreadsheet_id: str, source_tab_name: Optional[str]) -> List[str]:
    if not source_tab_name:
        source_tab_name = sheets_get_first_title(sheets, spreadsheet_id)
    header_resp = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=f"{source_tab_name}!A1:Z1"
    ).execute()
    headers = header_resp.get("values", [[]])
    headers = headers[0] if headers else []
    url_col_idx = None
    for idx, name in enumerate(headers, start=1):
        if str(name).strip().lower() == "url":
            url_col_idx = idx; break
    rng = f"{source_tab_name}!A2:A" if url_col_idx is None else f"{source_tab_name}!{a1(1, url_col_idx)[:-1]}2:{a1(1, url_col_idx)[:-1]}"
    resp = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng).execute()
    urls = [row[0].strip() for row in resp.get("values", []) if row and row[0].strip()]
    # de-duplicar preservando orden
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def sheets_ensure_tab(sheets, spreadsheet_id: str, sheet_name: str) -> int:
    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        if props.get("title") == sheet_name:
            return props.get("sheetId")
    reqs = [{"addSheet": {"properties": {"title": sheet_name}}}]
    resp = sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": reqs}).execute()
    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]

def sheets_ensure_grid(sheets, spreadsheet_id: str, sheet_id: int, needed_rows: int, needed_cols: int):
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

def sheets_clear(sheets, spreadsheet_id: str, sheet_name: str):
    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id, range=f"{sheet_name}!A:Z", body={}
    ).execute()

def sheets_write_batched(sheets, spreadsheet_id: str, sheet_name: str, values: List[List], batch_size=10000):
    num_cols = len(values[0]) if values else 0
    start = 0
    row_index = 1
    while start < len(values):
        chunk = values[start:start + batch_size]
        end_row = row_index + len(chunk) - 1
        rng = f"{sheet_name}!{a1(row_index, 1)}:{a1(end_row, num_cols)}"
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=rng,
            valueInputOption="RAW",
            body={"values": chunk}
        ).execute()
        start += batch_size
        row_index = end_row + 1

# ---------- WooCommerce fetch ----------

def woo_get_products(base_url: str, ck: str, cs: str) -> List[Dict]:
    items = []
    page = 1
    while True:
        params = {"consumer_key": ck, "consumer_secret": cs, "per_page": ROW_LIMIT_WOO, "page": page, "status": "publish"}
        url = f"{base_url.rstrip('/')}/wp-json/wc/v3/products"
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"Woo products HTTP {r.status_code}: {r.text[:200]}")
        batch = r.json()
        if not batch: break
        for p in batch:
            items.append({
                "type": "product",
                "id": p.get("id"),
                "name": p.get("name") or "",
                "slug": p.get("slug") or "",
                "sku": p.get("sku") or "",
                "permalink": p.get("permalink") or "",
                "categories": [c.get("name","") for c in p.get("categories", [])]
            })
        page += 1
    return items

def woo_get_product_cats(base_url: str, ck: str, cs: str) -> List[Dict]:
    items = []
    page = 1
    while True:
        params = {"per_page": ROW_LIMIT_WOO, "page": page}
        url = f"{base_url.rstrip('/')}/wp-json/wp/v2/product_cat"
        r = requests.get(url, params=params, timeout=TIMEOUT, auth=(ck, cs))
        if r.status_code == 401:
            r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"WP product_cat HTTP {r.status_code}: {r.text[:200]}")
        batch = r.json()
        if not batch: break
        for c in batch:
            items.append({
                "type": "category",
                "id": c.get("id"),
                "name": c.get("name") or "",
                "slug": c.get("slug") or "",
                "permalink": c.get("link") or ""
            })
        page += 1
    return items

# ---------- GSC helpers ----------

def gsc_get_dates(months_back: int, end_offset_days: int) -> Tuple[str, str]:
    today = dt.date.today()
    end_date = today - dt.timedelta(days=end_offset_days)
    start_date = end_date - relativedelta(months=months_back)
    return start_date.isoformat(), end_date.isoformat()

def host_variants_for_site(site_url_env: str) -> List[str]:
    """Genera hosts candidatos para el equals, a partir del site_url env."""
    if site_url_env.startswith("sc-domain:"):
        dom = site_url_env.split(":",1)[1]
        return [f"https://{dom}", f"https://www.{dom}"]
    # Prefijo de URL
    try:
        p = urlparse(site_url_env)
        base = f"{p.scheme}://{p.netloc}"
        # también considerar www/non-www si aplica
        hosts = {base}
        if p.netloc.startswith("www."):
            hosts.add(f"{p.scheme}://{p.netloc[4:]}")
        else:
            hosts.add(f"{p.scheme}://www.{p.netloc}")
        return list(hosts)
    except Exception:
        return [site_url_env]

def gsc_page_variations(old_url: str, site_url_env: str) -> List[str]:
    """Genera variaciones razonables de la URL para filtrar por 'page equals' en GSC."""
    path = urlparse(old_url).path or "/"
    path_noslash = path[:-1] if path.endswith("/") and path != "/" else path + "/"
    candidates = set()
    for host in host_variants_for_site(site_url_env):
        candidates.add(host + path)
        candidates.add(host + path_noslash)
    return list(candidates)

def gsc_fetch_queries_for_page(gsc, site_url: str, page_url: str, start_date: str, end_date: str,
                               search_type: str = "web", data_state: str = "final", row_limit: int = 25000) -> List[Dict]:
    """Consulta GSC por queries de una página concreta (operator 'equals'). Pagina con startRow."""
    all_rows = []
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
                    "operator": "equals",
                    "expression": page_url
                }]
            }],
            "rowLimit": row_limit,
            "startRow": start_row,
            "searchType": search_type,
            "dataState": data_state
        }
        try:
            resp = gsc.searchanalytics().query(siteUrl=site_url, body=body).execute()
            backoff = 1.0
        except HttpError as e:
            if e.resp.status in (429, 500, 503):
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            # Otros errores ⇒ corta
            # print(f"[WARN] equals error: {e}")
            break

        rows = resp.get("rows", [])
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < row_limit:
            break
        start_row += row_limit
        if start_row > 5_000_000:
            print(f"[WARN] {page_url} → límite paginación GSC", file=sys.stderr)
            break
    return all_rows

def gsc_fetch_queries_any(gsc, site_url: str, old_url: str, site_url_env: str,
                          start_date: str, end_date: str, search_type: str, data_state: str,
                          row_limit: int = 25000, use_contains_fallback: bool = True) -> List[Dict]:
    """Intenta equals con variaciones razonables; si nada, usa contains(slug)."""
    # equals (todas variaciones)
    variations = gsc_page_variations(old_url, site_url_env)
    for v in variations:
        rows = gsc_fetch_queries_for_page(gsc, site_url, v, start_date, end_date, search_type, data_state, row_limit)
        if rows:
            return rows
    # contains(slug) (fallback)
    if use_contains_fallback:
        path = urlparse(old_url).path or ""
        segs = [s for s in path.split("/") if s]
        slug = segs[-1] if segs else ""
        if slug:
            start_row = 0
            backoff = 1.0
            all_rows = []
            while True:
                body = {
                    "startDate": start_date,
                    "endDate": end_date,
                    "dimensions": ["query"],
                    "dimensionFilterGroups": [{
                        "groupType": "and",
                        "filters": [{
                            "dimension": "page",
                            "operator": "contains",
                            "expression": slug
                        }]
                    }],
                    "rowLimit": row_limit,
                    "startRow": start_row,
                    "searchType": search_type,
                    "dataState": data_state
                }
                try:
                    resp = gsc.searchanalytics().query(siteUrl=site_url, body=body).execute()
                    backoff = 1.0
                except HttpError as e:
                    if e.resp.status in (429, 500, 503):
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue
                    break
                rows = resp.get("rows", [])
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < row_limit:
                    break
                start_row += row_limit
                if start_row > 5_000_000:
                    break
            if all_rows:
                return all_rows
    return []

# ---------- Indexado y sinónimos ----------

def parse_synonyms_env() -> Dict[str, List[str]]:
    raw = os.getenv("TOKEN_SYNONYMS_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        norm = {}
        for k, vs in data.items():
            key = normalize(k)
            vals = [normalize(v) for v in (vs if isinstance(vs, list) else [vs])]
            norm[key] = vals
        return norm
    except Exception:
        return {}

def expand_with_synonyms(tokens: List[str], synonyms: Dict[str, List[str]]) -> List[str]:
    if not synonyms:
        return tokens
    out = list(tokens)
    for t in tokens:
        if t in synonyms:
            out.extend(synonyms[t])
    return out

def text_for_item(item: Dict) -> str:
    if item["type"] == "product":
        cats = " ".join(item.get("categories", []))
        return f"{item.get('name','')} {item.get('slug','')} {cats} {item.get('sku','')} {urlparse(item.get('permalink','')).path}"
    else:
        return f"{item.get('name','')} {item.get('slug','')} {urlparse(item.get('permalink','')).path}"

def build_inverted_index(items: List[Dict], use_ngrams: int, synonyms: Dict[str, List[str]]):
    inv = {}
    item_tokens = []
    item_texts = []
    for idx, it in enumerate(items):
        txt = text_for_item(it)
        toks = tokenize(txt)
        if use_ngrams >= 2:
            toks += make_ngrams(toks, 2)
        if use_ngrams >= 3:
            toks += make_ngrams(toks, 3)
        toks = expand_with_synonyms(toks, synonyms)
        toks_set = set(toks)
        item_tokens.append(toks_set)
        item_texts.append(txt)
        for t in toks_set:
            inv.setdefault(t, set()).add(idx)
    return inv, item_tokens, item_texts

# ---------- Scoring ----------

def build_query_token_weights(query_rows: List[Dict], query_occurrence: Dict[str, int],
                              use_ngrams: int, synonyms: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Peso para tokens de queries de UNA URL:
    w = (0.6*log1p(clicks) + 0.3*log1p(impressions) + 0.1*pos_weight) * idf
    idf = log(1 + N / occ_q) con N = total_urls (se aplica a nivel de query)
    Se reparte el peso entre tokens (y n-gramas si USE_NGRAMS>1).
    """
    weights = {}
    # Sumatorio bruto por token
    for r in query_rows:
        q = r.get("keys", [""])[0] if r.get("keys") else ""
        if not q:
            continue
        clicks = float(r.get("clicks", 0.0) or 0.0)
        impressions = float(r.get("impressions", 0.0) or 0.0)
        position = float(r.get("position", 0.0) or 0.0)
        if clicks <= 0 and impressions <= 0:
            continue
        # pos_weight: 1 en pos<=1, 0 en pos>=40 (aprox)
        pos_weight = max(0.0, min(1.0, (40.0 - min(40.0, position)) / 40.0))
        base = 0.6*math.log1p(clicks) + 0.3*math.log1p(impressions) + 0.1*pos_weight

        occ = max(1, int(query_occurrence.get(q, 1)))
        idf = math.log(1.0 + (1.0 * query_occurrence.get("__N_URLS__", 1)) / occ)

        toks = tokenize(q)
        if use_ngrams >= 2:
            toks += make_ngrams(toks, 2)
        if use_ngrams >= 3:
            toks += make_ngrams(toks, 3)
        toks = expand_with_synonyms(toks, synonyms)

        if not toks:
            continue
        per_token = base * idf / len(toks)
        for t in toks:
            weights[t] = weights.get(t, 0.0) + per_token
    return weights

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def difflib_ratio(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(a=a, b=b).ratio()

def score_components(old_tokens: List[str], item_tokens: set, item_text: str,
                     q_token_weights: Dict[str, float],
                     old_url: str, item: Dict, lang_hints: List[str]) -> Dict[str, float]:
    # Peso mayor al último segmento de la URL vieja
    last = last_segment(old_url)
    old_tokens_set = set(old_tokens)
    last_tokens = set(tokenize(last))

    # URL part
    jac = jaccard(old_tokens_set, item_tokens)
    dif = difflib_ratio(" ".join(sorted(list(old_tokens_set))), item_text)
    last_boost = 0.1 if last and last_tokens and last_tokens.issubset(item_tokens) else 0.0
    url_part = 0.5*jac + 0.5*dif + last_boost  # 0..1 aprox

    # Queries coverage
    if q_token_weights:
        total_w = sum(q_token_weights.values())
        matched_w = sum(w for t, w in q_token_weights.items() if t in item_tokens)
        query_part = (matched_w / total_w) if total_w > 0 else 0.0
    else:
        query_part = 0.0

    # Boosts
    slug_eq = 0.0
    try:
        old_last = last
        new_last = [seg for seg in (urlparse(item.get("permalink","")).path or "").split("/") if seg][-1]
        if old_last and new_last and old_last == new_last:
            slug_eq = 0.08
    except Exception:
        pass

    sku_boost = 0.0
    sku = (item.get("sku") or "").strip().lower()
    if sku and sku in old_url.lower():
        sku_boost = 0.10

    lang_boost = 0.0
    lang_code = guess_language_code_from_path((urlparse(old_url).path or ""), lang_hints) if lang_hints else None
    if lang_code and f"/{lang_code}/" in (item.get("permalink") or ""):
        lang_boost = 0.05

    type_boost = 0.0
    t_guess = guess_type_from_old_path((urlparse(old_url).path or ""))
    if t_guess and t_guess == item.get("type"):
        type_boost = 0.03

    return {
        "url_part": max(0.0, min(1.0, url_part)),
        "query_part": max(0.0, min(1.0, query_part)),
        "slug_boost": slug_eq,
        "sku_boost": sku_boost,
        "lang_boost": lang_boost,
        "type_boost": type_boost
    }

def combine_score(components: Dict[str, float], url_weight: float, query_weight: float) -> float:
    base = url_weight * components["url_part"] + query_weight * components["query_part"]
    boost = components["slug_boost"] + components["sku_boost"] + components["lang_boost"] + components["type_boost"]
    score = base + boost
    return round(max(0.0, min(1.0, score)) * 100.0, 2)

# ---------- Main ----------

def main():
    # ENV
    site_url = env_or_fail("GSC_SITE_URL").strip()
    spreadsheet_id = env_or_fail("SOURCE_SPREADSHEET_ID").strip()
    source_tab = os.getenv("SOURCE_TAB_NAME", "").strip() or None

    woo_base = env_or_fail("WOO_BASE_URL").rstrip("/")
    woo_ck = env_or_fail("WOO_CONSUMER_KEY")
    woo_cs = env_or_fail("WOO_CONSUMER_SECRET")

    out_tab_matches = os.getenv("OUTPUT_TAB_MATCHES", "Matches")
    out_tab_unmatched = os.getenv("OUTPUT_TAB_UNMATCHED", "Unmatched")
    out_tab_htaccess = os.getenv("OUTPUT_TAB_HTACCESS", "Redirects_301")
    write_debug = os.getenv("WRITE_DEBUG_SHEET", "0") in ("1","true","True","yes")

    # Scoring knobs (afinados para SEO)
    min_score = float(os.getenv("MIN_SCORE", "65"))
    min_margin = float(os.getenv("MIN_MARGIN", "5"))
    url_weight = float(os.getenv("URL_WEIGHT", "0.35"))
    query_weight = float(os.getenv("QUERY_WEIGHT", "0.65"))
    lang_hints = [c.strip() for c in os.getenv("LANG_HINTS", "es,en").split(",") if c.strip()]
    max_candidates = int(os.getenv("MAX_CANDIDATES", "400"))
    use_ngrams = int(os.getenv("USE_NGRAMS", "2"))
    synonyms = parse_synonyms_env()

    months_back = int(os.getenv("DATE_RANGE_MONTHS", "16"))
    end_offset_days = int(os.getenv("END_DATE_OFFSET_DAYS", "2"))
    max_queries_per_url = int(os.getenv("MAX_QUERIES_PER_URL", "200"))
    use_contains_fallback = os.getenv("USE_PAGE_CONTAINS_FALLBACK", "1") in ("1","true","True","yes")

    data_state = os.getenv("DATA_STATE", "final").lower()
    if data_state not in ("final", "all"):
        data_state = "final"
    search_type = os.getenv("GSC_SEARCH_TYPE", "web")

    start_date, end_date = gsc_get_dates(months_back, end_offset_days)
    print(f"[INFO] Rango GSC: {start_date} → {end_date} | dataState: {data_state} | searchType: {search_type}")
    print(f"[INFO] Scoring weights → URL: {url_weight} | QUERY: {query_weight} | MIN_SCORE={min_score} | MIN_MARGIN={min_margin}")

    # Servicios
    scopes = [
        "https://www.googleapis.com/auth/webmasters.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    gsc, sheets = build_services(scopes)

    # 1) URLs antiguas
    urls = sheets_read_url_column(sheets, spreadsheet_id, source_tab)
    if not urls:
        print("[ERROR] No se encontraron URLs en la hoja fuente.", file=sys.stderr)
        sys.exit(5)
    print(f"[INFO] URLs antiguas: {len(urls)}")

    # 2) Woo catálogo
    print("[INFO] Descargando productos WooCommerce...")
    products = woo_get_products(woo_base, woo_ck, woo_cs)
    print(f"[INFO] Productos: {len(products)}")
    print("[INFO] Descargando categorías 'product_cat'...")
    cats = woo_get_product_cats(woo_base, woo_ck, woo_cs)
    print(f"[INFO] Categorías: {len(cats)}")
    items = products + cats

    # 3) Índice invertido
    inv, item_tokens_list, item_texts = build_inverted_index(items, use_ngrams, synonyms)

    # 4) Consultar queries GSC por URL
    url_queries = {}
    query_occurrence = {"__N_URLS__": len(urls)}  # para IDF
    for i, u in enumerate(urls, start=1):
        print(f"[INFO] ({i}/{len(urls)}) GSC queries: {u}")
        rows = gsc_fetch_queries_any(gsc, site_url, u, os.getenv("GSC_SITE_URL",""), start_date, end_date, search_type, data_state, 25000, use_contains_fallback)
        # orden y recorte
        rows.sort(key=lambda r: (r.get("clicks",0), r.get("impressions",0)), reverse=True)
        rows = rows[:max_queries_per_url]
        url_queries[u] = rows
        # contar ambigüedad
        seen_q = set()
        for r in rows:
            q = r.get("keys", [""])[0] if r.get("keys") else ""
            if q and q not in seen_q:
                query_occurrence[q] = query_occurrence.get(q, 0) + 1
                seen_q.add(q)

    # 5) Matching por URL
    matches = [["old_url", "new_url", "type", "score", "name", "slug", "sku", "id", "notes"]]
    unmatched = [["old_url", "reason"]]
    redirects = [["Redirect 301 lines (put in .htaccess or VirtualHost)"], ["# one rule per line"]]
    debug_rows = [["old_url","cand1_url","cand1_score","cand1_urlPart","cand1_queryPart","cand1_boosts","cand2_url","cand2_score","cand3_url","cand3_score","queries_used","ambiguous_queries"]]

    for i, old in enumerate(urls, start=1):
        print(f"[INFO] ({i}/{len(urls)}) Matching: {old}")
        # Tokens de URL antigua (con n-gramas)
        old_tokens = tokens_from_url(old, use_ngrams)
        old_token_set = set(expand_with_synonyms(old_tokens, synonyms))

        # Tokens ponderados de queries
        q_rows = url_queries.get(old, [])
        q_weights = build_query_token_weights(q_rows, query_occurrence, use_ngrams, synonyms)

        # Preselección por unión de tokens (URL + queries)
        candidate_tokens = old_token_set | set(q_weights.keys())
        # construir set por item rápidamente (use inv to fetch candidate idxs)
        candidates = set()
        for t in candidate_tokens:
            if t in inv:
                candidates |= inv[t]
        if not candidates:
            # Fallback: evalúa primeros 500
            candidates = set(range(min(len(items), 500)))

        # Calcular score de candidatos
        scored = []
        for idx in candidates:
            comps = score_components(old_tokens=list(old_token_set), item_tokens=item_tokens_list[idx], item_text=item_texts[idx],
                                     q_token_weights=q_weights, old_url=old, item=items[idx], lang_hints=lang_hints)
            s = combine_score(comps, url_weight, query_weight)
            scored.append((s, idx, comps))
        scored.sort(reverse=True, key=lambda x: x[0])
        top3 = scored[:3]

        if not top3 or top3[0][0] < min_score:
            reason = "low_score" if top3 else "no_candidates"
            if top3:
                reason += f"({top3[0][0]:.2f})"
            unmatched.append([old, reason])
            if write_debug:
                debug_rows.append([old] + ( [items[top3[0][1]]["permalink"], top3[0][0], top3[0][2]["url_part"], top3[0][2]["query_part"],
                                             top3[0][2]["slug_boost"]+top3[0][2]["sku_boost"]+top3[0][2]["lang_boost"]+top3[0][2]["type_boost"],
                                             items[top3[1][1]]["permalink"] if len(top3)>1 else "", top3[1][0] if len(top3)>1 else "",
                                             items[top3[2][1]]["permalink"] if len(top3)>2 else "", top3[2][0] if len(top3)>2 else "",
                                             len(q_rows), sum(1 for r in q_rows if query_occurrence.get((r.get('keys',[None])[0] or ''),0)>1) ] if top3 else
                                            ["","","","","","","","","", len(q_rows), 0] ))
            continue

        # margen mínimo entre top1 y top2
        margin_ok = True
        if len(top3) > 1 and (top3[0][0] - top3[1][0]) < min_margin:
            margin_ok = False

        if not margin_ok:
            unmatched.append([old, f"low_margin({top3[0][0]-top3[1][0]:.2f})"])
            if write_debug:
                debug_rows.append([old,
                                   items[top3[0][1]]["permalink"], top3[0][0], top3[0][2]["url_part"], top3[0][2]["query_part"],
                                   top3[0][2]["slug_boost"]+top3[0][2]["sku_boost"]+top3[0][2]["lang_boost"]+top3[0][2]["type_boost"],
                                   items[top3[1][1]]["permalink"] if len(top3)>1 else "", top3[1][0] if len(top3)>1 else "",
                                   items[top3[2][1]]["permalink"] if len(top3)>2 else "", top3[2][0] if len(top3)>2 else "",
                                   len(q_rows), sum(1 for r in q_rows if query_occurrence.get((r.get('keys',[None])[0] or ''),0)>1) ])
            continue

        # aceptar top1
        best_score, best_idx, best_comps = top3[0]
        it = items[best_idx]
        ambiguous = sum(1 for r in q_rows if query_occurrence.get((r.get("keys",[None])[0] or ""),0) > 1)
        notes = f"queries={len(q_rows)}; ambiguous_queries={ambiguous}; margin={ (top3[0][0]-top3[1][0]) if len(top3)>1 else 'NA'}"
        matches.append([old, it.get("permalink"), it.get("type"), best_score, it.get("name"), it.get("slug"), it.get("sku",""), it.get("id"), notes])

        # Redirect 301
        path = urlparse(old).path or "/"
        if not path.startswith("/"):
            path = "/" + path
        redirects.append([f"Redirect 301 {path} {it.get('permalink')}"])

        if write_debug:
            debug_rows.append([old,
                               it.get("permalink"), best_score, best_comps["url_part"], best_comps["query_part"],
                               best_comps["slug_boost"]+best_comps["sku_boost"]+best_comps["lang_boost"]+best_comps["type_boost"],
                               items[top3[1][1]]["permalink"] if len(top3)>1 else "", top3[1][0] if len(top3)>1 else "",
                               items[top3[2][1]]["permalink"] if len(top3)>2 else "", top3[2][0] if len(top3)>2 else "",
                               len(q_rows), ambiguous])

    # 6) Escribir a Sheets
    # Matches
    sh = sheets_ensure_tab(sheets, spreadsheet_id, out_tab_matches)
    sheets_ensure_grid(sheets, spreadsheet_id, sh, len(matches), len(matches[0]))
    sheets_clear(sheets, spreadsheet_id, out_tab_matches)
    sheets_write_batched(sheets, spreadsheet_id, out_tab_matches, matches)

    # Unmatched
    shu = sheets_ensure_tab(sheets, spreadsheet_id, out_tab_unmatched)
    sheets_ensure_grid(sheets, spreadsheet_id, shu, len(unmatched), len(unmatched[0]))
    sheets_clear(sheets, spreadsheet_id, out_tab_unmatched)
    sheets_write_batched(sheets, spreadsheet_id, out_tab_unmatched, unmatched)

    # Redirects
    shr = sheets_ensure_tab(sheets, spreadsheet_id, out_tab_htaccess)
    sheets_ensure_grid(sheets, spreadsheet_id, shr, len(redirects), len(redirects[0]))
    sheets_clear(sheets, spreadsheet_id, out_tab_htaccess)
    sheets_write_batched(sheets, spreadsheet_id, out_tab_htaccess, redirects)

    # Debug
    if write_debug:
        shd = sheets_ensure_tab(sheets, spreadsheet_id, "Debug")
        sheets_ensure_grid(sheets, spreadsheet_id, shd, len(debug_rows), len(debug_rows[0]))
        sheets_clear(sheets, spreadsheet_id, "Debug")
        sheets_write_batched(sheets, spreadsheet_id, "Debug", debug_rows)

    # 7) Archivo plano
    try:
        lines = [row[0] for row in redirects if row and row[0].startswith("Redirect 301 ")]
        with open("redirects_301.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print("[OK] Redirecciones escritas en redirects_301.txt")
    except Exception as e:
        print(f"[WARN] No se pudo escribir redirects_301.txt: {e}")

    print("[OK] Matching terminado. Revisa las pestañas en tu Google Sheet.")

if __name__ == "__main__":
    main()
