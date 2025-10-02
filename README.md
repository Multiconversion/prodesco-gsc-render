# GSC → Google Sheets (Render cron) | prodesco.es

Extrae **últimos 16 meses** de **page + query** desde Google Search Console y lo escribe en Google Sheets.
Formato final (**ordenado por clics desc**):

```
URL | Clicks | Impressions | Query
```

## 0) Datos de tu Service Account (confirmación)
Tu JSON indica:
- `project_id`: **conectores-468816**
- `client_email`: **integracion-search-console@conectores-468816.iam.gserviceaccount.com**
(estos datos vienen del archivo JSON de tu Service Account)

> Asegúrate de añadir ese **client_email** como usuario en Search Console y como **Editor** en tu Google Sheet.

## 1) Habilitar APIs en Google Cloud
Con el proyecto **conectores-468816**:
- Habilita **Search Console API**
- Habilita **Google Sheets API**

## 2) Permisos
- **Search Console**: en la propiedad `sc-domain:prodesco.es` (o el prefijo exacto `https://prodesco.es/`) añade el usuario **integracion-search-console@conectores-468816.iam.gserviceaccount.com** con acceso de lectura (Restringido o Completo).
- **Google Sheets**: comparte la hoja con ese mismo email como **Editor**.

## 3) Variables de entorno (Render)
En Render, en tu Cron Job, define:
- `GOOGLE_CREDENTIALS_JSON` → contenido **completo** del JSON de la Service Account (pegar tal cual, con saltos de línea).
  - Alternativamente puedes usar `GOOGLE_CREDENTIALS_BASE64` con el JSON en base64.
- `GSC_SITE_URL` → `sc-domain:prodesco.es` (recomendado) o `https://prodesco.es/` si tu propiedad es de prefijo de URL.
- `GS_SPREADSHEET_ID` → `1YIX6aITtyZISK99g_QXot3oBGShJWHanxFwZDRvO3F4`
- `GS_SHEET_NAME` → `SC_16m` (o el que prefieras)
- `GSC_SEARCH_TYPE` → `web` (por defecto)

> El archivo `render.yaml` ya deja `GSC_SITE_URL`, `GS_SPREADSHEET_ID`, `GS_SHEET_NAME` y `GSC_SEARCH_TYPE`. Sólo te falta añadir la variable **GOOGLE_CREDENTIALS_JSON** desde el dashboard de Render.

## 4) Deploy en Render
1. Sube este repo (estos 3 archivos) a GitHub.
2. Render → **New + → Cron Job** → conecta el repo.
3. Render usará:
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `python main.py`
   - **Schedule**: `0 6 * * *` (diario, 06:00 UTC)
4. Añade las variables de entorno mencionadas arriba.

## 5) Probar
En la página del cron en Render → **Run Job** para ejecutar ahora y ver logs.

### Logs esperados
- Rango de fechas (16 meses hasta *hoy-2*)
- Total de filas descargadas
- Filas escritas en la pestaña

## 6) Notas
- Si usas `sc-domain:prodesco.es`, asegúrate de que la Service Account esté añadida **en esa propiedad** de Search Console.
- Si prefieres `https://prodesco.es/`, usa exactamente la URL registrada en GSC.
- El script pagina a 25k filas y reintenta (backoff) en 429/5xx.
- No incluye CTR ni Position (según tu petición).

## 7) Ejecución local (debug opcional)
```bash
pip install -r requirements.txt
export GOOGLE_CREDENTIALS_JSON="$(cat /ruta/tu-service-account.json)"
export GSC_SITE_URL="sc-domain:prodesco.es"
export GS_SPREADSHEET_ID="1YIX6aITtyZISK99g_QXot3oBGShJWHanxFwZDRvO3F4"
export GS_SHEET_NAME="SC_16m"
python main.py
```
