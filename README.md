# WooCommerce Redirect Matcher + GSC (v2, SEO-tuned)

Cambios principales frente a v1:
- Ponderación de queries por clicks+impressions+position e IDF por ambigüedad.
- Variaciones `page equals` para GSC (https/www + slash/no slash); fallback `contains(slug)`.
- Más señales SEO: último segmento, tipo (product/category), idioma, SKU.
- N-gramas (bigramas por defecto) y sinónimos configurables.
- Margen mínimo entre top1 y top2 para evitar empates sospechosos.
- Pestaña `Debug` opcional con top-3 candidatos y descomposición del score.
- Auto-resize de grid en Sheets y escritura batched.

Uso rápido:
1) `pip install -r requirements.txt`
2) Exporta env vars mínimas (ver README general) y ejecuta:
   `python match_woocommerce_redirects_gsc.py`
