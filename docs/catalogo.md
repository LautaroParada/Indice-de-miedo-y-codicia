# Explorar el catálogo completo de la BDE

El catálogo se distribuye dentro de `bcch-sdk`. Describe **30.920 filas de contexto y
25.291 códigos de serie distintos**. Hay 3.942 códigos con más de un contexto, porque
una misma serie puede aparecer en varios cuadros. Los resultados conservan esos
contextos: una fila de catálogo no equivale a una serie adicional.

La fuente es `catalogo_series_bde_2026-10-29.xlsx`, entregado para esta actualización.
La etiqueta del snapshot es **2026-10-29**, tomada del nombre del archivo; no debe
interpretarse como fecha de validación de todos sus endpoints. La consulta de las
fuentes del indicador se realizó el **2026-10-28**, hora de Chile.

SHA-256 del archivo fuente:

```text
6a411f87ac7b7a1b1cb630b91f898b10e8ca3e00abd0ab868606b2aa0eb71936
```

## Cobertura

| Frecuencia | Filas de contexto |
|---|---:|
| Diaria | 2.101 |
| Mensual | 10.751 |
| Trimestral | 10.393 |
| Anual | 7.675 |
| Total | 30.920 |

La vigencia declarada es: 19.925 filas vigentes, 10.800 históricas y 195 descontinuadas.
Hay 61 filas sin enlace API y sin fecha inicial. Que un código figure en el archivo
no garantiza que `GetSeries` responda hoy ni que su última observación sea reciente.
La descarga real es la comprobación de disponibilidad.

## Recorridos por los 20 capítulos

Los filtros por capítulo aceptan fragmentos de texto, sin distinguir acentos ni
mayúsculas. Las siguientes búsquedas permiten recorrer el catálogo sin convertir
todas sus series en componentes del indicador.

| Área | Fragmento para `--capitulo` | Uso descriptivo posible |
|---|---|---|
| Cuentas nacionales | `Cuentas Nacionales` | Actividad, gasto, inversión y sectores |
| Sector externo | `Sector Externo` | Comercio, balanza de pagos y posición internacional |
| Precios | `Precios` | Inflación y estructura de precios |
| Tasas de interés | `Tasas` | Tasas de política, mercado y crédito |
| Dólar y tipos de cambio | `Tipos de Cambio` | Monedas y tipos de cambio reales |
| Mercado de capitales | `Mercado de Capitales` | Índices y variables financieras disponibles |
| Estadísticas monetarias y financieras | `Monetarias` | Dinero, depósitos y colocaciones |
| Derivados y spot | `Derivados` | Operaciones cambiarias por segmento |
| Operaciones de mercado abierto | `Mercado Abierto` | Operaciones e instrumentos del BCCh |
| Expectativas económicas | `Expectativas` | Encuestas y distribuciones de respuestas |
| Mercado laboral y demografía | `Laboral` | Empleo, salarios y población |
| Finanzas públicas | `Finanzas Públicas` | Ingresos, gastos y deuda |
| Indicadores sectoriales | `Sectoriales` | Producción y ventas por actividad |
| Mercado de vivienda | `Vivienda` | Precios y actividad inmobiliaria |
| Economía internacional | `Internacional` | Contexto internacional publicado en la BDE |
| Estadísticas experimentales | `Experimentales` | Mediciones nuevas con sus límites metodológicos |
| Series históricas | `Históricas` | Contexto de largo plazo; revisar continuidad |
| Principales indicadores macroeconómicos | `Principales` | Síntesis macroeconómica |
| Estadísticas de género | `Género` | Desagregaciones por sexo o género según la fuente |
| Estadísticas regionales | `Regionales` | Comparaciones territoriales |

La disponibilidad de series internacionales en la BDE no requiere consultar otro
servicio: el único proveedor de descarga del programa es el BCCh.

## Búsquedas y exportación

```bash
python indice.py catalogo "" --capitulo "Expectativas" --frecuencia mensual --limite 50
python indice.py catalogo "" --capitulo "Regionales" --frecuencia trimestral --limite 50
python indice.py catalogo "" --capitulo "Precios" --frecuencia mensual --salida outputs/precios.json
python indice.py catalogo "" --estado "Histórico" --limite 50
python indice.py catalogo "" --estado "" --limite 25291 --salida outputs/catalogo-completo.json
```

El estado predeterminado es `Vigente`. `--estado ""` incluye todos los estados.
`--limite` limita códigos, no contextos. La última orden exporta el catálogo completo
del snapshot, sin llamadas API. Para una exploración programática:

```python
from bcch import SeriesCatalog

catalog = SeriesCatalog()
print(catalog.metadata)
series = catalog.search("cobre", frequency="monthly", status="Vigente", limit=10)
for entry in series:
    print(entry.code, entry.contexts[0].name)
```

Selecciona después un código y descárgalo con `python indice.py serie CODIGO`.
Verifica unidad, base, ajuste estacional, frecuencia, fuente y vigencia antes de
compararlo con otra serie. Una serie mensual no contiene observaciones diarias;
este programa no interpola ni rellena sus valores para construirlas.

## Actualización del snapshot

El archivo comprimido y su metadata se mantienen en el repositorio `bcch-sdk`, mediante
`scripts/update_catalog.py`. El importador verifica recuentos y reconstruye los enlaces
con el marcador literal `TOKEN`, sin copiar posibles credenciales del Excel. El proyecto
del índice fija una revisión concreta del SDK para que el catálogo no cambie silenciosamente.

`SearchSeries` permite contrastar el catálogo vivo de la API, pero sus resultados pueden
diferir de este archivo. El SDK admite `ND` como fecha ausente en metadata; sigue
rechazando fechas inválidas en las observaciones de `GetSeries`.
