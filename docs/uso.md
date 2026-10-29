# Instalación y uso

## Requisitos

Python 3.11 o superior, Git y una cuenta BDE con acceso a la API habilitado. Todas las
consultas de datos se dirigen al Banco Central de Chile. La herramienta no requiere
cuentas ni claves de otros proveedores.

```bash
git clone https://github.com/LautaroParada/Indice-de-miedo-y-codicia.git
cd Indice-de-miedo-y-codicia
python -m venv .venv
```

Activa el entorno en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instala el proyecto:

```bash
python -m pip install -e .
```

La dependencia `bcch` se obtiene del repositorio
[bcch-sdk](https://github.com/LautaroParada/bcch-sdk), fijada a un commit específico.
No se utiliza la distribución histórica `bcch 0.0.2` de PyPI.

## Configuración local

Copia `.env.example` a `.env` en la raíz del proyecto. Escribe el token únicamente en
el archivo `.env` local, en la variable `BCCH_API_KEY`. Sigue la
[guía oficial de acceso](https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/acceso_api.html)
para obtenerlo y administrar su vigencia. No lo publiques ni lo pegues en comandos,
notebooks, capturas o incidencias.

El entorno del proceso tiene prioridad sobre `.env`. No hay interpolación de variables
dentro del archivo. Se carga únicamente el `.env` indicado, sin buscar archivos en
directorios superiores. Para otro archivo local, usa `--env-file ruta/al/archivo.env`.

`.gitignore` excluye `.env`, sus variantes, archivos `*.env`, claves y carpetas de
resultados. `.env.example` contiene solo nombres de variables y valores vacíos.

## Calcular el índice

```bash
python indice.py calcular --inicio 2012-01-01 --fin 2026-10-28 --salida outputs/indice
```

Omite `--fin` para usar la fecha local del equipo. La fecha inicial predeterminada es
2012-01-01. Mantenerla constante facilita comparar ejecuciones. El periodo inicial
contiene observaciones de calentamiento sin índice.

La instalación también habilita el comando equivalente `imc-chile calcular`.
`--sin-graficos` omite solo los gráficos. `--ventana` y `--min-historia` permiten
examinar otros tamaños de normalización; los parámetros efectivos quedan en metadata.

| Archivo | Contenido |
|---|---|
| `indice.csv` | Variables, puntajes, contribuciones, fechas, antigüedad, cobertura e índice |
| `metadata.json` | Versión, parámetros, fuentes, intervalo solicitado, fecha de ejecución y hashes |
| `series/*.csv` | Observaciones descargadas; estados originales y valores numéricos válidos |
| `indice.svg`, `indice.png` | Historia y componentes del último índice completo |

Las fechas de las series pueden diferir de la fecha del índice. Revisa
`latest_source_dates` y `age_*_days` antes de interpretar un resultado.

## Explorar todo el catálogo

No requiere token ni conexión:

```bash
python indice.py catalogo "cobre" --frecuencia diaria
python indice.py catalogo "imacec" --frecuencia mensual
python indice.py catalogo "" --capitulo "Mercado de Capitales" --limite 50
python indice.py catalogo "" --capitulo "Derivados y Spot" --salida outputs/derivados.json
```

Filtros: `--frecuencia`, `--capitulo`, `--fuente`, `--estado` y `--limite`. El estado
predeterminado es `Vigente`. La búsqueda no distingue mayúsculas ni acentos. Devuelve
todos los contextos de cada código encontrado, pues una misma serie aparece en varios
cuadros. **El catálogo es un snapshot**: un resultado no garantiza disponibilidad ni
actualización del endpoint. Revisa la [guía del catálogo](catalogo.md).

## Descargar una serie

```bash
python indice.py serie F032.IMC.IND.Z.Z.EP18.Z.Z.0.M --inicio 2020-01-01 --salida outputs/imacec.csv
```

Usa primero `catalogo` para elegir el código exacto, su unidad y frecuencia. La descarga
conserva fechas y `status_code`; los valores sin estado `OK` se marcan ausentes. No se
modifica la periodicidad. Descargar una serie no la incorpora al índice.

## Desde Python

```python
from imc.data import configured_client, fetch_series
from imc.engine import calculate_index

with configured_client() as client:
    series, sources = fetch_series(client, start="2012-01-01", end="2026-10-28")

result = calculate_index(series)
print(result.loc[result["complete"], ["index", "oldest_input_date"]].tail())
```

`calculate_index` no consulta la red: recibe cuatro DataFrames con fechas como índice,
columnas `value` y, opcionalmente, `status_code`. Su contrato y validaciones se describen
en la [referencia técnica](referencia.md).

## Errores y diagnósticos

| Situación | Comportamiento |
|---|---|
| Falta `BCCH_API_KEY` | Falla antes de consultar la API |
| Token vencido, red o error de BDE | Devuelve un error sin URL autenticada ni token |
| Sin historia suficiente | No genera un puntaje artificial |
| Fuente desactualizada | Falla si el último resultado utilizable excede la tolerancia |
| Último día incompleto pero hay un resultado reciente | Informa la fecha realmente calculada |
| Serie con fechas duplicadas o precios no positivos | Rechaza los datos |

La API REST transmite el token en parámetros HTTPS, conforme a su contrato. El SDK
oculta credenciales en sus errores y el proyecto filtra los logs del transporte HTTP.
No habilites captura externa de tráfico ni publiques trazas con variables de entorno.

## Desarrollo

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build
```

Las pruebas normales usan datos sintéticos y dobles de API; no necesitan token. Para
repetir la sensibilidad con una descarga real ya guardada:

```bash
python scripts/sensitivity.py outputs/indice/series --salida outputs/sensibilidad.json
```

Las pruebas de software y sensibilidad no constituyen evidencia de rentabilidad.
