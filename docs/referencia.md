# Referencia técnica

## Organización del código

| Archivo | Responsabilidad |
|---|---|
| `indice.py` | CLI: `calcular`, `catalogo`, `serie` |
| `imc/config.py` | Códigos BDE, unidades, tolerancias, pesos y parámetros |
| `imc/data.py` | Configuración, acceso BCCh, protección del token y validación |
| `imc/engine.py` | Transformaciones, percentiles, alineación y agregación |
| `imc/report.py` | Exportación, metadata, hashes y gráficos |
| `scripts/sensitivity.py` | Comparación descriptiva de ventanas y bloques |
| `tests/` | Casos temporales, datos ausentes, integración simulada y seguridad |

Importar estos módulos no consulta la API ni genera archivos. El motor funciona
independientemente del transporte: recibe DataFrames y devuelve otro DataFrame.

## Contrato del motor

```python
from imc.config import Methodology
from imc.engine import calculate_index, latest_complete

config = Methodology(horizon=20, normalization_window=504, min_history=252, max_source_gap_days=31)
result = calculate_index(frames, config)
latest, pending_sources = latest_complete(result, as_of="2026-10-28")
```

`frames` debe contener exactamente las claves `dolar`, `cobre`, `tasa_10y` e
`incertidumbre`. Cada DataFrame requiere un `DatetimeIndex` sin zona horaria, sin
duplicados ni fechas nulas, y la columna numérica `value`. Si está presente
`status_code`, solo se aceptan valores `OK`. Los valores infinitos se tratan como
ausentes. Los precios deben ser positivos; las tasas pueden ser negativas.

El resultado conserva las filas de calentamiento y los días incompletos. Las columnas
principales son:

| Columnas | Significado |
|---|---|
| `value_<fuente>` | Última observación alineada dentro de la tolerancia |
| `date_<fuente>`, `age_<fuente>_days` | Fecha de origen y antigüedad en días calendario |
| `<factor>` | Transformación antes de normalizar |
| `score_<factor>` | Puntaje percentil orientado, entre 0 y 100 |
| `contribution_<factor>` | Puntaje multiplicado por su peso |
| `available_components` | Número de factores con puntaje válido, de 0 a 5 |
| `index`, `complete` | Suma ponderada y disponibilidad de los cinco factores |
| `oldest_input_date` | Fecha más antigua entre las fuentes alineadas |

`latest_complete` restringe la consulta a `as_of`, exige cobertura reciente y devuelve
las fuentes ausentes en la última fila del calendario. Un resultado incompleto tiene
`index=NaN`; no debe reemplazarse por cero en aplicaciones posteriores.

## Integración BCCh

El cliente es `BancoCentralDeChile` del repositorio `bcch-sdk`. El endpoint utilizado
es `https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx`, con autenticación mediante
token y funciones `GetSeries`/`SearchSeries`, según la
[documentación oficial](https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/acceso_api.html).

La consulta del indicador realiza una descarga por cada una de sus cuatro series.
El SDK aplica límite de solicitudes, timeout y reintentos acotados. Los errores API
y de transporte se convierten en mensajes breves sin copiar URLs autenticadas.
Los datos económicos no se solicitan a otros servicios.

## Credenciales y archivos

- `BCCH_API_KEY` se toma del proceso o del `.env` indicado; el proceso tiene prioridad.
- `.env.example` solo declara la variable vacía. No se incluye un token de demostración.
- `.gitignore` excluye `.env`, variantes locales, claves y archivos de credenciales.
- La sesión HTTP rechaza redirecciones. El filtro de logs oculta el token literal y
  sus representaciones escapadas en los mensajes del transporte utilizado.
- Los exports usan campos explícitos: no serializan respuestas crudas, solicitudes,
  objetos de sesión, variables de entorno ni URLs con parámetros de autenticación.
- Los CSV y gráficos de ejecuciones quedan en `outputs/`, fuera del control de versiones.

Antes de un commit, verifica `git check-ignore .env .env.local credentials.json` y
revisa `git diff --cached`. La presencia de `.gitignore` no elimina archivos que ya
estuvieran versionados.

## Reproducibilidad y pruebas

`metadata.json` contiene la versión de metodología, sus parámetros, pesos, intervalo
solicitado, hora de ejecución UTC, cobertura por fuente y hashes SHA-256 de los CSV.
Conserva esos archivos juntos para reproducir una ejecución incluso si la BDE revisa
datos históricos. Los hashes prueban identidad de archivos, no calidad estadística.

El flujo de CI ejecuta Ruff, pytest y construcción del paquete en Python 3.11–3.14.
Las pruebas no utilizan credenciales ni red. La consulta real documentada es una
verificación separada, porque disponibilidad, rezagos y revisiones dependen del BCCh.

La metodología debe versionarse cuando cambien fuentes, transformaciones, pesos u
orientaciones. Cambiar una fuente en `SERIES` sin revisar su contrato económico y
temporal no es una extensión válida del indicador.
