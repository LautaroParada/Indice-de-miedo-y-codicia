# Verificación de la versión 2

## Consulta real al BCCh

Ejecución realizada el 28 de octubre de 2026, hora de Chile, con corte solicitado
`2026-10-28` e inicio `2012-01-01`. Se utilizó exclusivamente `GetSeries` del BCCh.
El token se tomó de configuración local excluida de Git.

| Serie | Observaciones válidas | Primera recibida | Última recibida |
|---|---:|---|---|
| Dólar observado | 3.688 | 2012-01-02 | 2026-10-28 |
| Cobre refinado BML | 3.498 | 2012-10-05 | 2026-10-28 |
| Bonos en pesos a diez años | 3.445 | 2012-01-02 | 2026-10-27 |
| Incertidumbre económica | 5.381 | 2012-01-30 | 2026-10-23 |

Se obtuvieron **3.187 filas completas de 3.688 fechas del calendario**; el resto incluye
calentamiento y falta de cobertura. El último resultado completo corresponde al
2026-10-28 y vale **19,30/100** con los parámetros predeterminados. La fuente más antigua
utilizada en ese resultado es del 2026-10-23. Se trata de un ejemplo fechado, no de una
lectura actual permanente ni de una señal de inversión.

La [metadata del ejemplo](assets/ejemplo-metadata.json) conserva los códigos, parámetros,
fechas y hashes de los CSV. La hora está expresada en UTC, por lo que el sello de
ejecución corresponde al día 29 en UTC y al día 28 en Chile. La
[figura del README](assets/indice.svg) procede de esa ejecución.

La verificación de las fuentes originales encontró `-50` en los códigos diarios
`F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D` y `F019.SPS.PBP.91.D`; tampoco aparecieron en
`SearchSeries(DAILY)`. El IPSA mensual `F013.IBC.IND.N.7.LAC.CL.CLP.BLO.M` sí respondió.
Esto describe la consulta efectuada y no prueba que una serie haya sido retirada
permanentemente. La nueva metodología no necesita esos códigos diarios.

## Sensibilidad de la normalización

Se recalculó la misma descarga con ventanas de 252, 504 y 756 observaciones, manteniendo
252 valores mínimos y todas las demás reglas. La comparación utiliza las mismas
3.187 filas completas. Las diferencias se expresan en puntos de la escala 0–100.

| Ventana | Diferencia absoluta media frente a 504 | Diferencia máxima | Correlación con 504 | Último valor |
|---:|---:|---:|---:|---:|
| 252 | 5,15 | 21,38 | 0,932 | 17,76 |
| 504 | 0,00 | 0,00 | 1,000 | 19,30 |
| 756 | 3,18 | 15,20 | 0,973 | 24,52 |

Hay dependencia material de la ventana. La correlación elevada no implica que las
lecturas puntuales sean intercambiables, ni valida una ventana como óptima.

## Sensibilidad a los bloques

Como diagnóstico, se retiró cada bloque de 25% y se dividió la suma restante por 0,75.
Solo se usaron las filas originalmente completas. **Esta reponderación es un experimento
separado**: el indicador normal no repondera cuando faltan datos.

| Bloque retirado | Diferencia absoluta media | Diferencia máxima | Último valor alternativo |
|---|---:|---:|---:|
| Cambiario | 4,70 | 17,86 | 12,93 |
| Cobre | 6,55 | 21,74 | 17,63 |
| Tasas | 6,33 | 21,03 | 20,93 |
| Incertidumbre | 7,34 | 21,98 | 25,69 |

Estas diferencias muestran que la elección de componentes importa. No miden aportes
causales, redundancia exacta ni capacidad predictiva. Los resultados sin redondear
están en [sensibilidad.json](assets/sensibilidad.json).

## Verificación de software

En Python 3.12.14, las **31 pruebas del indicador** y las **78 pruebas del SDK** pasan.
Cubren invariancia al agregar datos futuros, ausencia de relleno ilimitado, reinicio
tras huecos largos, empates, datos incompletos, antigüedad, errores API y protección del
token. Las pruebas del catálogo verifican sus recuentos y el saneamiento de enlaces.
Ruff y la construcción de los paquetes forman parte de las comprobaciones de entrega.
La matriz de CI configura Python 3.11, 3.12, 3.13 y 3.14.

Para repetir la consulta y el análisis:

```bash
python indice.py calcular --inicio 2012-01-01 --fin 2026-10-28 --salida outputs/verificacion
python scripts/sensitivity.py outputs/verificacion/series --salida outputs/sensibilidad.json
python -m pytest
python -m ruff check .
python -m build
```

Una nueva descarga puede diferir si el BCCh revisa la historia. No se reconstruyeron
fechas de publicación ni versiones históricas de los datos. Estas comprobaciones
acreditan el comportamiento descrito del software y una consulta real; no acreditan
rentabilidad, medición directa de emociones ni validez predictiva.
