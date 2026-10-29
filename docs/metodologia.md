# Metodología v2.0.0

Esta versión describe condiciones de mercado mediante cuatro fuentes diarias de la BDE.
Su nombre operativo es **Índice de condiciones de mercado de Chile**. El repositorio
conserva su nombre histórico, `Indice-de-miedo-y-codicia`.

## Alcance

El índice resume variaciones del peso y del cobre, variabilidad de tasas e incertidumbre
económica. Un puntaje mayor representa condiciones relativamente más distendidas dentro
de las ventanas observadas, según las orientaciones definidas aquí. No es una medición
directa de emociones, valor justo, riesgo sistémico, solvencia ni rendimiento futuro.

La v2 introduce una metodología nueva. **No se debe empalmar con la serie de 2022**.
En la consulta del 28 de octubre de 2026, los códigos diarios antiguos de IPSA y EMBI
Chile devolvieron el código API `-50` y no figuraron en `SearchSeries(DAILY)`. El IPSA
mensual sí respondió, pero no se interpola para fabricar observaciones diarias.

## Fuentes y unidades

| Alias | Código BDE | Unidad | Antigüedad máxima al alinear |
|---|---|---|---:|
| `dolar` | `F073.TCO.PRE.Z.D` | CLP/USD | 0 días |
| `cobre` | `F019.PPB.PRE.100.D` | USD/libra | 3 días calendario |
| `tasa_10y` | `F022.BCLP.TIS.AN10.NO.Z.D` | Porcentaje anual | 3 días calendario |
| `incertidumbre` | `F029.IDIE.IND.TOT.D` | Índice | 7 días calendario |

La BDE identifica como fuentes originales al BCCh, la Bolsa de Metales de Londres y
Becerra y Sagner (2020), según la serie. Todas las descargas se realizan al endpoint
del BCCh. No se consulta directamente a las fuentes originales.

## Transformaciones

Cada cálculo se realiza sobre observaciones válidas de su propia fuente, ordenadas por
fecha. No se imputan precios ni se incorporan ceros en feriados o fines de semana.
`h = 20` **observaciones válidas**. En precios y tasas suelen corresponder a días
hábiles observados; en IDIE pueden incluir fines de semana. No se afirma que todos los
horizontes cubran el mismo número de días calendario.

Sea `D` el dólar observado, `C` el cobre, `Y` la tasa a diez años y `U` el IDIE.

| Medida | Variable antes de normalizar | Orientación del puntaje | Peso |
|---|---|---|---:|
| Presión cambiaria | `ln(D[t] / D[t-h])` | Aumentos reducen el puntaje | 12,5% |
| Volatilidad cambiaria | Desviación estándar de 20 retornos `Δln(D)`, `ddof=0` | Aumentos reducen el puntaje | 12,5% |
| Variación del cobre | `ln(C[t] / C[t-h])` | Aumentos elevan el puntaje | 25% |
| Variabilidad de tasas | Desviación estándar de 20 cambios `ΔY`, `ddof=0` | Aumentos reducen el puntaje | 25% |
| Incertidumbre económica | Media de 20 valores de `U` | Aumentos reducen el puntaje | 25% |

La tasa se diferencia en **puntos porcentuales**, sin calcular retornos porcentuales
sobre rendimientos ni dividir por una tasa que pueda ser cero. No se anualizan las
desviaciones estándar: la normalización posterior es relativa a cada serie.

La orientación del peso y del cobre es una **convención descriptiva**. Una depreciación
puede beneficiar a exportadores; un alza del cobre puede reflejar restricciones de oferta.
La variabilidad de tasas puede aumentar tanto con subidas como con bajadas. Estas medidas
no identifican por sí solas causas económicas ni estados psicológicos.

## Normalización retrospectiva

Cada variable se transforma mediante su rango percentil dentro de las últimas
`504` observaciones de su fuente, incluida la actual. Se requieren `252` valores
válidos del factor. Estos tamaños no equivalen exactamente a dos años o a un año en
todas las fuentes.

Para el valor actual `x`, sobre los `n` valores válidos de la ventana:

```text
P(x) = 100 × (cantidad de valores menores que x + 0,5 × cantidad de empates con x) / n
```

Para el cobre se usa `P(x)`; para los otros cuatro factores, `100 − P(x)`.
Una ventana constante produce 50. Una observación ausente conserva `NaN`.
Este rango medio de empates evita clasificar una historia constante como un extremo.

Los percentiles hacen comparables unidades distintas y reducen la influencia de la
magnitud de valores extremos. También pierden información sobre distancias absolutas.
La elección de 504/252 es un parámetro de diseño expuesto en el programa.

**Agregar observaciones futuras no cambia los puntajes anteriores si los datos anteriores
permanecen idénticos.** Cambiar la fecha inicial de descarga, los parámetros o revisiones
de la fuente sí puede cambiar resultados. La prueba de invariancia temporal cubre la
primera propiedad; no convierte la base en una historia de publicaciones en tiempo real.

## Agregación y cobertura

Hay cuatro bloques de 25%: cambiario, cobre, tasas e incertidumbre. El bloque cambiario
se reparte entre dos medidas para que tener dos variables no duplique automáticamente
su importancia. Los pesos no se ajustaron para maximizar correlación o rentabilidad.

```text
ICM = 0,125 × presión cambiaria normalizada
    + 0,125 × volatilidad cambiaria normalizada
    + 0,250 × variación del cobre normalizada
    + 0,250 × variabilidad de tasas normalizada
    + 0,250 × incertidumbre normalizada
```

La suma exige los cinco componentes. Si falta uno, el resultado queda ausente. No se
reponderan automáticamente los restantes. La ponderación igual por bloques no elimina
correlaciones entre cobre, peso e incertidumbre.

## Fechas, huecos y rezagos

1. El calendario de salida se fija con las fechas válidas del dólar observado.
2. Los factores se calculan y normalizan antes de alinearse entre fuentes.
3. Se usa la observación del factor más reciente **en esa fecha o antes**, dentro de las
   tolerancias de la tabla. Nunca una observación futura.
4. El CSV conserva la fecha y antigüedad de cada fuente en cada fila.
5. Un intervalo de más de 31 días entre observaciones válidas de una fuente reinicia sus
   ventanas de cálculo y normalización. Intervalos menores permanecen explícitos en las
   fechas: 20 observaciones de bonos poco transados pueden abarcar más de un mes.
6. La consulta solo publica un resultado si existe un índice completo reciente y la
   fuente más antigua empleada no excede siete días respecto de la fecha solicitada.

Las tolerancias son límites operativos, no calendarios oficiales de publicación.
Cuando el último día está incompleto, se informa la fecha del último índice completo y
las fuentes pendientes; no se etiqueta un valor antiguo como si fuera de hoy.

## Qué significan 0, 50 y 100

Son referencias de una escala relativa. Un 50 no demuestra equilibrio económico ni
precio justo. Un 20 o un 80 no son probabilidades ni reglas para comprar o vender.
La lectura debe acompañarse de sus componentes, fechas y variables originales.

## Evidencia y límites

La investigación de incertidumbre económica respalda el uso de una medida textual como
contexto económico; no valida este promedio de cinco factores. La literatura sobre
indicadores compuestos informa la transparencia de ponderaciones y normalización,
sin otorgar validez empírica automática a esta implementación. Véase
[fundamentos y referencias](fundamentos.md).

La [verificación](verificacion.md) comprueba software, contrato API y sensibilidad
descriptiva. No contiene una validación predictiva ni una comparación de estrategias
financieras. Las series históricas pueden revisarse y el programa no almacena vintages
ni reconstruye la fecha en que cada dato estuvo públicamente disponible.
