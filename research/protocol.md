# Protocolo de segunda revisión — 29 de octubre de 2026

Definido antes de consultar los nuevos candidatos y estimar sus dependencias.

## Ruta de razonamiento

- outcome: one
- id: first-principles
- role: reconstruir el constructo sin asumir cuatro fuentes ni frecuencia diaria.
- unique_question: qué información económica necesita el usuario y qué observaciones la representan.
- domain: innovation / risk
- problem_type: create / evaluate
- constraints: solo BCCh, interpretabilidad, fechas de publicación no reconstruidas, secretos locales.
- rationale: la restricción de frecuencia y el número de bloques eran convenciones sin validación.
- blind_spots: la coherencia conceptual no prueba utilidad; se requieren mediciones separadas.
- exit: descartar transformaciones sin significado o que solo añadan complejidad; no forzar un escalar.

## Hipótesis y contrastes

H1: el catálogo contiene información de expectativas, participación bursátil, valoración,
oferta de crédito y fondeo que el índice v2 no representa. Falsador: ausencia de datos o
reconstrucción estable de estas medidas a partir de los cinco factores v2.

H2: indicadores seleccionados por mecanismo pueden aportar información sin exigir
ortogonalidad exacta. Falsador: correlaciones elevadas persistentes y falta de información
adicional en periodos separados. Correlación baja tampoco basta para declarar utilidad.

H3: separar confianza/apetito y fricción financiera permite mostrar desacoples que una
media oculta. Falsador: ambos perfiles resultan equivalentes o sus diferencias no admiten
una explicación concreta desde componentes observados. Este contraste es descriptivo.

Se revisa el catálogo completo por capítulos, conservando todos los contextos de cada
código, y se eligen candidatos antes de mirar correlaciones. No existe cuota deseada de
variables: los candidatos deben tener mecanismo y transformación identificados.

## Diseño empírico

- Descargar 2006-01-01 a 2026-10-28, conservando valores, estados y fechas.
- Prototipo mensual: diarios agregados por mes; mensual sin interpolación; trimestral en
  panel aparte, sin convertir tres meses repetidos en observaciones independientes.
- Precios: cambios logarítmicos de tres meses; valoraciones y encuestas: niveles;
  diferenciales: diferencias entre tasas comparables; cantidades: tasas de cambio o ratios.
- Normalización de cada factor: rango percentil frente a los 60 meses anteriores,
  mínimo 36; sin usar el mes evaluado para ajustar la referencia.
- Desarrollo hasta 2018-12; contraste temporal desde 2019-01, con informe adicional
  2023–2026. No optimizar pesos ni selección con los resultados del contraste.
- Spearman por pares con tamaños muestrales explícitos, dimensión efectiva de la matriz
  completa, y diagnóstico de candidatos redundantes. No confundirlo con independencia.
- Explicar cada candidato desde los cinco factores mensuales de v2 mediante regresión
  ridge fija (alpha=10), ajustada solo hasta 2018. Reportar R² temporal, no una predicción
  económica. R² bajo puede ser información adicional, ruido o inestabilidad.
- Diagnóstico de error de reconstrucción frente a predicción constante aprendida en
  desarrollo. No es prueba de rentabilidad ni de causalidad.

Los años posteriores a 2018 no son un holdout prístino: el investigador conoce su historia
y la v2 ya fue examinada hasta 2026. Se describen como contraste temporal retrospectivo.
Las series revisadas y sus fechas de observación no permiten simular disponibilidad real.
La validación prospectiva y un registro de publicaciones quedan como requisitos del paper.

## Adenda exploratoria de esta ejecución

La primera consulta cubrió 45 candidatos. Después se incorporaron siete insumos
justificados por definición: IPC para deflactar crédito, bono real a dos años y las
cinco categorías de respuestas ECB pyme. Se verificaron las fórmulas contra el saldo
oficial. No se escogieron por maximizar una correlación o un resultado financiero.
La comparación geométrica de doce factores y la descomposición ECB son exploratorias;
no se presentan como pruebas confirmatorias registradas antes de toda inspección.
El baseline mensual toma los cinco scores de la última fila diaria completa del mes.

## Límites vinculantes y supuestos descartados

Vinculantes: único proveedor BCCh, trazabilidad, no inventar series ni opiniones, no usar
datos futuros en transformaciones. Convenciones descartadas: todo debe ser diario, más
variables implica mejor medición, PCA implica novedad, fuentes distintas son ortogonales.
Prueba de descarte: si los nuevos bloques no aportan un diagnóstico interpretable y
estable, conservarlos como contexto y no agregarlos al indicador.
