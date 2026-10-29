# Cierre metodológico — MyM Ingenierías

**Veredicto B: NO RESPALDADO para un índice agregado de miedo/codicia ni para un titular
de diagnóstico generalmente estable.** La campaña se cierra con resultados negativos
y positivos; no constituye una validación del índice.

Los datos permiten describir precios, confianza, participación y condiciones de crédito,
con algunos vínculos económicos con objetivos excluidos de los componentes. Esto no mide
preferencias de riesgo o emociones. Se conserva una referencia descriptiva con componentes,
rango entre horizontes y abstención. No se declara una ganadora validada ni se recomienda invertir.

## Contrato, datos y alcance

Referencia `7dfcedb`; hash del contrato v2:
`d943db3890f310b6d92cdd2737a8f5eb637a9781d4be91ea9b22a8799a0f17b0`.
Los investigadores conocían toda la exploración previa: no hay una muestra de reserva intacta,
un equipo independiente ni validación externa por separar archivos. Desarrollo hasta 2020;
evaluación desde 2021 hasta septiembre de 2026. Las observaciones de 2006–2026 proceden
exclusivamente de la API del BCCh. El snapshot previo del 29 de octubre de 2026 contiene
52 series y 91.144 filas con hashes; la nueva descarga tampoco pudo conectarse al endpoint
público. No se ampliaron permisos. La reproducción sin conexión no constituye una segunda
consulta exitosa ni una comparación entre versiones históricas de publicación.

Se ejecutaron cinco arquitecturas, nueve ablaciones económicas y
**152 escenarios de sensibilidad**, dentro del máximo de 200.
Se usaron 400 remuestreos por contraste, bloques de 4 trimestres e IC 95 % exploratorios.
No se formulan conclusiones confirmatorias después de comparar múltiples métodos y objetivos.
Las cifras no prueban causalidad ni capacidad de predicción financiera. Los criterios y
correcciones previas constan en [el contrato](cierre-contrato.md) y
[el registro de la campaña](../campaign/ledger.json).

## Qué pierde el agregado

Exactitud porcentual en 68 meses comunes de contraste y 69 de ajuste. El mínimo era 90 %.
Los paneles 3 y 6 leen sus ejes directamente: el 100 % representa conservación de información,
no validación económica.

| Representación | Mercado | Confianza | Presiones | Dirección de cambios | Familia que más cambia |
| --- | --- | --- | --- | --- | --- |
| Índice v2 | 69.1 | 67.6 | 58.8 | 44.4 | 36.1 |
| Equiponderado | 70.6 | 73.5 | 67.6 | 56.6 | 23.0 |
| DT905 adaptado | 72.1 | 67.6 | 58.8 | 43.9 | 41.0 |
| Panel 3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Panel 6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

El agregado no sustituye el desglose. Se encontraron
18 pares con una diferencia de puntaje ≤ 2
y al menos un perfil con una diferencia ≥ 40 puntos. Ejemplos:

| Mes A | Mes B | Agregado A | Agregado B | Mayor diferencia de perfil |
| --- | --- | --- | --- | --- |
| 2021-07-31 | 2022-08-31 | 33.89 | 33.96 | 63.33 |
| 2023-07-31 | 2024-08-31 | 56.94 | 56.60 | 60.83 |
| 2023-07-31 | 2024-09-30 | 56.94 | 56.53 | 56.25 |

No hubo casos de oposición extrema —mercado < 25 y confianza > 75, o viceversa— en la
muestra común. El 100 % de exactitud de esa tarea es trivial; la sensibilidad de detección
(recall) no es evaluable. No se cambió el umbral para fabricar positivos. Un agregado
acompañado de la misma descomposición conserva la misma
información que el panel: la información está en el desglose, no es nueva por añadir una brecha.
La distribución ECB distingue además saldos cero con y sin cambios opuestos; ese control
algebraico no constituye validación independiente de emociones.

## Aporte económico y complejidad

Reducción relativa del error absoluto medio, MAE (%), frente a un modelo con Imacec, IPC y TPM,
en la misma muestra trimestral. Un valor positivo indica menor error. Los seis objetivos ECB
no forman parte de los perfiles.
No se repite un trimestre como tres observaciones. Oferta positiva es flexibilización;
demanda positiva es fortalecimiento. Crédito de consumo no equivale a todos los créditos
de los hogares.

| ECB | N ajuste | N contraste | Índice v2 | Equiponderado | DT905 adaptado | Panel 3 | Panel 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oferta a pymes | 21 | 22 | 8.9 | 6.2 | -0.2 | -7.2 | 2.1 |
| Oferta a empresas grandes | 21 | 22 | -9.3 | 35.6 | 17.6 | 4.7 | 13.0 |
| Oferta de crédito de consumo | 21 | 22 | -15.7 | 5.5 | 6.2 | 27.4 | 26.0 |
| Demanda de pymes | 21 | 22 | 0.7 | -3.7 | 5.0 | 15.1 | 7.8 |
| Demanda de empresas grandes | 21 | 22 | 9.4 | 1.5 | -9.4 | 24.2 | 29.6 |
| Demanda de crédito de consumo | 21 | 22 | -11.9 | 8.1 | 5.8 | 26.2 | 33.9 |

Los intervalos completos están en
[las métricas económicas](../campaign/results/economics_metrics.json)
y en la figura de comparación. Panel 3 alcanza al menos 10 % de reducción del MAE, con
intervalo exploratorio positivo, en oferta de consumo y demanda de pymes y empresas grandes.
Panel 6 cumple en oferta de consumo y demanda de empresas grandes y de consumo.
El equiponderado mejora la oferta a empresas grandes en 35,6 %. No hay dominancia uniforme.
Panel 3 excede el margen puntual de 5 % frente a Panel 6 en 4 de 6 objetivos.
Estos resultados no refutan otras relaciones económicas posibles.

El aporte también depende del episodio. Para oferta a pymes, Panel 3 aumenta el MAE
frente a los controles en 73,9 % durante 2021–2022 (n = 8 trimestres), pero lo reduce en
20,3 % durante 2023–2026 (n = 14). En demanda de crédito de consumo, su reducción del MAE
pasa de 40,2 % a 5,5 % entre esos mismos periodos. Son resultados descriptivos de
[economic_episodes.csv](../campaign/results/economic_episodes.csv): el promedio de todo
el contraste no demuestra un beneficio estable entre episodios.

La siguiente tabla muestra el aumento relativo del MAE (%) al retirar cada familia de Panel 6.
Un valor negativo significa que retirarla mejora ese objetivo. Todas las ablaciones mantienen
las mismas fechas:

| Familia retirada | Oferta a pymes | Oferta a empresas grandes | Oferta de crédito de consumo | Demanda de pymes | Demanda de empresas grandes | Demanda de crédito de consumo |
| --- | --- | --- | --- | --- | --- | --- |
| Mercado | -4.9 | -16.6 | -0.0 | 4.5 | -1.3 | 3.2 |
| Confianza | -9.1 | 5.5 | 52.8 | -16.1 | 12.4 | 56.0 |
| Presiones e incertidumbre | 22.3 | 55.9 | 1.1 | 21.6 | -4.1 | 3.1 |
| Participación | 11.0 | 11.0 | -8.8 | -3.6 | 11.4 | -7.2 |
| Crédito real | 1.9 | -2.3 | -0.7 | -0.4 | -5.3 | 28.2 |
| Composición monetaria | -4.1 | -0.2 | 4.1 | -8.0 | 0.0 | -1.5 |

Confianza y presiones tienen aportes específicos; participación aporta en los modelos de oferta
a pymes y empresas grandes, y crédito real en el de demanda de consumo. Composición monetaria
parecía prescindible al medir el margen puntual de 5 % en todo el contraste. Esa elección usaría
el mismo periodo para seleccionar y comprobar la simplificación. Antes del cierre se registró
una corrección del evaluador: aplicar el mismo margen a las tres exclusiones opcionales,
seleccionando con 2021–2022 y comprobando con 2023–2026, sin reajustar modelos ni umbrales.

**Ninguna exclusión superó la etapa de selección.** Quitar M1/M2 aumentó el MAE de demanda
pyme 7,23 % en 2021–2022, y el de oferta de consumo 10,19 % en 2023–2026. La propuesta
Panel 5 se descarta. Se conserva **Panel 6 como referencia descriptiva no validada**, con
las tres familias principales y tres contextos. No se fuerza una metodología ganadora.
El control está en selection_audit.json y su protocolo en selection-check.json. Se definió
después de conocer resultados agregados: separa periodos dentro del control retrospectivo,
pero no crea una reserva intacta ni demuestra no inferioridad estadística.

## Estabilidad: dónde cambia la interpretación

Los 24 cambios moderados de pesos cumplen las tolerancias en
288/288
comparaciones de coordenadas.
Eso no significa que todos los escenarios aprueben. Máximos entre escenarios de cada tipo:

| Cambio | Método | Peor p95 de Δ absoluto | Máximo Δ absoluto | Cambio de categoría (%) | Extremos opuestos (%) | N mínimo | N máximo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| component_ablation | DT905 adaptado | 42.6 | 62.4 | 38.2 | 1.5 | 68 | 68 |
| component_ablation | Equiponderado | 11.9 | 14.4 | 22.1 | 0.0 | 68 | 68 |
| component_ablation | Panel 3 | 35.8 | 43.3 | 50.7 | 0.0 | 68 | 69 |
| component_ablation | Panel 6 | 35.8 | 43.3 | 50.7 | 0.0 | 68 | 69 |
| component_ablation | Índice v2 | 14.7 | 17.6 | 14.5 | 0.0 | 69 | 69 |
| family_ablation | Equiponderado | 13.8 | 20.4 | 14.7 | 0.0 | 68 | 68 |
| historical_outlier | DT905 adaptado | 17.9 | 21.5 | 25.0 | 0.0 | 68 | 68 |
| historical_outlier | Equiponderado | 0.3 | 0.3 | 0.0 | 0.0 | 68 | 68 |
| historical_outlier | Panel 3 | 0.8 | 0.8 | 2.9 | 0.0 | 68 | 69 |
| historical_outlier | Panel 6 | 1.7 | 1.7 | 2.9 | 0.0 | 68 | 69 |
| historical_outlier | Índice v2 | 1.0 | 18.4 | 0.0 | 0.0 | 69 | 69 |
| hypothetical_revision | DT905 adaptado | 2.2 | 4.3 | 2.9 | 0.0 | 68 | 68 |
| hypothetical_revision | Equiponderado | 1.6 | 2.4 | 1.5 | 0.0 | 68 | 68 |
| hypothetical_revision | Panel 3 | 4.7 | 7.1 | 4.4 | 0.0 | 68 | 69 |
| hypothetical_revision | Panel 6 | 5.0 | 8.3 | 7.2 | 0.0 | 68 | 69 |
| hypothetical_revision | Índice v2 | 1.0 | 11.9 | 1.4 | 0.0 | 69 | 69 |
| minor_weights | DT905 adaptado | 9.3 | 12.9 | 2.9 | 0.0 | 68 | 68 |
| minor_weights | Equiponderado | 3.8 | 5.3 | 7.4 | 0.0 | 68 | 68 |
| minor_weights | Panel 3 | 6.2 | 7.4 | 4.4 | 0.0 | 68 | 69 |
| minor_weights | Panel 6 | 6.2 | 7.4 | 4.4 | 0.0 | 68 | 69 |
| minor_weights | Índice v2 | 3.9 | 5.0 | 4.3 | 0.0 | 69 | 69 |
| publication_delay | DT905 adaptado | 51.1 | 73.1 | 48.5 | 4.4 | 67 | 68 |
| publication_delay | Equiponderado | 19.7 | 27.8 | 14.7 | 0.0 | 67 | 68 |
| publication_delay | Panel 3 | 40.3 | 55.8 | 39.1 | 0.0 | 67 | 69 |
| publication_delay | Panel 6 | 81.7 | 90.0 | 78.3 | 23.2 | 67 | 69 |
| publication_delay | Índice v2 | 39.6 | 69.6 | 50.7 | 2.9 | 69 | 69 |
| reference_horizon | Equiponderado | 19.3 | 19.9 | 27.6 | 0.0 | 29 | 67 |
| reference_horizon | Panel 3 | 27.6 | 27.7 | 55.2 | 0.0 | 29 | 69 |
| reference_horizon | Panel 6 | 42.5 | 44.2 | 55.2 | 0.0 | 29 | 69 |
| reference_horizon | Índice v2 | 18.8 | 20.0 | 17.4 | 0.0 | 46 | 69 |

Las ventanas cambian el referente económico; sus diferencias no se disfrazan de fallos
puramente numéricos. Los extremos históricos aislados tienen poco efecto en rangos,
pero faltantes llevan a abstención y retrasos pueden invertir lecturas. Los shocks de revisión
son hipotéticos y distintos en unidades entre representaciones; no son revisiones observadas.
Las ventanas de v2 equivalen aproximadamente a 21 observaciones diarias por mes, a diferencia
de los meses calendario del panel. PCA usa la referencia fija de desarrollo de la adaptación;
forzarle ventanas móviles sería cambiar ese comparador.

Una etiqueta estable exige que las referencias de 36, 60 y 120 meses coincidan en categoría
y que su rango no exceda 20 puntos. La cobertura incluye los 69 meses del contraste,
incluso cuando falta historia para la ventana de 120 meses:

| Perfil | Etiquetas estables | Todos los meses | Historia común completa | Cobertura (%) | p95 de amplitud |
| --- | --- | --- | --- | --- | --- |
| Mercado | 53 | 69 | 69 | 76.8 | 23.8 |
| Confianza | 28 | 69 | 69 | 40.6 | 34.5 |
| Presiones e incertidumbre | 6 | 69 | 29 | 8.7 | 30.1 |

Confianza y presiones no alcanzan el 50 % requerido. Se muestran valores ligados a su horizonte,
rangos y ausencia de etiqueta; no un titular universalmente estable. El rango es de
escenarios, no un intervalo de confianza probabilístico. Las tablas de episodios y máximos
por fecha están en [sensitivity_episodes.csv](../campaign/results/sensitivity_episodes.csv),
[source_stress.csv](../campaign/results/source_stress.csv) y
[economic_episodes.csv](../campaign/results/economic_episodes.csv).
El episodio 2019–2020 pertenece al desarrollo y no se reutiliza como evaluación posterior.

## Dinámica, rivales y novedad

SINDy no se ajustó: 70 transiciones válidas de desarrollo para 10 términos por ecuación y
un N efectivo heurístico de 1,55–6,55, lejos de las 100 observaciones exigidas. Ese N efectivo
no es un conteo exacto de observaciones
independientes para un modelo multivariado; el incumplimiento basta como control conservador.
El VAR(1) con ridge reduce el MSE frente a persistencia en 9,0 % para mercado, pero lo aumenta
en 23,6 % para confianza y en 0,7 % para presiones. No hay apoyo para presentar una dinámica
conjunta descubierta.

Sobrevive la explicación de que separar mecanismos conserva información que comprimir
elimina. No se descartan completamente ciclo compartido, normalización, cambios de régimen
ni ruido: los resultados ECB son heterogéneos y la ventana importa. La baja correlación o
mayor dimensión efectiva no se usó como criterio de utilidad. DT905 ya descompone contribuciones;
CISS ya combina segmentos y rangos; DT1080 distingue shocks estructurales. La contribución
comprobada es el experimento reproducible y la trazabilidad del diagnóstico BCCh; no una
nueva teoría de miedo/codicia ni prioridad científica demostrada.

## Qué quedó demostrado y qué no

- Software: rangos móviles calculados solo con pasado y comparadores ajustados exclusivamente
  en desarrollo, hashes, ausencia de relleno,
  pruebas sintéticas y reproducción registrada. PCA usa todo el desarrollo para calcular
  sus valores dentro de ese periodo; estos no representan un indicador disponible entonces.
  No usa evaluación para ajustar parámetros. Esto no valida el significado económico.
- Economía: corroboración descriptiva parcial en seis objetivos de crédito retenidos;
  ninguna medición identificada de emociones ni superioridad universal.
- Tiempo: retrospectivo con datos revisados. Fechas de publicación por observación y
  versiones históricas no se reconstruyeron. Hay escenarios de rezago y un registro de primeras
  capturas/revisiones futuras; una primera captura histórica no es validación prospectiva.
- Usuarios: sin participantes, no se probaron exactitud ni tiempo de comprensión humanos.
- Ciencia: no se probó novedad; se documentaron antecedentes y resultados negativos.

La campaña está cerrada. Permanecen identificadas las brechas que impiden ampliar el
alcance: preferencias de riesgo independientes, historia de publicaciones/versiones y
pruebas con personas. No se presentan como tareas rutinarias pendientes para sostener
un éxito inexistente. La [metodología retenida](../campaign/results/methodology.json) documenta
la referencia descriptiva; la [matriz completa](cierre-matriz.md) separa cada afirmación
de la evidencia que la respalda o la limita.

## Documentación de la campaña

1. [Contrato y criterios registrados](cierre-contrato.md).
2. [Definiciones, fórmulas y límites de interpretación](cierre-definiciones.md).
3. [Disponibilidad, fechas de publicación y versiones](cierre-disponibilidad.md).
4. [Literatura, antecedentes y explicaciones rivales](cierre-literatura.md).
5. [Veredicto completo de la campaña](cierre-veredicto.md).
6. [Matriz de afirmaciones y evidencia](cierre-matriz.md).
