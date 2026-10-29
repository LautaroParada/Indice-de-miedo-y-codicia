# Contrato de evaluación de cierre

Estado inicial: fijado antes de ejecutar los nuevos experimentos. El archivo canónico
es [contract.json](../campaign/contract.json); su hash se registra en cada ejecución.
Referencia inalterada: commit `7dfcedbad4a08fd0b4d51f4ea57df1997d69fed5`.
Versión efectiva: v2, hash `d943db3890f310b6d92cdd2737a8f5eb637a9781d4be91ea9b22a8799a0f17b0`.
La v1 permanece en [contract-v1.json](../campaign/contract-v1.json).
Todos los participantes conocen la exploración previa hasta 2026. Separar agentes o
archivos no produce evaluación independiente; el contraste será retrospectivo.

## Alcance finito y recursos

Cinco arquitecturas, 52 fuentes ya identificadas, una descarga y un reintento solo si
falla, hasta 200 escenarios, 400 remuestreos por contraste, hasta 20 consultas
bibliográficas. Sin servicios nuevos, compras ni permisos adicionales. Las reparaciones
del evaluador se registran y repiten todas las alternativas afectadas. No se reinician
estos límites ni se exploran pesos hasta encontrar un ganador.

## Pregunta y constructo

Apetito por riesgo es disposición a aceptar exposición incierta, manteniendo constantes
expectativas sobre pagos, restricciones y oportunidades. Miedo/codicia añade una
interpretación psicológica. Precios, volumen, confianza y condiciones crediticias no
identifican por sí solos esa disposición: cambios de fundamentos pueden moverlos igual.
El test de este constructo exige evidencia discriminante externa; ECB es una encuesta
de condiciones de crédito, no una medición independiente de emociones de inversionistas.

Se evalúan además dos alcances distintos: información descriptiva conservada en una
representación y asociación con condiciones crediticias declaradas fuera de sus inputs.
Ninguno sustituye pruebas de comprensión con personas.

## Comparaciones y partición

V2 diaria original, agregado equiponderado de tres familias, panel de seis perfiles,
adaptación PCA de DT905 y panel mínimo de tres perfiles. El PCA respeta ajuste y signos
en desarrollo; tiene sustituciones explícitas y no reproduce las once series del paper.
Periodo de desarrollo hasta diciembre de 2020 y contraste desde enero de 2021. Esta
partición permite más trimestres comunes con v2 que el corte exploratorio 2018; se fija
por cobertura, sin examinar resultados de la campaña. Mínimos: 20 trimestres de ajuste,
16 de contraste; se informa incertidumbre amplia cuando la muestra es pequeña.

ECB oferta/demanda de pymes, grandes empresas y consumo queda excluida de todos los
inputs mensuales; se evalúa trimestralmente, sin repetir el mismo dato tres meses.
Imacec, IPC y TPM son controles de ciclo, inflación y política. La comparación económica
es contemporánea y retrospectiva, no un pronóstico ni validación causal.

## Tolerancias justificadas por lectura

- Escala 0–100 con tres intervalos descriptivos: bajo <25, central 25–75, alto >75.
  No se llaman miedo/neutro/codicia. Un cambio de hasta cinco puntos se considera pequeño
  para una lectura pública; se muestra como número, sin exagerar un giro de régimen.
- Para cambios menores de pesos: mediana <=5, percentil95 <=10, cambios de categoría
  <=10% e inversión entre extremos <=1%. Las dos primeras tolerancias no permiten
  esconder inversiones bajo una alta correlación global.
- Ventanas 36/60/120 responden a referencias económicas distintas. Se exige cobertura
  completa para compararlas. La etiqueta firme requiere acuerdo de las tres ventanas
  y rango <=20; de otro modo se muestra sensibilidad o ausencia de historia. El rango
  es de escenarios, no un intervalo probabilístico. Se exige >=50% de cobertura para
  respaldar que esa etiqueta resulte habitualmente disponible.
- Una ampliación debe mejorar al menos 10% el MAE sobre controles en dos objetivos ECB,
  uno de oferta y uno de demanda, con intervalo pareado que excluya cero. Cinco puntos
  porcentuales de exactitud o 5% de MAE es el margen máximo para preferir una opción
  más simple. Los intervalos económicos son exploratorios al 95%; no permiten declarar
  superioridad confirmatoria tras seleccionar entre cinco arquitecturas y seis targets.
- Las tareas automatizadas requieren >=90% de respuestas correctas. Se identifican como
  pruebas de conservación de información: leer un componente mostrado no valida su
  interpretación económica. Un escalar con toda la descomposición puede igualar al panel;
  la información proviene entonces de la descomposición.

## Resultados que limitan o rechazan

Sin evidencia discriminante no se respalda miedo/codicia aunque el código funcione.
Si el agregado oculta >10% de categorías o desacoples, no sirve como sustituto del panel.
Si un perfil depende de pesos, horizonte o rezago se restringe su etiqueta y se informa
el rango. Si no hay vintages se descarta toda validación histórica en tiempo real.
Sin participantes no se declara mejora de comprensión o tiempo de usuarios.
La separación de perfiles ya tiene antecedentes; no se reclama como invención.

## Matriz de afirmaciones inicial

| Afirmación | Evidencia previa | Brecha | Experimento y métrica | Criterio | Resultado/estado inicial |
|---|---|---|---|---|---|
| Mide apetito/aversión | Proxies de mercado y encuestas | Identificación frente a fundamentos | A: mecanismo, contraejemplos, ECB con controles | Evidencia discriminante independiente | Sin evaluar |
| Un agregado basta | Rango escalar anterior | Cancelación y pérdida de composición | B: tareas de categoría, cambio, desacople y ablación | >=90% acierto por tarea | Sin evaluar |
| Nuevas familias aportan | Correlaciones/riqueza estadística | Utilidad adicional no tautológica | B: ridge ECB retenida y ablation | >=10% ganancia y CI positiva | Sin evaluar |
| Lectura estable | Una ventana exploratoria | Pesos/horizontes/extremos | C: diferencias, categorías, rangos y episodios | Tolerancias declaradas | Sin evaluar |
| Disponible al cierre | Fechas de observación | Publicación/revisión | D: metadata, doble snapshot, lag1/2 | Calendario/vintages comprobables | Sin evaluar |
| Desacoples tienen dinámica identificable | Historia corta/autocorrelada | Memoria de normalización | E: gate SINDy y persistencia/VAR | 10 observaciones por término y Neff>=100 | Sin evaluar |
| Contribución novedosa | Antecedentes próximos | Coincidencias/metodologías previas | E: búsqueda dirigida y rivales | Contribución delimitada, sin prioridad global | Sin evaluar |
| Útil para personas | Intención de uso | Sin participantes | Separación explícita de evidencia | Pruebas humanas reales | No ejecutable con recursos actuales |
| Software reproducible | 47 tests y snapshot previo | Pipeline campaña y regresiones | Reejecución desde snapshot/hash | Resultados idénticos salvo timestamps | Sin evaluar |

## Correcciones antes de resultados

La revisión del evaluador detectó que 400 remuestreos no estiman bien las colas de un
intervalo de 99,167% ni resuelven seleccionar entre cinco arquitecturas. Se sustituyó
por intervalo exploratorio al 95% y se retiró cualquier interpretación confirmatoria.
No se aumentó el presupuesto de simulaciones. El panel mínimo conserva exactamente
las tres familias centrales del panel6; los tres contextos adicionales se evalúan
como aportes opcionales, para no favorecer al panel6 por definir seis tareas obligatorias.
La cobertura se divide por todos los meses de contraste. La revisión hipotética afecta
solo enero–diciembre 2021: trasladar toda la historia por una constante sería una prueba
trivial porque no cambia los rangos. Estos cambios preceden todos los resultados de la
campaña y aplican a todas las comparaciones afectadas; constan en [ledger.json](../campaign/ledger.json).

## Corrección posterior al primer lote: separación de la elección final

Después de ver los resultados agregados, retirar M1/M2 parecía satisfacer el margen
puntual de 5% en todo el contraste. Elegirlo y comprobarlo sobre el mismo periodo sería
insuficiente. Se registró [selection-check.json](../campaign/selection-check.json) antes
de ejecutar este control: los mismos modelos ajustados hasta 2020; selección de una
exclusión opcional con 2021–2022 y comprobación con 2023–2026. Se aplicó a las tres
exclusiones, manteniendo el 5%, sin nuevos ajustes, escenarios ni remuestreos. Si ninguna
pasaba selección se conservaría el panel completo; no se escogería otra usando evaluación.

Los investigadores ya conocían los resultados agregados y parte de sus episodios. Este
control separa periodos dentro de la evaluación retrospectiva, pero no es independiente
ni transforma años conocidos en una reserva intacta. Su hash es
`6de1ca219d885a3f757faa95229e5879075aa83fa66a49761bf56bc131cfec0f`.
El resultado fue negativo: ninguna exclusión pasó selección. Las tres se reportan en
[selection_audit.json](../campaign/results/selection_audit.json).
