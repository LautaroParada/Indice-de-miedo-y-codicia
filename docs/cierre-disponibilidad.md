# Disponibilidad temporal, revisiones y contrastes reservados

Auditoría del 29 de octubre de 2026. Alcance: los 52 códigos de `research/candidates.json`, documentación oficial de la API y modelos del SDK local. La evidencia estructurada está en [`campaign/availability_evidence.json`](../campaign/availability_evidence.json). Esta auditoría no consulta credenciales ni realiza descargas autenticadas. La comparación de snapshots corresponde a la campaña empírica.

## Qué fechas se conocen

| Concepto | Evidencia disponible | Lo que no permite afirmar |
|---|---|---|
| Observación | `GetSeries`: `indexDateString`, `value`, `statusCode`, código y títulos de la serie | Que el dato estaba publicado ese día |
| Extensión y actualización | `SearchSeries`: `seriesId`, `frequencyCode`, títulos, `firstObservation`, `lastObservation`, `createdAt`, `updatedAt` | Fecha de primera publicación o revisión de cada observación |
| Descarga | Hora UTC registrada por nuestro colector | Fecha de publicación oficial |
| Publicación | Calendario o comunicado oficial, cuando se identifica | Hora de incorporación a la API ni valores de todas las entregas anteriores |
| Versión histórica | Snapshot conservado o código explícito de compilación | Reconstrucción de cualquier fecha pasada a partir de una descarga actual |

`firstdate` y `lastdate` filtran el período observado. No son una fecha de corte informativo. La documentación de REST/SOAP revisada no ofrece un parámetro general `as_of` ni una fecha de publicación por observación. `statusCode=OK` tampoco certifica que una cifra sea definitiva. El SDK conserva el JSON original y admite fechas ausentes en metadata, sin inventarlas. [Documentación oficial de la API](https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/documentacion.html).

No corresponde afirmar que BCCh carece totalmente de versiones históricas. El catálogo contiene **869 códigos** con sufijos de año y mes de compilación en familias de cuentas nacionales: PIB, consumo, demanda, inversión, exportaciones e importaciones. El conteo usa el patrón `20\d{2}(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)\.` y no pretende enumerar todo posible mecanismo de versiones. Ninguno pertenece a los 52 códigos del estudio; no se identificó un juego equivalente para su IMACEC. Los códigos alternativos por año base tampoco son automáticamente vintages.

## Publicación y rezagos por familia

| Familia | Evidencia primaria | Consecuencia para el estudio |
|---|---|---|
| Estadísticas monetarias y financieras mensuales | Primera versión el día 7 del mes siguiente, o hábil posterior; versión definitiva con desfase de dos meses y 23 días; posibles revisiones excepcionales | Un mes es una aproximación de disponibilidad inicial. Dos meses no garantizan cifras definitivas ni eliminan revisiones |
| ECB trimestral | Publicación en enero, abril, julio y octubre. En 2026: 14 de enero, 15 de abril, 15 de julio y 14 de octubre | Fechar el trimestre por su cierre y mantener aparte la publicación; nunca adelantar el resultado al primer mes del trimestre |
| EEE mensual y EOF | Calendario de eventos; EEE de octubre de 2026 el 13, EOF previa a RPM el 22 | La EEE puede estar disponible durante su propio mes; EOF tiene frecuencia de eventos, aunque la API la clasifique como diaria |
| IMACEC | BCCh informa publicación 31 días después del cierre del mes; agosto de 2026 figura el 1 de octubre | En cortes de fin de mes, usar dos meses como supuesto prudente sin calendario histórico exacto; no afirmar disponibilidad en el mes observado |
| IPC | Calendario INE: septiembre de 2026 se publicó el 8 de octubre | Un mes como escenario de disponibilidad. El calendario del productor no prueba cuándo ingresó a BDE |
| Derivados mensuales | Calendario BCCh: informe de septiembre de 2026 el 23 de octubre | Es evidencia de un evento de esa familia; no demuestra el rezago exacto de cada serie diaria o mensual |
| Bolsa, IPEC, IMCE, ICC, IDIE y series diarias de mercado | No se reconstruyó un historial de incorporación por código a la API | Mantener incertidumbre explícita y escenarios; frecuencia diaria no equivale a publicación instantánea |

Fuentes por fila: [calendario monetario y política de revisión](https://www.bcentral.cl/calendario-estadistico?categoria=Estadisticas+Monetarias&nowYear=2026&p_p_id=bcentral_calendario_global_CalendarioGlobalPortlet_INSTANCE_gvZr3D7oZBKa&p_p_lifecycle=0&p_p_mode=view&p_p_state=normal), [encuestas 2026](https://www.bcentral.cl/en/content/-/detalle/encuestas-de-expectativas-2026), [preguntas IMACEC](https://www.bcentral.cl/es/web/banco-central/contenido/-/details/preguntas-imacec-preguntas-frecuentes-estadisticas-1), [agenda INE](https://ineportaldpr.ine.gob.cl/inicio/agendaestadistica/), [calendario general BCCh](https://www.bcentral.cl/es/calendario-estadistico). El calendario general se recuperó por contenido indexado; su apertura directa falló. Son reglas vigentes y ejemplos de 2026, **no un historial verificado de publicaciones de 2006–2026**. La consulta al INE se limita a documentación; todas las observaciones del estudio siguen llegando por la API BCCh.

## Contrato para sensibilidad temporal

Los escenarios `L0`, `L1` y `L2` desplazan la disponibilidad supuesta de cada observación mensual a **fin de su mes de referencia más 0, 1 o 2 meses**, respectivamente. Primero se identifica el período; nunca se usa el día 1 que codifica un mes como fecha efectiva de publicación. Los insumos diarios se agregan solo cuando el mes está cerrado; para componentes derivados, manda la última disponibilidad de todos sus insumos.

`L0` es un escenario retrospectivo optimista, especialmente inadecuado como supuesto operativo para crédito, agregados, IPC y actividad. `L1` y `L2` miden sensibilidad a demoras, pero conservan los valores revisados descargados hoy. No constituyen backtests en tiempo real, cotas garantizadas ni reparación de revisión histórica. La comparación debe mantener iguales la muestra evaluada, el horizonte y el procedimiento de ajuste; el rezago no se escoge según el resultado más favorable.

Para ECB, usar el cierre del trimestre como período y la publicación documentada cuando exista; en ausencia de ella, fin del mes siguiente como supuesto. Desplazar 0/1/2 meses desde el cierre trimestral sirve únicamente como sensibilidad de calendario. Evaluar una fila por trimestre. Si se presenta información ya publicada en una vista mensual, repetirla no aumenta el tamaño muestral.

Dos snapshots iguales prueban igualdad entre esas dos descargas, no ausencia de revisiones pasadas. Dos snapshots distintos exigen separar observaciones añadidas, valores revisados, estados modificados y datos retirados. Conservar los dos archivos, sus hashes y sus horas UTC; la fecha exacta del cambio queda acotada entre consultas, salvo otra evidencia. No sobrescribir la versión anterior.

## Contrastes que pueden reservarse

| Uso | Alias y códigos | Restricción contra circularidad |
|---|---|---|
| Oferta pyme | `ecb_supply_sme` — `F089.ECB.IND.COMEA2.T` | Solo frente a un panel sin ECB ni sus categorías. El saldo es una identidad de sus cinco proporciones |
| Demanda pyme | `ecb_demand_sme` — `F089.ECB.IND.COMDNC2.T` | Pregunta distinta; mantener fuera del panel y del ajuste de selección/pesos |
| Oferta de consumo | `ecb_supply_households` — `F089.ECB.IND.CONSEA1.T` | Alias heredado: representa consumo, no todos los hogares ni crédito hipotecario |
| Demanda de consumo | `ecb_demand_households` — `F089.ECB.IND.CONSDNC1.T` | Misma restricción; no es cantidad efectiva de crédito |
| Control de actividad | `imacec` — `F032.IMC.IND.Z.Z.EP18.Z.Z.0.M` | Nivel original empalmado; usar variación interanual explícita, no variación mensual como si estuviera desestacionalizado |
| Control de precios | `cpi` — `F074.IPC.IND.Z.EP23.C.M` | IPC general; para inflación calcular variación interanual. No es target independiente de un predictor que ya lo usa como deflactor |

Las cuatro preguntas ECB no son duplicados algebraicos entre sí, pero comparten encuesta y parte de los informantes; no son cuatro validaciones independientes. Una relación descriptiva con demanda u oferta no identifica sentimiento psicológico ni causalidad. Mayor demanda puede responder a refinanciamiento o necesidad de liquidez. Reservar estos contrastes impide usarlos para optimizar retrospectivamente los mismos factores que se evalúan. La oferta pyme queda excluida como target del módulo de distribución pyme; IMACEC e IPC son controles, no pruebas directas de miedo o codicia.

## Unidades y perímetros que requieren cuidado

- **Crédito comercial:** `F022.COMBI.STO.Z.Z.CLP.M` es un saldo nominal de balances individuales en miles de millones de pesos. No es flujo de crédito nuevo ni crédito exclusivamente a pymes. Para crecimiento real, deflactar por IPC antes de diferenciar; la base del índice cambia la escala del nivel real, no su crecimiento logarítmico. El IPC seleccionado comienza en diciembre de 2009.
- **Mora:** `F022.IM4.IND.Z.Z.Z.M` figura como «Índice» en el catálogo. El IEF presenta mora de 90 días o más como porcentaje de colocaciones y advierte el perímetro de consolidación. Mantener el nombre «índice de mora total» y señalar la discrepancia hasta verificar la definición exacta de ese código. No convertirlo en número de deudores morosos ni identificar su denominador con crédito comercial de balances individuales. [Definición y consolidación en IEF 2026](https://www.bcentral.cl/documents/d/banco-central/informe-de-estabilidad-financiera-primer-semestre-2026-pdf?download=true).
- **Bolsa/libro y RPU:** el catálogo y el buscador oficial etiquetan ambos como porcentaje de utilidad. Esto no autoriza a denominarlos automáticamente múltiplos en veces ni a interpretar umbrales de valor razonable. Conservar escala original y marcar definición pendiente. Capitalización y transacciones sí comparten millones de pesos; su cociente es un flujo mensual sobre stock, no conteo de operaciones. [Metadata bursátil publicada](https://si3.bcentral.cl/Search/Buscar/Index/?or=pop&pageID=1&st=mensual%7D&t=Mercado+Capitales+Mensual%7CMercado+de+Capitales).
- **Tasas:** diferencias entre porcentajes se expresan en puntos porcentuales; multiplicar por 100 para puntos base. El spread bruto bancario no es un spread soberano de incumplimiento. Compensación de inflación: nominal menos real al mismo plazo, con primas de riesgo y liquidez; no equipararla a EEE de otro horizonte.
- **Posiciones y volúmenes:** los F099 están expresados en millones de dólares; posición neta, monto vigente y flujo transado son objetos distintos. El signo de la posición depende de la contraparte y no identifica automáticamente apuestas especulativas. No agregar magnitudes netas y brutas como votos equivalentes.
- **Encuestas y liquidez:** EEE es mediana, no dispersión; sus alias `12`/`24` corresponden a horizontes catalogados de 11/23 meses. EOF de condiciones financieras es un factor de incidencia sobre TPM, no un ICF. M1 es parte de M2; sus niveles y su cociente no son tres señales independientes. IPEC total y expectativas, IMCE total y sin minería, así como saldo ECB y categorías, requieren control de redundancia.

La evidencia temporal disponible permite una campaña retrospectiva con supuestos de rezago explícitos. La validación prospectiva requiere conservar desde ahora snapshots y primeras fechas de captura; esa fecha acota la disponibilidad observada por el sistema, sin reemplazar una fecha de publicación oficial que no se conoce.

## Registro local de capturas

Desde la raíz del repositorio:

```console
python -m campaign.prospective --snapshot campaign/snapshot-reference --registry outputs/prospective.sqlite
```

El comando no consulta la API ni programa ejecuciones. Verifica los 52 alias, códigos y SHA256 antes de escribir. SQLite conserva el identificador inmutable de cada snapshot, el estado actual por `(alias, date)` con `first_seen_at` y `last_seen_at`, y un historial de altas y revisiones con valores y estados anteriores/nuevos. Devuelve cantidades añadidas, modificadas y sin cambios, número de snapshots y máxima primera captura.

`first_seen_at` significa primera captura en los snapshots suministrados, **no publicación oficial**. Cuando existe una hora por serie se conserva; en un manifest con lotes sin asignación individual se usa la mayor hora declarada para las series sin asignación. Se registra también la hora de incorporación a SQLite. El mismo snapshot es idempotente; los nuevos deben importarse en orden temporal estricto. Una observación ausente de una descarga posterior permanece en el historial con su última captura y no se convierte silenciosamente en cero o eliminación.

Reproducir snapshots históricos inicializa un archivo de observaciones, pero no constituye validación prospectiva. El historial de revisiones empieza con las capturas disponibles; no recupera cambios anteriores a ellas.
