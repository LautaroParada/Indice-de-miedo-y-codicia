# Cierre crítico de literatura y posición metodológica

Fecha de revisión: 29 de octubre de 2026. Alcance: herramienta pública MyM con datos obtenidos exclusivamente desde la API del Banco Central de Chile (BCCh). La literatura se usa para contrastar interpretaciones y exigencias de evidencia; no incorpora datos de otros proveedores.

## Dictamen

La contribución defendible es **evaluar cuánto mejora una lectura desagregada del estado financiero chileno cuando las dimensiones discrepan, frente a una adaptación competente de un índice agregado**. No está demostrada una nueva medida de sentimiento, un factor puro de apetito por riesgo ni una metodología inédita. Tampoco basta agregar más series: el aporte depende de información incremental, estabilidad y utilidad de lectura.

Los antecedentes ya contienen factores comunes, subíndices, contribuciones por variable, percentiles recursivos y agregación sensible a las correlaciones. La combinación de esos recursos en una herramienta abierta puede aportar accesibilidad y reproducibilidad. Para sostener una contribución académica adicional debe sobrevivir a hipótesis rivales concretas, detalladas abajo.

La búsqueda se cerró con **12 llamadas de consulta: 7 a Scite Papers y 5 a web**, incluyendo en estas últimas las lecturas dirigidas. Se seleccionaron **7 antecedentes**. No es una revisión sistemática ni una búsqueda de anterioridad exhaustiva. Todo el equipo conoce la exploración anterior: el nuevo contraste temporal no constituye una validación formalmente independiente de las decisiones ya exploradas.

## Qué se mide y qué no queda identificado

| Constructo | Evidencia consultada | Consecuencia para MyM |
|---|---|---|
| Apetito por riesgo | Gai y Vause distinguen preferencias, riesgo y probabilidades de valoración; el apetito verdadero no se observa directamente. | Precios, spreads y volatilidades son resultados compatibles con varias causas. Un puntaje no identifica preferencias por sí mismo. |
| Sentimiento inversor | Baker y Wurgler lo vinculan a creencias que no se justifican por los fundamentos disponibles. | Un IPSA alto o una valoración alta no prueba exuberancia: faltaría separar fundamentos y creencias. |
| Confianza de hogares | Ludvigson encuentra que parte importante de su contenido se comparte con otras variables y que el aporte adicional al gasto es limitado en su aplicación. | IPEC debe conservar su significado de encuesta. No equivale a disposición de inversionistas a asumir riesgo. |
| Condiciones financieras | DT905 resume restricciones y holguras macrofinancieras; DT1080 distingue causas de movimientos de precios. | Separar estado observado e interpretación causal. Una subida de tasas admite distintas explicaciones. |
| Estrés sistémico | CISS distingue estrés materializado y su extensión simultánea entre segmentos. | Desacuerdo entre segmentos y estrés generalizado son objetos diferentes; no etiquetar automáticamente una brecha como crisis. |

Fuentes: [Gai y Vause (2005/2006)](https://www.bankofengland.co.uk/working-paper/2005/measuring-investors-risk-appetite), [Baker y Wurgler (2007)](https://doi.org/10.1257/jep.21.2.129), [Ludvigson (2004)](https://www.aeaweb.org/articles?id=10.1257/0895330041371222), [Álvarez et al. (2021)](https://www.bcentral.cl/documents/33528/133326/DTBC_905.pdf/098a96a4-1314-3dd7-213d-d7f8881fa5a4?t=1693318657310), [Albagli et al. (2026)](https://www.bcentral.cl/contenido/-/detalle/publicaciones/investigacion-economica/documento-de-trabajo-1080) y [Holló et al. (2012)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf).

**Ortogonalidad** es una propiedad de una representación y una muestra. PCA no acredita independencia económica o causal, ni garantiza mantener esa propiedad fuera del ajuste. **Baja redundancia** significa que una variable no repite excesivamente las otras bajo el criterio elegido. **Utilidad incremental** exige mejorar una tarea frente a un comparador. Ninguna implica automáticamente las restantes. Por ejemplo, una serie independiente de todo pero dominada por ruido puede ser inútil. Son criterios metodológicos de esta revisión, no hallazgos empíricos nuevos.

## DT905: especificación exacta y adaptación honesta

DT905 utiliza once variables, marzo de 1994–junio de 2020; exige correlación absoluta ≤0,8. Fija medias, desviaciones y PCA en marzo de 1994–diciembre de 2005; combina las variables estandarizadas y divide por la desviación de esa combinación en la base. Contrasta PCA expansivo y pesos iguales con signos. Véanse pp. 6–8; ecuaciones 3–4; tabla 1, p. 19; apéndice, p. 13. Las páginas del visor PDF suman tres a la numeración impresa. [Álvarez et al. (2021)](https://www.bcentral.cl/documents/33528/133326/DTBC_905.pdf/098a96a4-1314-3dd7-213d-d7f8881fa5a4?t=1693318657310).

La siguiente correspondencia es una **evaluación propia del archivo local `research/candidates.json`**, no equivalencia certificada por los autores. Estar en ese archivo no demuestra identidad de universo, frecuencia o construcción con la serie del artículo.

| Variable de DT905, tabla 1 | Insumo disponible en candidates | Dictamen para adaptación BCCh |
|---|---|---|
| BCU a 10 años | `real10` | Coincidencia conceptual más cercana. Revisar empalmes, periodicidad y tratamiento mensual. |
| Depósitos menos soberano a 3 meses | `bank_spread`, `interbank`, `tpm` | Ninguno reproduce esa diferencia. `bank_spread` permite representar otro margen de intermediación, con nombre distinto. `interbank−tpm` es otra fricción. |
| Spread corporativo–soberano real | `real5` | Falta la pata corporativa de plazo comparable. No sustituirla por tasa comercial total sin declarar cambio de riesgo y plazo. |
| EMBI Chile | Ninguno | Omitir en adaptación actual; no sustituir por dólar o IDIE como si fueran riesgo soberano. |
| Precio/utilidad IPSA | `price_earnings` | RPU de Bolsa de Santiago: falta acreditar mismo universo. Usar solo como proxy declarado de valoración. |
| Volatilidad IPSA | `ipsa` mensual | Puede construirse dispersión de retornos mensuales; no recupera volatilidad diaria ni intramensual. |
| Brecha crédito/PIB | `commercial_credit`, `cpi` | Permiten crecimiento real del crédito comercial, no brecha agregada crédito/PIB. Faltan denominador y especificación de tendencia. |
| Producción industrial, cambio anual | `imacec` | Sustituto de actividad más amplio; no misma actividad sectorial. |
| Términos de intercambio, cambio anual | `copper` | Precio del cobre no incluye precios de importación ni ponderaciones del comercio. Solo proxy externo. |
| Tipo de cambio real | `usd`, `cpi` | Insuficientes para TCR multilateral: faltan precios extranjeros y ponderaciones. No fabricar equivalencia. |
| VIX | Ninguno | Omitir hasta verificar serie BCCh apropiada. Volatilidad local de tasas no es incertidumbre implícita accionaria global. |

La agrupación ya es posible en DT905: “the aggregation of user-specified subgroups” (p. 7); la figura 2, p. 15, muestra contribuciones. Por tanto, separar canales no establece novedad frente a ese antecedente. [Álvarez et al. (2021)](https://www.bcentral.cl/documents/33528/133326/DTBC_905.pdf/098a96a4-1314-3dd7-213d-d7f8881fa5a4?t=1693318657310).

**Selección razonable con lo existente:** `real10`, `bank_spread`, `price_earnings`, volatilidad de retornos mensuales de `ipsa`, crecimiento real anual de `commercial_credit/cpi`, crecimiento anual de `imacec` y crecimiento anual de `copper`. Son siete canales candidatos para un comparador adaptado, no una decisión de que exactamente siete bastan. `usd` puede ser una variante de sensibilidad separada; no debe ocupar automáticamente el lugar del TCR. La cartera final requiere revisar calidad y redundancia en el ajuste. Si el catálogo permite recuperar las definiciones exactas faltantes, tienen prioridad frente a estos sustitutos.

La adaptación competente conserva la disciplina de **estandarización y pesos calculados en una base previa**, y una variante expansiva. No hereda coeficientes estimados para otras series y años. Debe publicar matriz de insumos, transformaciones, signo del factor, varianza explicada y contribuciones; usar las mismas fechas y condiciones de información que el panel rival. Su nombre correcto es «adaptación BDE inspirada en DT905». Una PCA indiscriminada de todas las series o un promedio de percentiles sin control comparable debilitarían artificialmente al rival.

## Antecedentes que limitan la novedad

**CISS.** Quince medidas, cinco segmentos, transformación por distribución empírica y agregación con correlaciones cambiantes ya forman parte del antecedente. Un pasaje recuperado por Scite precisa que el indicador reconoce cuando “high stress prevails in several market segments at the same time”. El texto oficial, pp. 1–6, distingue coincidencia de tensión y tensión localizada. Por ello, usar percentiles, observar varios canales o incorporar sus relaciones no constituye por sí solo una contribución nueva. La pregunta propia puede ser si la *divergencia* mercado–confianza ayuda a interpretar episodios que el factor común comprime, sin llamar a esa divergencia riesgo sistémico. [Holló et al. (2012)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf).

**DT1080.** Identifica siete shocks mediante un VAR estructural con restricciones de signo y datos diarios; no se limita a mostrar correlaciones. Su introducción advierte que movimientos similares pueden tener causas diferentes. El objetivo de interpretación por causas y la construcción de un ICF interpretable tienen antecedentes cercanos dentro del propio BCCh. La herramienta MyM no adquiere esa identificación al residualizar variables o convertirlas a percentiles. [Albagli et al. (2026), pp. 1, 4–5](https://www.bcentral.cl/documents/33528/133326/Documento%2Bde%2BTrabajo%2B1080.pdf/da737d8e-ca08-2c12-312e-4ae66c6ab069?t=1779914428045).

**Sentimiento y apetito.** La definición de Baker y Wurgler incluye “is not justified by the facts at hand”. Gai y Vause reconocen que “the true path of investors' risk appetite remains unobserved”. Ambos pasajes fueron recuperados mediante Scite, del resumen y texto del artículo respectivamente. Con señales agregadas no basta observar una correlación plausible con episodios conocidos para separar expectativas fundamentales, percepción de riesgo y disposición a soportarlo. Se puede ofrecer un panel informativo sin fingir esa identificación. [Baker y Wurgler (2007)](https://doi.org/10.1257/jep.21.2.129); [Gai y Vause (2005)](https://doi.org/10.2139/ssrn.872695).

**Confianza del consumidor.** El resumen editorial de Ludvigson describe una “relatively modest amount of additional variation” atribuible a información independiente de confianza en su ejercicio sobre consumo. Esto no invalida IPEC ni se traslada cuantitativamente a Chile; exige contrastarlo frente a controles macrofinancieros y no confundir disponibilidad con aporte. [Ludvigson (2004)](https://www.aeaweb.org/articles?id=10.1257/0895330041371222).

## Contribución y explicaciones rivales falsables

La formulación modesta evaluable es: **con datos BCCh, una representación por dimensiones y divergencias conserva información interpretativa relevante que se pierde al comprimirla en un escalar, bajo una complejidad y costo de actualización bajos**. El verbo «conserva» es deliberado: una transformación determinista no crea información ausente en sus componentes.

Si `M` y `C` son mercado y confianza, `A=(M+C)/2` y `D=M−C`. Entonces `M=A+D/2` y `C=A−D/2`. Demostración algebraica: `(A,D)` equivale a `(M,C)`. Comparar solo `A` contra `(A,D)` prueba una pérdida por compresión; no prueba un mecanismo nuevo. Además, diferencias entre percentiles de variables distintas expresan posiciones relativas en sus respectivas historias, no una brecha de bienestar o precios en unidades económicas naturales.

| Hipótesis que disputa la contribución | Contraste que realmente discrimina | Resultado que restringe la conclusión |
|---|---|---|
| R1. El panel es un ICF reetiquetado. | Comparar contra adaptación DT905 con contribuciones y contra sus primeros factores; misma información y calendario. | Si reproduce las mismas lecturas y no mejora una tarea externa, solo aporta presentación o implementación. |
| R2. La brecha añade una identidad algebraica, no evidencia económica. | Contrastar escalar, escalar+brecha y componentes originales; usar un resultado BCCh excluido de los insumos o una tarea de lectura evaluada. | Ganar solo al escalar no acredita superioridad sobre los componentes. La capacidad de reconstruir componentes es esperable. |
| R3. El supuesto apetito refleja ciclo real o shock externo. | Condicionar la comparación a actividad, cobre y cambio; examinar sensibilidad de signos y contribuciones. | Si la lectura depende del control, describir el canal observable y abandonar la etiqueta psicológica pura. |
| R4. La mejora depende de una crisis, corte o ventana. | Contrastes temporales y por episodios sin reajustar el criterio a cada resultado; informar dependencia serial y tamaño efectivo. | Resultados concentrados en COVID o un único bloque son evidencia localizada, no robustez general. |
| R5. La complejidad vence por mayor número de parámetros. | Mismo presupuesto de ajuste; persistencia, promedio, PCA y VAR/ridge parsimonioso como rivales. | Si la mejora desaparece tras penalizar complejidad o en un tramo temporal reservado, no mantenerla por sofisticación aparente. |
| R6. Las encuestas repiten las variables de mercado. | Evaluar valor adicional de hogares y empresas por separado y frente a sus componentes originales. | Ausencia de mejora no obliga a borrarlas del tablero, pero limita el reclamo de información adicional. |

Ejemplos de tareas externas admisibles, sin recomendaciones de inversión: distinguir respuestas de oferta/demanda en la Encuesta de Crédito Bancario que **no** entren en el indicador; comparar episodios de financiación con mismo nivel del agregado; medir si usuarios identifican correctamente qué dimensión diverge. La encuesta no es «verdad de miedo/codicia», y una misma serie no puede servir de ingrediente y validación independiente de ese ingrediente. Estas son condiciones para interpretar los resultados de la campaña, no resultados ya obtenidos.

La evidencia retrospectiva puede justificar utilidad descriptiva y una hipótesis académica. No autoriza afirmar causalidad, superioridad universal, originalidad demostrada ni resultados de inversión. Una separación temporal sobre datos ya revisados tampoco reconstruye la disponibilidad histórica: requiere fechas efectivas y, cuando corresponda, versiones de publicación.

## SINDy: límite identificable, no prestigio metodológico

El antecedente de SINDy recuperado en Scite advierte que “the resulting derivative matrix may have large noise magnitude”. Propone regresión dispersa sobre una biblioteca de funciones y discute representación de baja dimensión. La fuente no establece que cualquier conjunto pequeño de índices financieros permita recuperar leyes de movimiento. [Brunton et al. (2016), artículo y material metodológico](https://doi.org/10.1073/pnas.1517384113).

En esta aplicación, la identificación de dinámica sigue sin estar acreditada. Puntajes de ventana móvil dependen de historia adicional; encuestas y precios mezclan frecuencias, ruido, revisión y respuestas a shocks no observados. Diferenciar numéricamente esos puntajes puede amplificar artefactos. Contar más filas que coeficientes no demuestra excitación suficiente ni unicidad del modelo.

**Dictamen:** SINDy no sostiene el argumento de novedad en el estado documentado. Solo merece resultado sustantivo si la campaña demuestra, antes de interpretar coeficientes, recuperación de una dinámica conocida con muestreo/ruido comparables, rango y condicionamiento de la biblioteca, soporte estable de términos y mejora temporal frente a persistencia y un modelo lineal comparable. Una formulación discreta evita estimar derivadas, pero sigue siendo identificación de un mapa condicional, no prueba de ecuaciones económicas causales. Si esos criterios no se cumplen, el resultado correcto es «no identificable o sin ventaja demostrada», conservando el panel descriptivo.

## Calidad de evidencia y registro de consulta

En todos los resultados Scite inspeccionados, **el campo `editorialNotices` estaba ausente**, no vacío. Se registra «no informado»; no equivale a ausencia comprobada de correcciones, preocupaciones o retractaciones. Las fichas primarias consultadas no ofrecieron un historial editorial completo. No se usaron conteos de citas como prueba de validez.

| Antecedente seleccionado | Acceso efectivo | Alcance de lo verificado |
|---|---|---|
| DT905 | PDF oficial completo, texto indexado; intento de capturas con una respuesta vacía y un fallo de caché | Variables, base, normalización, variantes, páginas y tablas mediante texto del PDF. Sin validación visual de la captura. No apareció en las búsquedas Scite acotadas. |
| DT1080 | Ficha y PDF oficiales | Autoría, fecha, introducción y formulación de identificación; no auditoría empírica del artículo completo. No apareció en la consulta Scite exacta. |
| CISS | PDF oficial y extractos de texto completo Scite | Agregación, subíndices, correlaciones y transformación recursiva. Dos registros SSRN corresponden al mismo antecedente; se cuenta una vez. |
| Baker–Wurgler | Metadatos/resumen y Smart Citations de su propio texto vía Scite | Constructo y ejemplos de proxies; no replicación del índice ni inferencia de validación para Chile. |
| Gai–Vause | Texto completo fragmentario Scite de WP2005; fichas BoE e IJCB | Distinción conceptual y método. WP2005 y publicación2006 se cuentan como un antecedente, no dos estudios independientes. |
| Ludvigson | Resumen editorial AEA y metadatos Scite | Conclusión general del resumen; Scite marcó `contentDenied: true`, sin pasajes de texto completo. |
| Brunton–Proctor–Kutz | Metadatos y pasajes de texto completo Scite | Biblioteca, reducción y ruido en derivadas; no una garantía para sistemas económicos observacionales. |

### Log cerrado: 12 llamadas

Las lecturas por URL y capturas se agruparon en llamadas web; su contenido se enumera para que el conteo sea auditable. Solo una llamada web incluyó búsqueda por palabras. No se descargaron series de fuentes externas ni se contrataron accesos.

| # | Servicio | Consulta o recuperación | Resultado y uso |
|---|---|---|---|
| 1 | Web | Abrir PDF oficial DT905 | Fuente primaria de la especificación. |
| 2 | Scite | `titles=[Economic Growth at Risk: An Application to Chile; A Financial Conditions Index for Chile; CISS - A Composite Indicator of Systemic Stress in the Financial System]`, `term=principal components financial conditions index variables`, `limit=5` | 2 resultados CISS; el segundo título era aproximado y no identificó DT1080. |
| 3 | Web | DT905 líneas345 y705 | Métodos, tablas y páginas del apéndice. |
| 4 | Scite | Cuatro títulos: Baker–Wurgler, Ludvigson, Gai–Vause, Brunton et al.; `limit=5` | Búsqueda por títulos demasiado amplia: 82 coincidencias, 5 devueltas irrelevantes. Se excluyeron todas. |
| 5 | Web | Abrir PDF CISS y ficha DT1080 | Recuperación primaria. |
| 6 | Scite | DOI `10.1257/jep.21.2.129`, `10.1257/0895330041371222`, `10.1073/pnas.1517384113`; `term=sentiment OR confidence OR noise OR measurements`, `limit=3` | 3 antecedentes correctos; registro de acceso y pasajes. |
| 7 | Scite | `title=Measuring Investors' Risk Appetite`, `author=Gai`, `limit=2` | 1 resultado, WP2005; texto disponible. |
| 8 | Scite | DOI CISS2012, Baker–Wurgler y Gai–Vause; `term=orthogonal OR correlation OR risk aversion`, `limit=3` | 3 resultados, contraste conceptual y de agregación. |
| 9 | Web | Buscar `site:ijcb.org "Gai" "Vause" "Measuring Investors"`; abrir AEA por DOI Ludvigson | Se usaron solamente IJCB/BoE como fuentes primarias de la búsqueda; no se incorporaron otros antecedentes retornados. |
| 10 | Scite | `title:("Economic Growth at Risk: An Application to Chile") OR title:("Fundamental Drivers of Financial Conditions")`, `limit=2` | 0 resultados. Verificación de ambos documentos mediante BCCh, sin afirmar inexistencia en Scite. |
| 11 | Scite | DOI SINDy; `term=coordinates OR basis OR measurements OR sparse`, `limit=1` | 1 resultado; representación y biblioteca. |
| 12 | Web | Capturas DT905 páginasPDF22 y10; AEA líneas23; ficha IJCB; PDF enlazado DT1080 | Una captura vacía y otra con error de caché; resto recuperado. No se reiteró para respetar el cupo. |

## Referencias

Álvarez, N., Fernandois, A., & Sagner, A. (2021). *Economic growth at risk: An application to Chile* (Documento de Trabajo N.º 905). Banco Central de Chile. [PDF oficial](https://www.bcentral.cl/documents/33528/133326/DTBC_905.pdf/098a96a4-1314-3dd7-213d-d7f8881fa5a4?t=1693318657310).

Albagli, E., Carlomagno, G., Ledezma, J., & Reszczynski, M. T. (2026). *Fundamental drivers of financial conditions* (Documento de Trabajo N.º 1080). Banco Central de Chile. [Ficha oficial](https://www.bcentral.cl/contenido/-/detalle/publicaciones/investigacion-economica/documento-de-trabajo-1080).

Baker, M., & Wurgler, J. (2007). Investor sentiment in the stock market. *Journal of Economic Perspectives, 21*(2), 129–151. [https://doi.org/10.1257/jep.21.2.129](https://doi.org/10.1257/jep.21.2.129).

Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences, 113*(15), 3932–3937. [https://doi.org/10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).

Gai, P., & Vause, N. (2005). *Measuring investors' risk appetite* (Working Paper N.º 283). Bank of England. [Ficha oficial](https://www.bankofengland.co.uk/working-paper/2005/measuring-investors-risk-appetite). Versión publicada: Gai, P., & Vause, N. (2006). Measuring investors' risk appetite. *International Journal of Central Banking, 2*(1), 167–188. [Artículo IJCB](https://www.ijcb.org/journal/v2n1/measuring-investors-risk-appetite). Un antecedente, dos versiones.

Holló, D., Kremer, M., & Lo Duca, M. (2012). *CISS—A composite indicator of systemic stress in the financial system* (Working Paper Series N.º 1426). European Central Bank. [PDF oficial](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf).

Ludvigson, S. C. (2004). Consumer confidence and consumer spending. *Journal of Economic Perspectives, 18*(2), 29–50. [https://doi.org/10.1257/0895330041371222](https://doi.org/10.1257/0895330041371222).
