# Definiciones operacionales y límites de lectura

Este documento fija qué representan los once componentes de un panel descriptivo de seis familias: mercado, confianza, fricción —alias técnico—, participación, crédito real y composición monetaria. Se conserva el Panel 6 original como **referencia no validada**, no como ganador forzado. La propuesta de Panel 5 se descartó tras la auditoría de separación temporal de la selección. Los tres primeros perfiles conservan exactamente sus componentes y fórmulas de la campaña. **La campaña no respalda un resultado agregado bipolar de miedo/codicia.** Conservar información en varios perfiles no identifica emociones ni preferencias por riesgo.

Las fórmulas se contrastaron con `research/analyze.py`, `campaign/engine.py` y el contrato de la campaña. Las fuentes bibliográficas y del catálogo se reutilizan de las revisiones ya documentadas; este anexo no realizó nuevas consultas externas. Los ejemplos económicos de la tabla son **contraejemplos hipotéticos**, no episodios históricos verificados.

## Qué significa un número entre 0 y 100

Para cada componente transformado $x_t$, el puntaje compara la observación actual con observaciones válidas **anteriores** dentro de la ventana de $w$ meses calendario:

$$
p_t^{(w)}=100\frac{\#\{x_s<x_t\}+0.5\#\{x_s=x_t\}}{n_t},
\qquad s\in\{t-w,\ldots,t-1\}.
$$

La base usa hasta 60 meses anteriores y exige al menos 36 observaciones válidas. La comparación de referencias de 36, 60 y 120 meses exige la ventana completa correspondiente. No incorpora la observación actual a su propia referencia. Un valor faltante no se transforma en cero ni en neutralidad.

- En un **componente**, 75 expresa posición relativa frente a su historia, con tratamiento de empates; no una probabilidad del 75 % de que ocurra algo.
- En una **familia**, el puntaje es el promedio de posiciones de sus componentes. No es necesariamente el percentil de la propia familia: un promedio de percentiles no tiene distribución uniforme por definición.
- Las categorías son convenciones de lectura: **bajo** si $p<25$, **central** si $25\le p\le75$, **alto** si $p>75$. «Alto» no significa codicia ni «bajo» miedo. Los cortes tampoco identifican sobrevaloración, crisis o solvencia.
- Cero no significa ausencia del fenómeno, y 100 no representa un máximo económico absoluto. Indican extremos respecto de una referencia acotada.
- El puntaje puede cambiar porque sale una observación antigua de la ventana, aun cuando el valor actual en unidades naturales permanezca igual.

El uso de distribuciones empíricas tiene antecedentes en indicadores compuestos, pero no valida las orientaciones ni la interpretación psicológica de esta herramienta. [Holló et al. (2012)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf).

## Diccionario de los once componentes

Notación: $t$ es un mes; $d$, una observación diaria; una barra denota promedio mensual observado; $P$, IPSA; $B$, valor Bolsa/Libro publicado; $I$, IPEC; $E$, IMCE sin minería; $S$, spread bancario publicado; $i$, tasa interbancaria; $r$, TPM; $y$, tasa nominal soberana a diez años; $U$, IDIE; $V$, monto transado; $K$, capitalización bursátil; $C$, saldo de crédito comercial; $IPC$, índice de precios. «Aumenta» se refiere a la variable transformada, antes de convertirla en un rango.

| Componente / familia | Fórmula y mecanismo observable | Si aumenta… | Contraejemplo económico hipotético | Lectura permitida | No permite afirmar |
|---|---|---|---|---|---|
| `ipsa` / mercado | $\ln(P_t)-\ln(P_{t-3})$. Variación acumulada del índice mensual publicado durante tres meses. | La variación de precios es mayor: puede ser más positiva o menos negativa. | Mejores utilidades de exportadoras elevan precios, con preferencias de riesgo constantes. | «El movimiento bursátil reciente ocupa una posición alta en su referencia». | «Los inversionistas tienen más codicia»; retorno futuro positivo. |
| `price_book` / mercado | $B_t$, sin transformar su nivel. Valoración bursátil relativa al valor contable según la serie publicada. | La razón publicada es mayor. | Una reducción contable del patrimonio eleva la razón sin aumento del precio. | «La valoración relativa al libro publicado es mayor». | Sobrevaloración, burbuja o alejamiento del valor justo. |
| `ipec` / confianza | $I_t$, nivel de la encuesta. Percepción económica declarada por hogares. | Mejora el valor del índice de percepción. | Mejora el empleo esperado, pero los hogares mantienen sus mismas decisiones de cartera. | «La percepción declarada es más favorable respecto de su historia». | Mayor apetito inversor, participación bursátil o consumo efectivamente realizado. |
| `imce_nonmining` / confianza | $E_t$, nivel de IMCE sin minería. Confianza empresarial declarada en ese universo. | Mejora el índice de confianza empresarial. | Aumentan pedidos de empresas domésticas sin cambio en las preferencias de inversionistas financieros. | «Las empresas encuestadas declaran un entorno más favorable». | Sentimiento de todos los inversionistas; aislamiento causal completo del cobre o la minería. |
| `bank_spread` / fricción | $S_t$, spread bruto de 30–89 días publicado por BDE. Margen de intermediación observado. | El margen publicado es mayor. | Cambia la composición de operaciones o su riesgo crediticio, sin cambio de actitud hacia el riesgo. | «El spread publicado ocupa una posición más alta». | Restricción de oferta identificada, margen bancario neto o miedo de los depositantes. |
| `funding_deviation` / fricción | $\lvert i_t-\bar r_t\rvert$. Distancia entre tasa interbancaria mensual y TPM media mensual. | La distancia respecto de TPM es mayor. | Exceso de liquidez lleva la tasa por debajo de TPM; escasez la lleva por encima. La misma distancia admite ambas situaciones. | «Hay mayor desviación absoluta frente a la referencia de política». Conservar también el signo para interpretar. | Estrés de fondeo o escasez de liquidez identificados a partir del valor absoluto. |
| `rate_volatility` / fricción | $\operatorname{sd}_{d\in t}(y_d-y_{d-1})$, desviación poblacional de cambios diarios observados. | Los cambios diarios de la tasa son más dispersos. | Una sorpresa de desinflación provoca ajustes rápidos de tasas, sin pérdida de solvencia ni miedo colectivo. | «La tasa soberana nominal varió más dentro del mes». | Volatilidad del retorno de bonos, probabilidad de impago o aversión al riesgo identificada. |
| `idie` / fricción | $\bar U_t$, media mensual del índice diario de incertidumbre económica basado en texto. | Aumenta el nivel de incertidumbre textual medido. | Un debate de política pública eleva la discusión incierta sin endurecimiento de las condiciones bancarias. | «El indicador textual de incertidumbre es relativamente elevado». | Fricción financiera identificada, temor de la población o encuesta de sentimiento inversor. |
| `turnover` / participación | $V_t/K_t$, con capitalización positiva. Intensidad de transacción relativa al tamaño bursátil. | Se transa más respecto del denominador, o disminuye ese denominador. | Una recomposición de un índice genera muchas operaciones sin cambio de convicción; también puede caer la capitalización con volumen constante. | «La rotación bursátil relativa es mayor». | Compras netas, entradas netas de capital, liquidez medida por spread/profundidad o más codicia. |
| `real_credit_growth` / crédito real | $\ln(C_t/IPC_t)-\ln(C_{t-12}/IPC_{t-12})$. Crecimiento real anual del saldo comercial. | El saldo real crece más, o se contrae menos. | Empresas pesimistas utilizan líneas de crédito para cubrir necesidades de caja. Aumenta el saldo, aunque empeore su confianza. | «El saldo comercial deflactado muestra mayor variación anual». | Crecimiento de crédito nuevo, mejora de acceso, inversión productiva o optimismo identificados. |
| `liquid_share` / composición monetaria | $M1_t/M2_t$, con M2 positivo. Proporción del agregado M1 dentro de M2, usando los valores mensuales publicados. | M1 representa una fracción mayor de M2; puede aumentar el numerador o disminuir el denominador. | Empresas trasladan depósitos a cuentas de uso inmediato para pagar remuneraciones e impuestos, sin modificar sus preferencias por riesgo. | «La composición monetaria muestra mayor peso relativo de M1». | Huida hacia efectivo, cautela inversora, liquidez de mercado o preferencia psicológica identificadas. |

Las fuentes de definición de variables son los metadatos de BDE preservados en [el inventario de candidatos](../research/candidates.json) y el [catálogo documentado](catalogo.md). La interpretación de los mecanismos y los contraejemplos son análisis propio. La distinción entre preferencias, riesgos, creencias y resultados observados se apoya en [Gai y Vause (2005)](https://www.bankofengland.co.uk/working-paper/2005/measuring-investors-risk-appetite) y [Baker y Wurgler (2007)](https://doi.org/10.1257/jep.21.2.129). La lectura del IDIE como incertidumbre textual reutiliza la ficha primaria de [Becerra y Sagner (2020)](https://www.bcentral.cl/contenido/-/detalle/documento-de-trabajo-n-883).

### Unidades naturales y detalles que cambian la interpretación

Los **once componentes transformados** proceden de **quince series de entrada**. No son once emociones, once mecanismos independientes ni once fuentes institucionales independientes.

| Insumo | Código BDE | Unidad y tratamiento local |
|---|---|---|
| IPSA mensual | `F013.IBC.IND.N.7.LAC.CL.CLP.BLO.M` | Índice publicado; no asumir una cotización diaria o de cierre no especificada por el metadato. |
| Bolsa/Libro | `F022.MB6.FLU.Z.Z.Z.M` | Valor de la razón publicado. El catálogo lo rotula «porcentaje de utilidad»; esa etiqueta es ambigua. No convertirlo a «veces» o a porcentaje económico sin aclarar la ficha. |
| IPEC | `F089.IPE.IND.75M2.M` | Puntos del índice de encuesta; no porcentaje de personas ni puntaje normalizado del panel. |
| IMCE sin minería | `G089.IME.IND.A04.M` | Puntos del índice de encuesta; no magnitud de crecimiento productivo. |
| Spread bruto, 30–89 días | `F022.SBR.TIN.D089.NO.Z.M` | Unidad original etiquetada «Porcentaje» en BDE; se usa el valor publicado, sin reconstruir otra definición de margen. |
| Tasa interbancaria mensual | `F022.TIB.TIP.D001.NO.Z.M` | Tasa en porcentaje. |
| TPM diaria | `F022.TPM.TIN.D001.NO.Z.D` | Tasa en porcentaje; se calcula media mensual observada. |
| Bonos nominales a diez años | `F022.BCLP.TIS.AN10.NO.Z.D` | Tasa en porcentaje; los cambios diarios y su dispersión se expresan en puntos porcentuales. |
| IDIE diario | `F029.IDIE.IND.TOT.D` | Puntos del índice publicado; media mensual observada. |
| Monto transado | `F022.MB4.FLU.Z.Z.Z.M` | Millones de pesos; numerador de la rotación mensual. |
| Capitalización bursátil | `F022.MB2.STO.Z.Z.Z.M` | Millones de pesos; denominador positivo de la rotación. |
| Crédito comercial | `F022.COMBI.STO.Z.Z.CLP.M` | Miles de millones de pesos según metadato; su escala monetaria se cancela en la diferencia logarítmica. |
| IPC general | `F074.IPC.IND.Z.EP23.C.M` | Índice publicado; se usa como deflactor, no como tasa de inflación directa. |
| M1 | `F021.M1.STO.N.CLP.5.M` | Miles de millones de pesos; numerador de la composición monetaria. |
| M2 | `F021.M2.STO.N.CLP.5.M` | Miles de millones de pesos; denominador positivo de la composición monetaria. |

La diferencia logarítmica se conserva internamente como fracción. Para mostrar una variación porcentual exacta se usa $100(\exp(x)-1)$; $100x$ es una aproximación. La rotación $V/K$ también es una razón: multiplicar por 100 expresa esa razón en porcentaje, **no** convierte el dato en el puntaje histórico. Lo mismo ocurre con M1/M2: su puntaje de 50 no significa que M1 represente la mitad de M2. La desviación de tasas en puntos porcentuales se multiplica por 100 para expresarla en puntos base; no se anualiza como si fuera volatilidad de retornos.

Las medias de series diarias requieren al menos tres observaciones válidas en el mes. La variabilidad de tasas exige al menos diez cambios diarios observados y omite saltos entre fechas separados por más de siete días. No reemplaza esa dispersión por un cambio entre medias mensuales. La desviación de fondeo es el valor absoluto de una diferencia mensual: **no** es el promedio de desviaciones absolutas diarias. Son operaciones distintas.

La API del BCCh es el único canal de obtención de datos. Que BDE distribuya una serie no implica que el BCCh produzca su estadística original: el catálogo atribuye algunas a encuestas, Bolsa de Santiago o investigadores. El código de serie fija el dato usado; una similitud de nombre no autoriza sustituir otro universo. [Banco Central de Chile, BDE y documentación API](https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/acceso_api.html).

## De los componentes a seis perfiles

Sea $p_j$ el puntaje histórico del componente $j$. La especificación descriptiva mantiene:

| Perfil | Fórmula | Alcance |
|---|---|---|
| Mercado | Media de `ipsa` y `price_book` | Movimiento reciente y valoración relativa; no preferencia pura por riesgo. |
| Confianza | Media de `ipec` e `imce_nonmining` | Dos universos de opinión económica; mostrar también hogares y empresas por separado. |
| Fricción, alias técnico | Media de `bank_spread`, `funding_deviation`, `rate_volatility` e `idie` | Mezcla de margen, desviación monetaria, variabilidad e incertidumbre. Debe leerse con sus cuatro componentes visibles. |
| Participación | `turnover` | Intensidad relativa de negociación; no dirección de los flujos. |
| Crédito real | `real_credit_growth` | Dinámica de saldo comercial deflactado; no identificación de oferta ni demanda. |
| Composición monetaria | `liquid_share` | Peso relativo de M1 dentro de M2; no preferencia por riesgo identificada. |

La media exige todos los componentes de cada perfil. Cuando falta uno, el perfil queda sin cálculo; no se redistribuye su peso automáticamente. Los seis perfiles no se suman en un indicador bipolar, y un valor alto no tiene el mismo sentido económico en todos ellos. La denominación «fricción» debe entenderse como alias conservado del experimento: **la incertidumbre no es una fricción identificada**, ni una media de proxies identifica por sí sola un canal causal.

La confianza tampoco equivale a apetito por riesgo. La evidencia previa sobre confianza del consumidor documenta solapamiento con otras variables; no prueba una equivalencia psicológica ni garantiza utilidad incremental en Chile. [Ludvigson (2004)](https://www.aeaweb.org/articles?id=10.1257/0895330041371222). Movimientos de precios semejantes pueden proceder de causas diferentes; una interpretación estructural requiere restricciones y evidencia adicionales. [Albagli et al. (2026)](https://www.bcentral.cl/contenido/-/detalle/publicaciones/investigacion-economica/documento-de-trabajo-1080).

### Por qué se descartó el Panel 5 y se conserva M1/M2

El primer lote sugería retirar `liquid_composition`: sobre todo el periodo 2021–2026, el mayor deterioro **puntual** de MAE frente a `panel6` era **4,083 %**, inferior al margen descriptivo registrado de 5 %. Ese resultado se conserva en el registro, pero **no justifica retener Panel 5**. Utilizar el mismo periodo para elegir una exclusión y comprobarla confunde selección y evaluación.

La corrección del evaluador se registró después del primer lote y **antes de calcular esta auditoría**, en [selection-check.json](../campaign/selection-check.json). Mantiene el margen de 5 % y lo aplica a las tres exclusiones opcionales —participación, crédito real y composición monetaria— sobre los seis objetivos ECB. La selección usa 2021–2022; la comprobación posterior usa 2023–2026. Solo una exclusión que cumpla en todos los objetivos del primer tramo puede continuar. No se permite elegir otra usando el segundo tramo ni acumular exclusiones evaluadas por separado.

**Ninguna de las tres exclusiones cumple la primera etapa.** En particular, retirar M1/M2 empeora el MAE de demanda de pymes en **7,23 %** durante 2021–2022, con 8 trimestres. Esto basta para descartar Panel 5 bajo la regla registrada. La comparación descriptiva del segundo tramo muestra además un deterioro de **10,19 %** en oferta de crédito de consumo durante 2023–2026, con 14 trimestres; no se utiliza para seleccionar una alternativa distinta. Ambos deterioros se miden frente al Panel 6 completo. Resultado conservado en `selection_audit.json`: ninguna exclusión seleccionada y referencia `panel6`.

La auditoría no crea una muestra de reserva intacta: los investigadores ya conocían los resultados agregados y la exploración de 2021–2026. Es una comprobación retrospectiva de una corrección explícita, no validación independiente. Conserva todos los resultados, sin nuevos ajustes, arquitecturas, escenarios de sensibilidad ni remuestreos.

La no inferioridad estadística **tampoco está establecida**: en el contraste agregado de oferta de consumo, el intervalo exploratorio de ganancia al retirar M1/M2 llega a aproximadamente −17,8 %, compatible con un deterioro mucho mayor que 5 %. Esos intervalos no son pruebas confirmatorias de no inferioridad y no corrigen multiplicidad. Conservar las seis familias es una decisión de referencia ante una simplificación no respaldada; no demuestra que cada una tenga utilidad independiente ni convierte al Panel 6 en un ganador validado. Fuentes: [auditoría de selección](../campaign/results/selection_audit.json), [métricas económicas](../campaign/results/economics_metrics.json) y [contrato original](../campaign/contract.json).

## Cómo leer el rango de escenarios y la abstención

Para cada uno de los tres perfiles principales se comparan las referencias de 36, 60 y 120 meses completos. El intervalo entre el menor y el mayor puntaje es una **envolvente de escenarios de normalización**. No es un intervalo de confianza, no tiene cobertura probabilística del 95 % y no cuantifica toda la incertidumbre posible. Pesos, revisiones y rezagos tienen otras pruebas y otros resultados.

Solo se asigna una categoría estable cuando están disponibles los tres horizontes, coinciden sus categorías y la amplitud no supera 20 puntos. En caso contrario corresponde abstenerse de una etiqueta única. La historia insuficiente se cuenta como abstención al calcular cobertura sobre todos los meses del contraste; no se excluye para mejorar el porcentaje.

La envolvente responde «cuánto cambia esta descripción al cambiar la referencia histórica». Los IC exploratorios del ensayo ECB responden otra pregunta: variación de una diferencia de error bajo remuestreo de los trimestres observados. Ninguno transforma el panel en una recomendación de inversión, una medición directa del sentimiento o un pronóstico validado. Las fechas de observación y las versiones revisadas tampoco reconstruyen automáticamente lo que estaba publicado en tiempo real.

## Referencias

Albagli, E., Carlomagno, G., Ledezma, J., & Reszczynski, M. T. (2026). *Fundamental drivers of financial conditions* (Documento de Trabajo N.º 1080). Banco Central de Chile. [Ficha oficial](https://www.bcentral.cl/contenido/-/detalle/publicaciones/investigacion-economica/documento-de-trabajo-1080).

Baker, M., & Wurgler, J. (2007). Investor sentiment in the stock market. *Journal of Economic Perspectives, 21*(2), 129–151. [https://doi.org/10.1257/jep.21.2.129](https://doi.org/10.1257/jep.21.2.129).

Banco Central de Chile. (s. f.). *Base de Datos Estadísticos y acceso mediante API*. [BDE](https://si3.bcentral.cl/siete); [documentación API](https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/acceso_api.html). Metadatos utilizados: archivo de catálogo suministrado por el usuario, etiquetado `catalogo_series_bde_2026-10-29.xlsx`; [procedencia y huella documentadas](catalogo.md). La etiqueta del archivo no certifica fecha de actualización de cada serie.

Becerra, J. S., & Sagner, A. (2020). *Twitter-based economic policy uncertainty index for Chile* (Documento de Trabajo N.º 883). Banco Central de Chile. [Ficha oficial](https://www.bcentral.cl/contenido/-/detalle/documento-de-trabajo-n-883). Se reutiliza la descripción de la ficha ya revisada; no se atribuyen resultados a texto completo no recuperado.

Gai, P., & Vause, N. (2005). *Measuring investors' risk appetite* (Working Paper N.º 283). Bank of England. [Ficha oficial](https://www.bankofengland.co.uk/working-paper/2005/measuring-investors-risk-appetite).

Holló, D., Kremer, M., & Lo Duca, M. (2012). *CISS—A composite indicator of systemic stress in the financial system* (Working Paper Series N.º 1426). European Central Bank. [PDF oficial](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf).

Ludvigson, S. C. (2004). Consumer confidence and consumer spending. *Journal of Economic Perspectives, 18*(2), 29–50. [https://doi.org/10.1257/0895330041371222](https://doi.org/10.1257/0895330041371222).

Los registros Scite previos no informaron `editorialNotices`; esa omisión no acredita ausencia de avisos editoriales. El acceso y los límites de cada antecedente están detallados en [el cierre de literatura](cierre-literatura.md) y, para IDIE, en [fundamentos](fundamentos.md).
