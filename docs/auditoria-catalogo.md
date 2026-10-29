# Auditoría del catálogo BDE para una segunda versión de investigación

Fecha de trabajo: 29 de octubre de 2026. Fuente: snapshot distribuido por `bcch-sdk`, procedente del archivo `catalogo_series_bde_2026-10-29.xlsx`. Se hizo un censo de los 25.291 códigos y 30.920 contextos de los 20 capítulos, seguido de búsquedas dirigidas por mecanismo y revisión de sus nombres, tablas, unidades, frecuencias, primeras fechas y vigencia. **Esto no equivale a leer y validar individualmente las 25.291 metodologías ni a verificar todos sus endpoints.** La auditoría inicial fue local y no accedió a credenciales. Posteriormente se contrastó la metodología ECB con páginas oficiales del BCCh, documentado en `ecb_distribuciones.md`.

## Hallazgo central

Restringir el índice a series diarias eliminaba de partida fuentes más próximas a actitudes y decisiones económicas: confianza de hogares y empresas, valoración y participación bursátil, estándares de oferta crediticia, demanda de crédito y composición de liquidez. La disponibilidad diaria no es un criterio suficiente de pertinencia conceptual. La API permite construir un sistema de evidencia mucho más rico que dólar, cobre, tasa larga e incertidumbre textual.

Tampoco basta con añadir todas estas series a un promedio. Un plazo de bonos adicional, una unidad monetaria alternativa, una desagregación regional o el total junto con sus componentes pueden duplicar el mismo mecanismo. El número de series es resultado de selección y validación, no objetivo de calidad.

## Censo de capítulos y papel potencial

Las cantidades son filas de contexto, no códigos únicos por capítulo.

| Capítulo | Contextos | Papel en esta investigación |
|---|---:|---|
| Sector Externo | 2.227 | Controles de exposición externa y flujos; distinguir transacciones de valorización de stocks |
| Principales Estadísticas Macro | 158 | Resúmenes duplicados de otros capítulos, no votos independientes |
| Estadísticas Monetarias y Financieras | 1.248 | Composición monetaria, cantidades de crédito y fondeo |
| Estadísticas Históricas | 119 | Posibles referencias largas; riesgo de rupturas metodológicas y series discontinuadas |
| Precios | 2.040 | Deflactores y controles; inflación no representa por sí sola miedo o codicia |
| Economía Internacional | 375 | Controles globales publicados en BDE; no requieren otros proveedores API |
| Cuentas Nacionales | 12.411 | Control de actividad y contraste con fundamentos; PIB sectorial no es sentimiento financiero |
| Indicadores Mercado de la Vivienda | 114 | Valuación, actividad y morosidad; contexto patrimonial con mayor rezago |
| Regionales | 4.074 | Heterogeneidad territorial, pertinente para un producto contextual; no inflar el índice nacional |
| Mercado de Capitales | 1.102 | Momentum bursátil, valoración, participación, mora y estructura bancaria |
| Derivados y Spot | 2.530 | Posicionamiento por contraparte y cobertura; gran riqueza pero quiebres/historias cortas |
| Mercado Laboral y Demografía | 581 | Controles del ciclo real y expectativas de empleo; no señales directas de apetito financiero |
| Género | 90 | Desagregaciones para análisis distributivo; fuera del núcleo de un termómetro agregado de mercado |
| Operaciones de Mercado Abierto | 770 | Política y facilidades de liquidez; separar respuesta oficial de estrés privado |
| Tasas de Interés | 202 | Spreads de fondeo, intermediación, curva nominal/real y costo de financiamiento |
| Dólar Observado y Tipos de Cambio | 167 | Precios y presión cambiaria; comprobar redundancia con cobre y posiciones |
| Estadísticas Experimentales | 1.036 | Fuentes potenciales de actividad empresarial y otras innovaciones; revisar cambios de cobertura |
| Expectativas Económicas | 718 | Confianza declarada, expectativas de precios/política y oferta/demanda de crédito |
| Finanzas Públicas | 618 | Contexto fiscal y controles de prima soberana; rezagos y causalidad indirecta |
| Indicadores Sectoriales | 340 | Controles de actividad y heterogeneidad, no sustitutos de sentimiento |

## Universo candidato entregado

`catalog_candidates.json` contiene **45 códigos distintos**, elegidos antes de evaluar sus correlaciones o resultados. Incluye metadata exacta del snapshot y razón de uso; las transformaciones son propuestas iniciales. Cada fila dice `availability=catalog_only_not_live_verified`. El JSON no contiene enlaces autenticados ni tokens.

No todos los candidatos son factores finales. Algunos son denominadores, controles, comparadores o alternativas deliberadamente redundantes que permiten comprobar qué conservar. `sign=0` significa que el sentido económico no está identificado todavía, o que la serie solo tiene sentido dentro de una transformación compuesta. **No convertir ese cero en una puntuación de índice ni asignar automáticamente un signo por correlación con la versión anterior.**

| Familia | Aporte que la versión de cuatro fuentes omitía | Ejemplos de códigos |
|---|---|---|
| Bolsa | Momentum mensual, valoración y participación | `F013.IBC.IND.N.7.LAC.CL.CLP.BLO.M`; `F022.MB6.FLU.Z.Z.Z.M`; `F022.MB4.FLU.Z.Z.Z.M` |
| Hogares | Percepción y disposición declarada | `F089.IPE.IND.75M2.M`; `F089.IPE.IND.72M2.M`; `F089.ICC.IND.B4M.M` |
| Empresas | Confianza de quienes toman decisiones productivas | `G089.IME.IND.A0.M`; `G089.IME.IND.A04.M` |
| Crédito: precio | Margen de intermediación y fricción de fondeo | `F022.SBR.TIN.D089.NO.Z.M`; `F022.TIB.TIP.D001.NO.Z.M` |
| Crédito: oferta y demanda | Distinguir racionamiento de crédito de debilidad de demanda | `F089.ECB.IND.COMEA2.T`; `F089.ECB.IND.COMDNC2.T` |
| Crédito: cantidad y riesgo realizado | Expansión de colocaciones y deterioro observado | `F022.COMBI.STO.Z.Z.CLP.M`; `F022.IM4.IND.Z.Z.Z.M` |
| Liquidez | Composición de tenencias monetarias | `F021.M1.STO.N.CLP.5.M`; `F021.M2.STO.N.CLP.5.M`; `F021.DPP.STO.N.CLP.5.M` |
| Curva e inflación de mercado | Pendiente y compensación, frente a solo volatilidad larga | `F022.BCLP.TIS.AN02.NO.Z.D`; `F022.BUF.TIS.AN05.UF.Z.D` |
| Expectativas | Revisiones de trayectorias esperadas, distintas del nivel observado | `F089.TCN.PRE.14.M`; `F089.IPC.V12.15.M`; `F089.TPM.TAS.14.M` |
| Posicionamiento | Cantidades y heterogeneidad de contrapartes | `F099.SPT.STO.Z.MCF.Z.Z.NET.Z.MMUSD.MLME.Z.Z.0.D`; posiciones NDF y derivados por contraparte |

## Transformaciones que agregan significado

1. **Participación bursátil:** monto transado / capitalización, con frecuencia mensual compatible. El volumen nominal solo confunde crecimiento del mercado, inflación y actividad. Un turnover alto puede acompañar entusiasmo o liquidación; conviene mostrarlo condicionado al movimiento de precios, no imponerle sentido positivo universal.
2. **Valuación relativa:** bolsa/libro y, solo tras verificar la definición y dominio, RPU. El catálogo etiqueta ambos en porcentaje, por lo que no debe suponerse automáticamente que el segundo es un P/E en veces. Una valoración elevada no identifica valor justo ni rentabilidad futura.
3. **Diferencia de confianza:** contraste entre confianza empresarial sin minería y hogares. Permite identificar desacoples entre agentes. El agregado IMCE y su versión sin minería son alternativas; incluir ambos con el mismo peso duplica información.
4. **Fricción de fondeo:** tasa interbancaria de un día, media mensual, menos TPM media mensual. Mantener unidades anuales compatibles. Usar valor absoluto o solo diferencial positivo exige una hipótesis explícita sobre el mecanismo de estrés.
5. **Oferta frente a demanda de crédito:** indicadores trimestrales ECB por pymes, grandes empresas y hogares. Son respuestas declaradas de bancos, no cantidades efectivas ni encuestas a empresas. La verificación oficial posterior confirma signo positivo para flexibilización neta de estándares y fortalecimiento neto de demanda. La distribución de cinco categorías permite distinguir inmovilidad de cambios opuestos que se compensan; ver `ecb_distribuciones.md` y `extras.json`. No interpolar tres observaciones mensuales ficticias por trimestre.
6. **Preferencia por liquidez:** cuota M1/M2 o depósitos a plazo/M2, antes que sumar niveles nominales. La sensibilidad a política monetaria, medios de pago y retiros previsionales impide una lectura unívoca de miedo/codicia.
7. **Curva nominal y real:** pendiente nominal10–2 y compensación nominal–real al mismo plazo. La compensación incluye primas de riesgo y liquidez, no únicamente inflación esperada.
8. **Expectativas frente a realizaciones:** la brecha entre EEE de tipo de cambio a11meses y spot describe depreciación esperada. No es una estimación de incertidumbre ni dispersión.
9. **Posicionamiento institucional:** cambios en posiciones netas frente a no residentes o fondos de pensiones pueden revelar desacoples entre agentes. Requieren verificar de qué lado está expresado el signo; cobertura regulatoria y especulación son explicaciones rivales.

Si se necesita comparar compensación de inflación con EEE23meses, la tasa real a2años `F022.BUF.TIS.AN02.UF.Z.D` también figura en el catálogo desde2005-01-25. No restar una compensación a5años de una expectativa a23meses como si fueran el mismo horizonte.

## Ausencias y límites encontrados

- No se identificó un índice oficial de condiciones financieras ICF con ese nombre en el catálogo. `F089.EOF.FI_TPM_CFL.1A.D` es la mediana de un factor de incidencia de condiciones financieras locales en TPM declarado en la EOF; **no es un ICF oficial**.
- Las búsquedas por deciles, cuartiles, percentiles, dispersión y extremos en metadata de expectativas no identificaron distribuciones EEE. Aparecen medianas y preguntas de crédito cuyo nombre contiene «máximo» o «mínimo»; estas no son cuantiles de expectativas. No fabricar desacuerdo a partir de una sola mediana.
- No se identificó una serie explícita de número de operaciones mediante esa búsqueda. Los montos transados no son conteos de transacciones.
- No se identificó breadth bursátil individual, como cantidad de acciones que suben/bajan. IGPA relativo a IPSA puede mostrar divergencia de índices, pero no es breadth por sí solo.
- La familia nueva F099 de derivados/spot por contraparte generalmente empieza en2022. No prometer validación de crisis2008/2020 con esas series. La posición neta spot del MCF sí registra inicio2001; ambas historias no se empalman sin documentar definiciones.
- ICC mensual empieza en2018; mora agregada en2015; volumen bursátil en2012. Deben distinguirse un panel largo para comparación histórica y módulos recientes con cobertura específica.
- El archivo no contiene calendarios de publicación ni vintages. Fechas de observación no equivalen a fechas de disponibilidad. La evaluación temporal debe ser declarada retrospectiva o usar supuestos de rezagos explícitos hasta construir un archivo de publicaciones.

## Selección empírica requerida

La hipótesis es que estos mecanismos agregan información respecto de las cuatro fuentes iniciales, no que los45 códigos sean útiles u ortogonales. La consulta real debe comprobar historia, cobertura, actualizaciones, valores constantes y disponibilidad. Después deben examinarse correlaciones sobre transformaciones, cambios de redundancia por subperiodo y contribución fuera de muestra.

Los factores correlacionados pueden agruparse o representarse por un candidato interpretable. Una ortogonalización matemática puede conservarse como diagnóstico o descomposición secundaria; no garantiza independencia causal y puede remover la señal común de estrés que se quiere medir. Los controles de actividad, política y entorno externo ayudan a describir qué parte es común, sin presentar sus residuos como sentimientos psicológicos observados.

Para usuarios, una salida defendible combina una lectura principal sencilla, sus mecanismos y los desacoples visibles entre hogares, empresas, bancos y mercados. No necesita convertir cada celda del catálogo en una fórmula opaca.
