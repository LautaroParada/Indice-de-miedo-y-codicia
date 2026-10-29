# Encuesta de Crédito Bancario: signo y distribución de respuestas

Verificación metodológica: 29 de octubre de 2026. Las series se seleccionaron del catálogo BDE; la explicación del cálculo se contrastó con las notas oficiales. La consulta de observaciones por API corresponde al colector principal de la investigación.

## Convención oficial

En estándares de aprobación para pymes, el indicador es la proporción que declara flexibilización, en cualquiera de sus dos grados, menos la proporción que declara restricción. Un valor positivo significa flexibilización neta. Cada banco del segmento tiene el mismo peso, independientemente de su tamaño. No es una medida ponderada por monto de crédito. [Notas BDE de oferta a pymes](https://si3.bcentral.cl/siete/ES/Siete/Cuadro/CAP_EXP_ECO/MN_EXP_EC11/ECB_COM_02?idSerie=F089.ECB.IND.COMEA2.T).

En demanda de nuevos créditos, el indicador resta la proporción que percibe solicitudes más débiles de aquella que las percibe más fuertes. Un valor positivo significa fortalecimiento neto. Es una percepción declarada por bancos, no una encuesta directa a empresas ni el crecimiento efectivo del crédito. [Notas BDE de demanda de pymes](https://si3.bcentral.cl/siete/ES/Siete/Cuadro/CAP_EXP_ECO/MN_EXP_EC11/ECB_COM_25?idSerie=F089.ECB.IND.COMDNC2.T).

Estas convenciones permiten orientar los indicadores hacia mayor disponibilidad y demanda de crédito, pero no prueban que cualquiera de ellos mida codicia. Crédito de emergencia o refinanciamiento pueden elevar la demanda durante episodios adversos.

## Cinco categorías disponibles por API

Las siguientes series del catálogo son trimestrales, están vigentes, empiezan en enero de 2003 y tienen unidad «Porcentaje de respuestas». La tabla de origen es «Banca de empresas: créditos comerciales, Estándares de aprobación, Pymes».

| Respuesta | Código |
|---|---|
| Menos restrictivos | `F089.ECB.MNRES.COMEA2.T` |
| Moderadamente menos restrictivos | `F089.ECB.MMNRES.COMEA2.T` |
| Sin Cambios | `F089.ECB.SCAM.COMEA2.T` |
| Moderadamente más restrictivos | `F089.ECB.MMSRES.COMEA2.T` |
| Más restrictivos | `F089.ECB.MSRES.COMEA2.T` |

La metadata completa está en `extras.json`. El indicador oficial agregado es `F089.ECB.IND.COMEA2.T`.

## Descomposición propuesta, explícitamente derivada

Sean `p+` la suma de las dos respuestas de flexibilización, `p−` la suma de las dos de restricción y `p0` la respuesta sin cambios. Todas están expresadas en puntos porcentuales.

- **Saldo neto:** `N = p+ − p−`. Reproduce el indicador oficial.
- **Amplitud de cambio:** `B = p+ + p− = 100 − p0`. Describe cuántas respuestas informan algún cambio.
- **Coexistencia de direcciones:** `C = 2 × min(p+, p−) = B − |N|`. Describe cuánto cambio en sentidos opuestos queda oculto por la compensación del saldo.

Un saldo cero puede corresponder tanto a todos los bancos sin cambios (`B=0`, `C=0`) como a la mitad flexibilizando y la mitad restringiendo (`B=100`, `C=100`). Esta distinción se obtiene de categorías publicadas; no requiere inferir una distribución a partir de la mediana.

`C` no es otro voto independiente para un promedio: es una transformación de las mismas respuestas. Conviene presentarlo como diagnóstico junto con `N`. Tampoco equivale a incertidumbre psicológica, dispersión cardinal, polarización estadística validada ni distancia entre opiniones. Su interpretación es estrictamente descriptiva. La fórmula es una identidad matemática propuesta para la herramienta; no se afirma originalidad académica.

## Controles antes del uso

1. Exigir las cinco categorías disponibles en el mismo trimestre; no imputar ausencias a cero.
2. Comprobar valores entre 0 y 100 y suma cercana a 100. Una tolerancia de 0,5 puntos porcentuales admite redondeo de cinco categorías a un decimal; revisar discrepancias mayores sin renormalizarlas silenciosamente.
3. Reconstruir `N` y contrastarlo con el indicador publicado con una tolerancia de 0,5 puntos. Si no reproduce, detener esa observación y revisar definición o cobertura.
4. Mantener frecuencia trimestral para diagnosticar correlaciones y redundancia. Su repetición mensual no crea observaciones independientes.
5. Alinear por disponibilidad pública, no por el inicio del trimestre que codifica la fecha de observación.
6. No inventar intervalos de confianza sin tamaños muestrales ni microdatos. Los cambios en participantes pueden afectar el resultado.

## Otros insumos solicitados

`extras.json` incluye la tasa real a dos años `F022.BUF.TIS.AN02.UF.Z.D` y el IPC general empalmado `F074.IPC.IND.Z.EP23.C.M`. El IPC inicia en diciembre de 2009; por tanto, el crecimiento interanual de crédito real con este deflactor comienza como máximo en diciembre de 2010. Deflactar antes de calcular la variación: `100 × [log(crédito_t / IPC_t) − log(crédito_t−12 / IPC_t−12)]`.

La compensación a cinco años debe restar tasas nominales y reales del mismo plazo; no se compara directamente con EEE a 23 meses. El fondeo usa tasa interbancaria mensual menos la media mensual de TPM. El turnover divide transacciones mensuales por capitalización compatible; su intensidad no identifica por sí sola entusiasmo o liquidación.
