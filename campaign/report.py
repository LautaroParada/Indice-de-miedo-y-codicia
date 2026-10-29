"""Informe de cierre y referencia descriptiva derivados de resultados registrados."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import save_json
from .selection import selection_audit

METHODS = ["v2", "equal_family", "dt905_adapted", "minimal3", "panel6"]
NAMES = {
    "v2": "Índice v2",
    "equal_family": "Equiponderado",
    "dt905_adapted": "DT905 adaptado",
    "minimal3": "Panel 3",
    "panel6": "Panel 6",
}
FAMILIES = {
    "market": "Mercado",
    "confidence": "Confianza",
    "friction": "Presiones e incertidumbre",
    "participation": "Participación",
    "real_credit": "Crédito real",
    "liquid_composition": "Composición monetaria",
}
TARGETS = {
    "ecb_supply_sme": "Oferta a pymes",
    "ecb_supply_large": "Oferta a empresas grandes",
    "ecb_supply_households": "Oferta de crédito de consumo",
    "ecb_demand_sme": "Demanda de pymes",
    "ecb_demand_large": "Demanda de empresas grandes",
    "ecb_demand_households": "Demanda de crédito de consumo",
}

REFERENCES = """## Documentación de la campaña

1. [Contrato y criterios registrados](cierre-contrato.md).
2. [Definiciones, fórmulas y límites de interpretación](cierre-definiciones.md).
3. [Disponibilidad, fechas de publicación y versiones](cierre-disponibilidad.md).
4. [Literatura, antecedentes y explicaciones rivales](cierre-literatura.md).
5. [Veredicto completo de la campaña](cierre-veredicto.md).
6. [Matriz de afirmaciones y evidencia](cierre-matriz.md).
"""


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def f(value, digits=1):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _economic_episodes(predictions, config):
    rows = []
    for target, data in predictions.groupby("target"):
        p = data.pivot(index="quarter", columns="model", values="error").abs()
        for begin, end in config["stability"]["episode_windows"]:
            part = p.loc[begin:end]
            if len(part) == 0:
                continue
            for method in METHODS:
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "episode": begin + "/" + end,
                        "n": len(part),
                        "gain_vs_controls": 1 - part[method].mean() / part.controls.mean(),
                    }
                )
        for method in METHODS:
            gains = []
            for date in p.index:
                part = p.drop(index=date)
                gains.append(1 - part[method].mean() / part.controls.mean())
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "episode": "leave_one_quarter",
                    "n": len(p),
                    "min_gain": min(gains),
                    "max_gain": max(gains),
                }
            )
    return pd.DataFrame(rows)


def make_report(results_dir, base_dir, output, config_path=Path("campaign/contract.json")):
    r = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    targets = r["economics"]["targets"]
    diagnostic = r["diagnostic_tasks"]["methods"]
    threshold = cfg["external_economic_test"]["minimal_noninferiority_mae"]
    drop_checks = {}
    for group in cfg["diagnostic_tasks"]["optional_families"]:
        gains = [
            e["models"]["panel6_without_" + group]["gain_vs_full_representation"]
            for e in targets.values()
            if e["status"] == "evaluated"
        ]
        worst_loss = max(-g["relative_mae_gain"] for g in gains)
        drop_checks[group] = {
            "worst_point_mae_loss": worst_loss,
            "within_point_margin_all_targets": bool(worst_loss <= threshold),
            "intervals": [g["ci95"] for g in gains],
        }
    # Retrospective selection/evaluation separation, registered after the first batch.
    selection = selection_audit(
        results_dir / "economics_predictions.csv", output / "selection_audit.json"
    )
    removed = selection["retained_after2023_2026"]
    groups = {k: v for k, v in cfg["families"].items() if k != removed}
    selected = {
        "id": "mym-descriptive-reference-20261029",
        "campaign_verdict": "B_NO_RESPALDADO",
        "verdict_scope": "Aggregated fear/greed and generally stable headline diagnosis",
        "status": "retained descriptive reference, not a validated winning index",
        "reference_method": "panel6_without_" + removed if removed else "panel6",
        "selection_rule": selection["protocol"]["rule"],
        "selection_separation_audit": "selection_audit.json",
        "drop_checks": drop_checks,
        "families": groups,
        "weights": {k: {v: 1 / len(cols) for v in cols} for k, cols in groups.items()},
        "frequency": "monthly; ECB quarterly separate",
        "normalization": cfg["normalization"],
        "missingness": "All components required per profile; do not reweight or interpolate",
        "label_abstention": cfg["stability"]["range_abstention"],
        "cutoff": cfg["data"]["end"],
        "update": (
            "Immutable snapshot with download timestamp; observation-period labels; "
            "no point-in-time claim. Recompute without future observations. "
            "Do not overwrite prior snapshots."
        ),
        "prohibited": [
            "fear/greed labels",
            "direct risk preference measurement",
            "investment recommendations",
            "causal interpretation",
            "real-time backtest",
            "human usability claim",
            "scientific novelty claim",
        ],
    }
    save_json(output / "methodology.json", selected)
    panel = pd.read_csv(results_dir / "panel6.csv", index_col=0, parse_dates=True)[list(groups)]
    panel.to_csv(output / "descriptive_profiles.csv", date_format="%Y-%m-%d")
    raw = pd.read_csv(base_dir / "features.csv", index_col=0, parse_dates=True)
    scores = pd.read_csv(base_dir / "scores.csv", index_col=0, parse_dates=True)
    rows = []
    for family, cols in groups.items():
        for component in cols:
            for date in panel.index:
                rows.append(
                    {
                        "date": date,
                        "family": family,
                        "component": component,
                        "raw_transformation": raw.loc[date, component],
                        "rank60": scores.loc[date, component],
                        "fixed_weight": 1 / len(cols),
                        "contribution": scores.loc[date, component] / len(cols),
                        "profile_complete": bool(pd.notna(panel.loc[date, family])),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "component_trace.csv", index=False, date_format="%Y-%m-%d")
    predictions = pd.read_csv(results_dir / "economics_predictions.csv", parse_dates=["quarter"])
    episodes = _economic_episodes(predictions, cfg)
    episodes.to_csv(output / "economic_episodes.csv", index=False)
    # Snapshot and representation controls are intentionally separate from economic evidence.
    matrix = [
        [
            "Miedo/codicia identificado",
            "Precios, confianza y crédito",
            "No hay preferencias observadas netas de fundamentos",
            "A: mecanismos y objetivos ECB excluidos de los componentes",
            "Identificación discriminante",
            "Evidencia independiente del constructo",
            "ECB mide crédito; no preferencias de riesgo",
            "NO IDENTIFICADO; no autoriza etiqueta",
        ],
        [
            "Un agregado conserva diagnóstico",
            "Cinco representaciones",
            "Composición comprimida",
            "B: reconstrucción temporal de categorías",
            "Exactitud en las familias principales",
            "≥ 90 %",
            "Escalares: 58,8–73,5 %; paneles: 100 % por lectura directa",
            "RECHAZADO como sustituto del desglose",
        ],
        [
            "Detecta oposición extrema",
            "Contraste mensual",
            "Cero positivos",
            "B: mercado < 25 y confianza > 75, o viceversa",
            "Sensibilidad de detección (recall)",
            "Casos positivos observables y ≥ 90 % de acierto",
            "0 casos en 68 meses; recall no evaluable",
            "INCONCLUSO para detección extrema",
        ],
        [
            "Familias aportan a condiciones crediticias",
            "Seis series ECB excluidas de los componentes",
            "Corroboración con objetivos externos a los componentes",
            "B: ridge con controles y ablaciones",
            "MAE e IC 95 % exploratorio",
            "Ganancia ≥ 10 % e intervalo positivo en oferta y demanda",
            "Paneles 3 y 6 cumplen en 3 objetivos cada uno; no hay ganador uniforme",
            "CORROBORACIÓN EXPLORATORIA",
        ],
        [
            "Panel 3 iguala a Panel 6",
            "Mismos periodos",
            "Menor complejidad",
            "B: comparación pareada",
            "Pérdida relativa de MAE",
            "≤ 5 % en cada objetivo ECB",
            "Excede el 5 % en 4 de 6 objetivos",
            "NO CUMPLE margen puntual",
        ],
        [
            "Pesos menores no alteran lectura",
            "24 perturbaciones de ±20 %",
            "Sensibilidad",
            "C: cinco arquitecturas",
            "Mediana / p95 / categoría / extremos",
            "≤ 5 puntos / 10 puntos / 10 % / 1 %, con N ≥ 36",
            (
                f"{r['sensitivity']['weight_cases_passed']}/"
                f"{r['sensitivity']['weight_cases_total']} comparaciones cumplen"
            ),
            "RESPALDADO para esas perturbaciones",
        ],
        [
            "Etiqueta independiente del horizonte",
            "Ventanas de 36, 60 y 120 meses",
            "Cambios legítimos de referencia",
            "C: rango y acuerdo entre categorías",
            "Cobertura de etiqueta",
            "≥ 50 % de los 69 meses por familia principal",
            "Mercado: 53/69; confianza: 28/69; presiones: 6/69",
            "FALLA como titular general",
        ],
        [
            "Información disponible al cierre",
            "Fechas de observación",
            "Publicación y vintages",
            "D: API, calendario y rezagos",
            "Cobertura histórica de publicación",
            "Fechas de publicación por observación y versiones",
            "No disponibles para estas 52 series; nueva descarga impedida por conectividad",
            "SOLO RETROSPECTIVO",
        ],
        [
            "Dinámica SINDy identificable",
            "Tres estados",
            "Poca muestra y autocorrelación",
            "E: criterio previo de identificabilidad",
            "Transiciones y N efectivo",
            "≥ 100 transiciones, N efectivo ≥ 100 y contraste ≥ 36 meses",
            f"{r['dynamics']['training_transitions']} transiciones; N efectivo: 1,55–6,55",
            "RECHAZADO ajuste SINDy en esta muestra",
        ],
        [
            "Novedad metodológica",
            "DT905, CISS, DT1080 y literatura sobre sentimiento",
            "Antecedentes cercanos",
            "E: 19 consultas dirigidas del conjunto de revisores",
            "Correspondencia de contribuciones",
            "Distinguir aporte de técnicas existentes",
            "Los subíndices, percentiles y contribuciones ya tienen antecedentes",
            "NO DEMOSTRADA; aporte técnico reproducible",
        ],
        [
            "Mejora la comprensión o el tiempo de uso",
            "Tareas automatizadas",
            "Sin personas participantes",
            "No se sustituyen usuarios por agentes",
            "Exactitud y tiempo de comprensión humanos",
            "Ensayo con personas",
            "No ejecutado; sin participantes autorizados o disponibles",
            "INCONCLUSO; no se afirma mejora de usabilidad",
        ],
        [
            "Software reproducible",
            "Snapshot de la API y hashes",
            "Reejecución integral",
            "Ejecución completa, pruebas y reproducción",
            "Igualdad numérica y hashes",
            "Mismo resultado a partir del snapshot y la configuración",
            "Se registra aparte en reproduction.json",
            "VERIFICACIÓN TÉCNICA separada",
        ],
    ]
    headers = [
        "Afirmación",
        "Evidencia",
        "Brecha",
        "Experimento",
        "Métrica",
        "Aceptación",
        "Resultado",
        "Veredicto",
    ]
    pd.DataFrame(matrix, columns=headers).to_csv(output / "claims_matrix.csv", index=False)
    claims = (
        "# Matriz de afirmaciones\n\n"
        "Veredicto B: no respaldado para un agregado de miedo/codicia ni para un "
        "titular generalmente estable. La corroboración económica es exploratoria; "
        "la referencia descriptiva no es un índice ganador validado.\n\n"
        + table(headers, matrix)
        + "\n\n"
        + REFERENCES
    )
    (output / "claims_matrix.md").write_text(claims, encoding="utf-8")
    diag_rows = [
        [
            NAMES[m],
            *[
                f(diagnostic[m]["core_category_accuracy"][c] * 100)
                for c in ["market", "confidence", "friction"]
            ],
            f(diagnostic[m]["change_direction_accuracy"] * 100),
            f(diagnostic[m]["dominant_change_accuracy"] * 100),
        ]
        for m in METHODS
    ]
    economic_rows = []
    for t, e in targets.items():
        economic_rows.append(
            [
                TARGETS[t],
                e["train"]["n"],
                e["test"]["n"],
                *[
                    f(e["models"][m]["gain_vs_controls"]["relative_mae_gain"] * 100)
                    for m in METHODS
                ],
            ]
        )
    ablation_rows = []
    for family in cfg["families"]:
        ablation_rows.append(
            [
                FAMILIES[family],
                *[
                    f(
                        -targets[t]["models"]["panel6_without_" + family][
                            "gain_vs_full_representation"
                        ]["relative_mae_gain"]
                        * 100
                    )
                    for t in TARGETS
                ],
            ]
        )
    s = pd.read_csv(results_dir / "sensitivity.csv")
    other = pd.read_csv(results_dir / "source_stress.csv")
    combined = pd.concat([s, other], ignore_index=True)
    stress_rows = []
    for (kind, method), g in combined.groupby(["kind", "method"]):
        stress_rows.append(
            [
                kind,
                NAMES[method],
                f(g.p95_abs.max()),
                f(g.max_abs.max()),
                f(100 * g.category_change_rate.max()),
                f(100 * g.opposite_tail_rate.max()),
                f(g.n.min(), 0),
                f(g.n.max(), 0),
            ]
        )
    env_rows = [
        [
            FAMILIES[k],
            v["stable_n"],
            v["denominator_all_test_months"],
            v["complete_windows_n"],
            f(v["stable_coverage"] * 100),
            f(v["p95_width_common"]),
        ]
        for k, v in r["stable_labels"].items()
    ]
    collisions = r["diagnostic_tasks"]["collisions"]
    collision_rows = [
        [
            c["a"],
            c["b"],
            f(c["aggregate_a"], 2),
            f(c["aggregate_b"], 2),
            f(c["max_profile_difference"], 2),
        ]
        for c in collisions[:3]
    ]
    output_report = f"""# Cierre metodológico — MyM Ingenierías

**Veredicto B: NO RESPALDADO para un índice agregado de miedo/codicia ni para un titular
de diagnóstico generalmente estable.** La campaña se cierra con resultados negativos
y positivos; no constituye una validación del índice.

Los datos permiten describir precios, confianza, participación y condiciones de crédito,
con algunos vínculos económicos con objetivos excluidos de los componentes. Esto no mide
preferencias de riesgo o emociones. Se conserva una referencia descriptiva con componentes,
rango entre horizontes y abstención. No se declara una ganadora validada ni se recomienda invertir.

## Contrato, datos y alcance

Referencia `7dfcedb`; hash del contrato v2:
`{r["contract_sha256"]}`.
Los investigadores conocían toda la exploración previa: no hay una muestra de reserva intacta,
un equipo independiente ni validación externa por separar archivos. Desarrollo hasta 2020;
evaluación desde 2021 hasta septiembre de 2026. Las observaciones de 2006–2026 proceden
exclusivamente de la API del BCCh. El snapshot previo del 29 de octubre de 2026 contiene
52 series y 91.144 filas con hashes; la nueva descarga tampoco pudo conectarse al endpoint
público. No se ampliaron permisos. La reproducción sin conexión no constituye una segunda
consulta exitosa ni una comparación entre versiones históricas de publicación.

Se ejecutaron cinco arquitecturas, nueve ablaciones económicas y
**{r["total_sensitivity_scenarios"]} escenarios de sensibilidad**, dentro del máximo de 200.
Se usaron 400 remuestreos por contraste, bloques de 4 trimestres e IC 95 % exploratorios.
No se formulan conclusiones confirmatorias después de comparar múltiples métodos y objetivos.
Las cifras no prueban causalidad ni capacidad de predicción financiera. Los criterios y
correcciones previas constan en [el contrato](cierre-contrato.md) y
[el registro de la campaña](../campaign/ledger.json).

## Qué pierde el agregado

Exactitud porcentual en 68 meses comunes de contraste y 69 de ajuste. El mínimo era 90 %.
Los paneles 3 y 6 leen sus ejes directamente: el 100 % representa conservación de información,
no validación económica.

{
        table(
            [
                "Representación",
                "Mercado",
                "Confianza",
                "Presiones",
                "Dirección de cambios",
                "Familia que más cambia",
            ],
            diag_rows,
        )
    }

El agregado no sustituye el desglose. Se encontraron
{r["diagnostic_tasks"]["collision_pairs_total"]} pares con una diferencia de puntaje ≤ 2
y al menos un perfil con una diferencia ≥ 40 puntos. Ejemplos:

{
        table(
            ["Mes A", "Mes B", "Agregado A", "Agregado B", "Mayor diferencia de perfil"],
            collision_rows,
        )
    }

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

{table(["ECB", "N ajuste", "N contraste", *[NAMES[m] for m in METHODS]], economic_rows)}

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

{table(["Familia retirada", *TARGETS.values()], ablation_rows)}

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
{r["sensitivity"]["weight_cases_passed"]}/{r["sensitivity"]["weight_cases_total"]}
comparaciones de coordenadas.
Eso no significa que todos los escenarios aprueben. Máximos entre escenarios de cada tipo:

{
        table(
            [
                "Cambio",
                "Método",
                "Peor p95 de Δ absoluto",
                "Máximo Δ absoluto",
                "Cambio de categoría (%)",
                "Extremos opuestos (%)",
                "N mínimo",
                "N máximo",
            ],
            stress_rows,
        )
    }

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

{
        table(
            [
                "Perfil",
                "Etiquetas estables",
                "Todos los meses",
                "Historia común completa",
                "Cobertura (%)",
                "p95 de amplitud",
            ],
            env_rows,
        )
    }

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

{REFERENCES}
"""
    (output / "verdict.md").write_text(output_report.rstrip() + "\n", encoding="utf-8")
    return selected
