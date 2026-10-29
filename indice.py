"""Punto de entrada del índice y del explorador BDE de MyM Ingenierías."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from bcch import SeriesCatalog
from bcch.exceptions import BCCHError

from imc.config import Methodology
from imc.data import DataError, configured_client, fetch_series, observations_frame
from imc.engine import calculate_index, latest_complete
from imc.report import export_run


def iso_date(value):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Usa una fecha válida YYYY-MM-DD.") from None


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    calc = sub.add_parser("calcular", help="Calcular y exportar el índice descriptivo v2")
    calc.add_argument("--inicio", type=iso_date, default="2012-01-01")
    calc.add_argument("--fin", type=iso_date, default=date.today().isoformat())
    calc.add_argument("--salida", type=Path, default=Path("outputs/indice"))
    calc.add_argument("--env-file", type=Path, default=Path(".env"))
    calc.add_argument("--ventana", type=int, default=504)
    calc.add_argument("--min-historia", type=int, default=252)
    calc.add_argument("--sin-graficos", action="store_true")
    catalog = sub.add_parser("catalogo", help="Explorar el catálogo local sin token ni red")
    catalog.add_argument("texto", nargs="?", default="")
    catalog.add_argument("--frecuencia", choices=["diaria", "mensual", "trimestral", "anual"])
    catalog.add_argument("--capitulo")
    catalog.add_argument("--fuente")
    catalog.add_argument("--estado", default="Vigente")
    catalog.add_argument("--limite", type=int, default=20)
    catalog.add_argument("--salida", type=Path)
    single = sub.add_parser("serie", help="Descargar cualquier serie BDE en su frecuencia original")
    single.add_argument("codigo")
    single.add_argument("--inicio", type=iso_date, default="2012-01-01")
    single.add_argument("--fin", type=iso_date, default=date.today().isoformat())
    single.add_argument("--salida", type=Path, default=Path("outputs/serie.csv"))
    single.add_argument("--env-file", type=Path, default=Path(".env"))
    return cli


def explore(args):
    if args.limite < 1:
        raise DataError("El límite debe ser positivo.")
    catalog = SeriesCatalog()
    matches = catalog.search(
        args.texto,
        frequency=args.frecuencia,
        chapter=args.capitulo,
        source=args.fuente,
        status=args.estado,
        limit=args.limite,
    )
    payload = {
        "catalog": catalog.metadata,
        "results": [
            {
                "code": item.code,
                "contexts": [
                    {
                        "name": ctx.name,
                        "chapter": ctx.chapter,
                        "table": ctx.table,
                        "frequency": ctx.frequency,
                        "unit": ctx.unit,
                        "source": ctx.source,
                        "status": ctx.status,
                        "first_observation": (
                            ctx.first_value_date.isoformat() if ctx.first_value_date else None
                        ),
                    }
                    for ctx in item.contexts
                ],
            }
            for item in matches
        ],
    }
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "catalogo":
            explore(args)
            return 0
        if args.inicio > args.fin:
            raise DataError("La fecha de inicio no puede ser posterior al fin.")
        if args.command == "serie":
            with configured_client(args.env_file) as client:
                result = client.get_series(args.codigo, start=args.inicio, end=args.fin)
            frame = observations_frame(result, code=args.codigo, alias="serie")
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            frame.loc[args.inicio : args.fin].to_csv(args.salida, date_format="%Y-%m-%d")
            print(f"Serie exportada: {args.salida}")
            return 0
        config = Methodology(normalization_window=args.ventana, min_history=args.min_historia)
        with configured_client(args.env_file) as client:
            frames, metadata = fetch_series(client, start=args.inicio, end=args.fin)
        calculated = calculate_index(frames, config)
        latest, unavailable = latest_complete(calculated, as_of=args.fin)
        export_run(
            calculated,
            frames,
            metadata,
            config,
            output=args.salida,
            start=args.inicio,
            end=args.fin,
            plot=not args.sin_graficos,
        )
        print(f"Índice v2: {latest['index']:.2f}/100 | Fecha: {latest.name.date()}")
        print(f"Observación más antigua utilizada: {latest['oldest_input_date'].date()}")
        if unavailable:
            print("La última fecha del calendario no está completa: " + ", ".join(unavailable))
        print(f"Resultados y trazabilidad: {args.salida}")
        return 0
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
    except BCCHError:
        print(
            "Error: el BCCh no pudo entregar la serie. Revisa código, acceso y conexión.",
            file=sys.stderr,
        )
    except (ValueError, OSError):
        print(
            "Error: configuración o archivo inválido. Revisa argumentos y permisos.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
