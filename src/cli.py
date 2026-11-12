"""Command line entry point: ``uv run pawmatch [--lang en|ja] <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.i18n import SUPPORTED, get_lang, t


def _prepare(args: argparse.Namespace) -> None:
    from src.data.oxford_pet import DATASET_DIR, download, parse_list, with_hashes
    from src.data.split import check_no_leakage, drop_duplicates, stratified_split, write_split_csv

    root = Path(args.data_root)
    if args.source == "hf":
        from src.data import hf_mirror

        hf_mirror.download(root / "hf")
        hf_mirror.convert(root)
    else:
        download(root)
    base = root / DATASET_DIR
    entries, _ = parse_list(base / "annotations" / "list.txt")
    entries = with_hashes(entries, base / "images")
    entries, removed = drop_duplicates(entries)
    print(t("split.dup", args.lang, n=removed))
    splits = stratified_split(entries, seed=args.seed)
    check_no_leakage(splits)
    out = Path(args.out)
    write_split_csv(splits, out)
    print(t("split.done", args.lang, path=out, train=len(splits["train"]),
            val=len(splits["val"]), test=len(splits["test"])))


def _train(args: argparse.Namespace) -> None:
    from src.models.train_classifier import run

    run(Path(args.config), args.lang)


def _report(args: argparse.Namespace) -> None:
    m = json.loads((Path(args.run_dir) / "metrics.json").read_text())
    print(t("eval.header", args.lang))
    for key, label in (("species", "eval.species"), ("breed", "eval.breed")):
        print(f"- {t(label, args.lang)}: {t('eval.accuracy', args.lang)}="
              f"{m[key]['accuracy']:.4f}, {t('eval.macro_f1', args.lang)}={m[key]['macro_f1']:.4f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pawmatch")
    p.add_argument("--lang", choices=SUPPORTED, default=get_lang())
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare-data", help=t("cli.prepare"))
    sp.add_argument("--data-root", default="data")
    sp.add_argument("--out", default="data/splits/oxford_pet_seed42.csv")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--source", choices=["official", "hf"], default="official",
                    help="official (slow outside UK) or Hugging Face mirror")
    sp.set_defaults(fn=_prepare)

    sp = sub.add_parser("train", help=t("cli.train"))
    sp.add_argument("--config", default="configs/classification.yaml")
    sp.set_defaults(fn=_train)

    sp = sub.add_parser("report", help=t("cli.evaluate"))
    sp.add_argument("run_dir")
    sp.set_defaults(fn=_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
