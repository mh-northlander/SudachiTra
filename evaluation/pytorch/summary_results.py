import argparse as ap
import itertools as it
import json
import logging
import sys
from collections import defaultdict as ddict
from enum import Enum
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger.setLevel(logging.INFO)


class Stages (Enum):
    VALIDATION = "validation"
    TEST = "test"


def summary_amazon(args):
    results = {}
    for subdir in args.input_dir.glob("*"):
        if not subdir.is_dir():
            continue
        results[subdir.name] = {}
        for stage in Stages:
            results[subdir.name][stage] = {}
            with (subdir / f"{stage.value}_predictions.tsv").open() as f:
                f.readline()  # skip headerline
                ids, labels, preds = zip(*(line.strip().split("\t")
                                         for line in f.readlines()))

            num_samples = len(ids)
            labels = [int(v[6:] if v.startswith("LABEL_") else v)
                      for v in labels]
            preds = [int(v[6:] if v.startswith("LABEL_") else v)
                     for v in preds]
            results[subdir.name][stage]["acc"] = sum(
                l == p for l, p in zip(labels, preds)) / num_samples
            results[subdir.name][stage]["mse"] = sum(
                (l-p)**2 for l, p in zip(labels, preds)) / num_samples
            results[subdir.name][stage]["mae"] = sum(
                abs(l-p) for l, p in zip(labels, preds)) / num_samples

    log_best_model(results, key=lambda k: results[k][Stages.VALIDATION]["acc"])

    df = pd.DataFrame(
        data=((hp,
               ret[Stages.VALIDATION]["acc"], ret[Stages.VALIDATION]["mse"], ret[Stages.VALIDATION]["mae"],
               ret[Stages.TEST]["acc"], ret[Stages.TEST]["mse"], ret[Stages.TEST]["mae"],)
              for hp, ret in results.items()),
        columns=["Parameter", "Dev Acc", "Dev MSE", "Dev MAE", "Test Acc", "Test MSE", "Test MAE"])
    df.to_csv(args.output_file)
    return


def summary_torch_metrics(args):
    results = {}
    for subdir in args.input_dir.glob("*"):
        if not subdir.is_dir():
            continue
        results[subdir.name] = {}
        for stage in Stages:
            with (subdir / f"{stage.value}_results.json").open() as f:
                metrics = json.load(f)
            results[subdir.name][stage] = metrics

    clms = args.columns
    csv_columns = ["Parameter"] + [f"Dev {c}" for c in clms] + [f"Test {c}" for c in clms]

    def ret2summary(hp, ret):
        return tuple(it.chain([hp], [
            ret[stage][f"{stage.value}_{c}"] if (f"{stage.value}_{c}" in ret[stage]) else ret[stage][f"test_{c}"]
            for stage in Stages for c in clms
        ]))

    df = pd.DataFrame(
        data=(ret2summary(hp, ret) for hp, ret in results.items()),
        columns=csv_columns)
    df.to_csv(args.output_file)
    return


def log_best_model(results, key):
    best_model = max(results, key=key)
    logger.info(f"best model: {best_model}")
    logger.info(f"result: {results[best_model]}")
    return best_model


SUMMARY_METRICS = {
    "kuci": ["accuracy"],
    "rcqa": ["exact", "f1"],
    # jglue
    "marc-ja": ["accuracy"],
    "jsts": ["pearson", "spearmanr"],
    "jnli": ["accuracy"], 
    "jcommonsenseqa": ["accuracy"],
    "jsquad": ["exact", "f1"],
}
SUMMARY_FUNCS = {k: summary_torch_metrics for k in SUMMARY_METRICS.keys()}

SUMMARY_FUNCS.update({"amazon": summary_amazon})


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(dest="dataset_name", type=str,
                        help="Target dataset name. Set \"list\" to list available datasets.")
    parser.add_argument(dest="input_dir", type=str,
                        help="Input directory. output_dir of run_evaluation.py.")

    parser.add_argument("-o", "--output", dest="output_file", type=str, default="./output.csv",
                        help="File to output summary. `output.csv` by default.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output files when they already exist.")

    args = parser.parse_args()
    args.dataset_name = args.dataset_name.lower()
    args.input_dir = Path(args.input_dir)
    args.output_file = Path(args.output_file)
    return args


def validate_args(args):
    if args.dataset_name not in SUMMARY_FUNCS and args.dataset_name != "list":
        logger.error(f"Unknown dataset name ({args.dataset_name}). "
                     f"It must be one of {list(SUMMARY_FUNCS.keys())} or \"list\".")
        raise ValueError

    if not args.input_dir.is_dir():
        raise ValueError("input should be directory.")

    if not args.overwrite:
        if args.output_file.exists():
            raise ValueError(
                f"File {args.output_file} already exists. Set --overwrite to continue anyway.")
    return


def main():
    args = parse_args()
    validate_args(args)

    if args.dataset_name == "list":
        logger.info(f"Available datasets: {list(SUMMARY_FUNCS.keys())}")
        return

    if args.dataset_name in SUMMARY_METRICS:
        args.columns = SUMMARY_METRICS[args.dataset_name]

    logger.info(f"input_dir: {args.input_dir}")

    summary_func = SUMMARY_FUNCS[args.dataset_name]
    summary_func(args)
    return


if __name__ == "__main__":
    main()
