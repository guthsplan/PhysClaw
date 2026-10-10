"""QuantiPhy benchmark loader for SpatialClaw's CoT baseline entrypoint.

This is a small custom adapter written to feed a handful of QuantiPhy
(https://quantiphy.stanford.edu/) open-ended physical-quantity-estimation
problems through SpatialClaw's evaluation harness. QuantiPhy is not one of
the 20 benchmarks that ship with the upstream repo, so this mirrors the
existing loaders (e.g. evals/dsibench.py) but scores with mean relative
accuracy (MRA) over the official QuantiPhy threshold set (0.1..0.95 step
0.1, matching quantiphy's own evaluator and code-as-world's evaluation.py)
instead of exact-match multiple choice.

Data layout expected under data_path (default: data/QuantiPhy/):
    problems.csv   - columns: problem_id, video_id, question, prior, answer,
                      quantity, reasoning_route (subset of quantiphy_taxonomy.csv)
    videos/*.mp4   - one file per video_id referenced in problems.csv
"""

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.config import get_config
from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_relative_accuracy,
    write_json,
    write_results_summary,
)

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Match the official QuantiPhy / code-as-world evaluation.py threshold set,
# not SpatialClaw's own VSI-Bench-style default (0.5..0.95 step 0.05), so
# scores are comparable across both repos.
QUANTIPHY_MRA_KWARGS = {"start": 0.1, "end": 0.95, "interval": 0.1}


def _first_float(text: Any) -> Optional[float]:
    if not text:
        return None
    m = _NUMBER_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


@dataclass
class QuantiPhySample(LazyVideoSample):
    prior: str = ""
    quantity: str = ""
    reasoning_route: str = ""


class QuantiPhyBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """QuantiPhy loader: open-ended numeric physical-quantity estimation from video."""

    data_specific_prompt = (
        "This is a quantitative physical reasoning question about the video. "
        "Use the given prior (a known reference measurement) to estimate the "
        "requested physical quantity as precisely as you can. "
        "Respond with ONLY a number and its unit, e.g. \\boxed{2.5 cm}."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        csv_path = os.path.join(self.data_path, "problems.csv")
        video_dir = os.path.join(self.data_path, "videos")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"QuantiPhy problems csv not found: {csv_path}")

        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            video_id = row["video_id"]
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            question = row["question"]
            prior = row.get("prior", "") or ""
            quantity = row.get("quantity", "") or "quantiphy"
            full_question = f"Given that {prior}. {question}" if prior else question

            if self.question_type_filter and quantity not in self.question_type_filter:
                continue

            self.data.append(
                QuantiPhySample(
                    sample_id=f"quantiphy_{row['problem_id']}",
                    question=full_question,
                    question_type=quantity,
                    images=[],
                    answer=str(row["answer"]),
                    video=video_path,
                    prior=prior,
                    quantity=quantity,
                    reasoning_route=row.get("reasoning_route", "") or "",
                    _bench_ref=self,
                )
            )

    def extract_answer(self, prediction: str) -> str:
        val = _first_float(prediction)
        return "" if val is None else str(val)

    def evaluate_single(self, sample: QuantiPhySample, prediction: str) -> float:
        pred_val = _first_float(prediction)
        gt_val = _first_float(sample.answer)
        if pred_val is None or gt_val is None:
            return 0.0
        return mean_relative_accuracy(pred_val, gt_val, **QUANTIPHY_MRA_KWARGS)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_quantity: Dict[str, List[float]] = {}
        details = []
        n_failed = 0

        for sample in self.data:
            sid = sample.sample_id
            raw_pred = get_prediction(predictions, sid)
            pred_val = _first_float(raw_pred)
            if pred_val is None:
                n_failed += 1
            score = self.evaluate_single(sample, raw_pred)
            per_quantity.setdefault(sample.quantity, []).append(score)
            details.append(
                {
                    "sample_id": sid,
                    "video_id": os.path.splitext(os.path.basename(sample.video))[0],
                    "question": sample.question,
                    "ground_truth": sample.answer,
                    "prediction": raw_pred,
                    "parsed_value": pred_val,
                    "mra": score,
                }
            )

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        results = {
            "total_samples": len(details),
            "failed_extractions": n_failed,
            "overall_mra": _mean([d["mra"] for d in details]),
            "per_quantity_mra": {k: _mean(v) for k, v in per_quantity.items()},
            "per_quantity_counts": {k: len(v) for k, v in per_quantity.items()},
            "detailed_results": details,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), details)

        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("QuantiPhy Results (MRA, official 0.1-0.95 thresholds)")
        print(f"{'='*60}")
        print(f"Total: {results['total_samples']}  Failed extractions: {results.get('failed_extractions', 0)}")
        print(f"Overall MRA: {results['overall_mra']:.4f}")
        for k, v in results.get("per_quantity_mra", {}).items():
            print(f"  {k}: {v:.4f} (n={results['per_quantity_counts'][k]})")
        print(f"{'='*60}\n")
