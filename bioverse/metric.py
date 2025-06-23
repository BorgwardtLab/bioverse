from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import awkward as ak
import numpy as np
from rich.table import Table

from .utilities import console, flatten


class Metric(ABC):
    better = "higher"

    def __init__(
        self,
        property: str | None = None,
        name: str = "Metric",
        reduction: str | None = "mean",
        on: int = 1,
        per: int | None = None,
    ):
        self.name = name if property is None else f"{name} ({property})"
        self.property = property
        self.reduce = getattr(np, reduction) if reduction is not None else lambda x: x
        self.per = per
        self.on = on
        self.values = []
        self.y_true = []
        self.y_pred = []

    def before_compute(
        self, y_true: ak.Array, y_pred: ak.Array
    ) -> tuple[ak.Array, ak.Array]:
        return y_true, y_pred

    @abstractmethod
    def compute(self, y_true: ak.Array, y_pred: ak.Array) -> float:
        raise NotImplementedError

    def __call__(self, y_true: ak.Array, y_pred: ak.Array) -> Result:
        self.update(y_true, y_pred)
        return self.result()

    def update(self, y_true: ak.Array, y_pred: ak.Array) -> None:
        y_true, y_pred = self.before_compute(y_true, y_pred)
        property = self.property or y_true.fields[0]
        y_true, y_pred = y_true[property], y_pred[property]  # type: ignore
        y_true = flatten(y_true, axes=slice(None, self.on), exclude=self.per)
        y_pred = flatten(y_pred, axes=slice(None, self.on), exclude=self.per)
        if self.per is None:
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)
        else:
            self.values.append(self.compute(y_true, y_pred))

    def result(self, model_name: str = "Ours") -> Result:
        if self.per is None:
            y_true = ak.concatenate(self.y_true)
            y_pred = ak.concatenate(self.y_pred)
            if self.on == 0:
                y_true = y_true[None]
                y_pred = y_pred[None]
            values = [self.compute(y_true, y_pred)]
        else:
            values = ak.Array(self.values)
            if values.ndim > 1:
                values = ak.concatenate(values)
        result = Result(
            [
                {
                    "Model": model_name,
                    "Metric": self.name,
                    "Value": self.reduce(values),
                }
            ],
            {self.name: self.better},
        )
        self.reset()
        return result

    def reset(self) -> None:
        self.values = []
        self.y_true = []
        self.y_pred = []

    def __add__(self, other: Metric | MultiMetric) -> Metric | MultiMetric:
        if isinstance(other, MultiMetric):
            return other + self
        return MultiMetric() + self + other


class MultiMetric:

    def __init__(self, metrics=[]) -> None:
        self.metrics = metrics
        self.better = {**{m.name: m.better for m in metrics}}

    def update(self, y_true: ak.Array, y_pred: ak.Array) -> None:
        for metric in self.metrics:
            metric.update(y_true, y_pred)

    def result(self, model_name: str = "Yours") -> Result:
        result = Result()
        for metric in self.metrics:
            result += metric.result(model_name)
        return result

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def __add__(self, other: Metric | MultiMetric) -> Metric | MultiMetric:
        if isinstance(other, MultiMetric):
            self.metrics.extend(other.metrics)
            self.better.update(other.better)
        else:
            self.metrics.append(other)
            self.better.update(other.better)
        return self


class Result:

    def __init__(self, data=[], better={}) -> None:
        self.data = data
        self.better = better

    def __add__(self, other: Result) -> Result:
        self.data += other.data
        self.better.update(other.better)
        return self

    def format(
        self,
        sort_by: str = "Model",
        reverse_order: bool = False,
        separate: list[str] = [],
        num_entries: int | None = None,
    ):
        wide_format_dict = defaultdict(lambda: defaultdict(list))
        for entry in self.data:
            model, metric, value = entry["Model"], entry["Metric"], entry["Value"]
            wide_format_dict[model]["Model"] = model
            wide_format_dict[model][metric].append(value)
        data = [dict(v) for v in list(wide_format_dict.values())]
        for entry in data:
            for metric, values in entry.items():
                if metric != "Model":
                    if isinstance(values[0], tuple) or isinstance(
                        values[0], list
                    ):  # values are given as (mean, sd)
                        entry[metric] = (values[0][0], values[0][1])  # type: ignore
                    else:
                        entry[metric] = (np.mean(values), np.std(values))  # type: ignore
        best = {}
        for entry in data:
            for metric, value in entry.items():
                if metric != "Model":
                    mean, std = value
                    if metric not in best:
                        best[metric] = mean
                    elif self.better[metric] == "higher" and mean > best[metric]:
                        best[metric] = mean
                    elif self.better[metric] == "lower" and mean < best[metric]:
                        best[metric] = mean
        for entry in data:
            for metric, value in entry.items():
                if metric != "Model":
                    mean, std = value
                    if mean == best[metric]:
                        entry[metric] = (mean, std, True)
                    else:
                        entry[metric] = (mean, std, False)
        if sort_by == "Model":
            data.sort(key=lambda x: x["Model"])
        else:
            data.sort(key=lambda x: x[sort_by][0])
            if self.better[sort_by] == "higher":
                data.reverse()
        if reverse_order:
            data.reverse()
        formatted = [[], []]
        for entry in data:
            if entry["Model"] in separate:
                formatted[1].append(entry)
            else:
                formatted[0].append(entry)
        if num_entries is not None:
            formatted[0] = formatted[0][:num_entries]
        return formatted

    def format_row(
        self, row, metrics, decimals, boldify, show_std, bold_open, bold_close
    ):
        if show_std:
            values = [
                (
                    f"{bold_open}{row[m][0]:.{decimals}f} {chr(0x00B1)} {row[m][1]:.{decimals}f}{bold_close}"
                    if row[m][2] and boldify
                    else f"{row[m][0]:.{decimals}f} {chr(0x00B1)} {row[m][1]:.{decimals}f}"
                )
                for m in metrics
            ]
        else:
            values = [
                (
                    f"{bold_open}{row[m][0]:.{decimals}f}{bold_close}"
                    if row[m][2] and boldify
                    else f"{row[m][0]:.{decimals}f}"
                )
                for m in metrics
            ]
        return values

    def to_string(self) -> str:
        return "String"  # prettytable

    def to_console(
        self,
        sort_by: str = "Model",
        reverse_order: bool = False,
        separate: list[str] = [],
        num_entries: int | None = None,
        justify_model: str = "left",
        justify_metrics: str = "center",
        decimals: int = 4,
        boldify: bool = True,
        show_std: bool = False,
    ) -> None:
        metrics = set([row["Metric"] for row in self.data])
        main, footer = self.format(
            sort_by=sort_by,
            reverse_order=reverse_order,
            separate=separate,
            num_entries=num_entries,
        )
        table = Table(title="Results")
        table.add_column("Model", justify=justify_model, style="cyan", no_wrap=True)  # type: ignore
        for metric in metrics:
            table.add_column(metric, justify=justify_metrics, style="#666666")  # type: ignore
        for row in main:
            values = self.format_row(
                row,
                metrics,
                decimals,
                boldify,
                show_std,
                bold_open="[bold]",
                bold_close="[/bold]",
            )
            table.add_row(row["Model"], *values)
        if len(footer) > 0:
            table.add_section()
            for row in footer:
                values = self.format_row(
                    row,
                    metrics,
                    decimals,
                    boldify,
                    show_std,
                    bold_open="[bold]",
                    bold_close="[/bold]",
                )
                table.add_row(row["Model"], *values)
        console.print(table)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        models = set([row["Model"] for row in self.data])
        return {
            model: {
                item["Metric"]: item["Value"]
                for item in self.data
                if item["Model"] == model
            }
            for model in models
        }

    def to_csv(self) -> str:
        raise NotImplementedError

    def to_latex(
        self,
        sort_by: str = "Model",
        reverse_order: bool = False,
        separate: list[str] = [],
        num_entries: int | None = None,
        justify_model: str = "left",
        justify_metrics: str = "center",
        decimals: int = 4,
        boldify: bool = True,
        show_std: bool = False,
    ) -> str:
        metrics = set([row["Metric"] for row in self.data])
        main, footer = self.format(
            sort_by=sort_by,
            reverse_order=reverse_order,
            separate=separate,
            num_entries=num_entries,
        )

        latex = (
            "\\begin{table}[h]\n\\centering\n\\begin{tabular}{"
            + justify_model[0]
            + justify_metrics[0] * len(metrics)
            + "}\n"
        )
        latex += "\\toprule\n"
        header = ["Model"] + list(metrics)
        latex += " & ".join(header) + " \\\\\n"
        latex += "\\midrule\n"
        for row in main:
            values = self.format_row(
                row,
                metrics,
                decimals,
                boldify,
                show_std,
                bold_open="\\textbf{",
                bold_close="}",
            )
            latex += " & ".join([row["Model"]] + values) + " \\\\\n"
        if len(footer) > 0:
            latex += "\\midrule\n"
            for row in footer:
                values = self.format_row(
                    row,
                    metrics,
                    decimals,
                    boldify,
                    show_std,
                    bold_open="\\textbf{",
                    bold_close="}",
                )
                latex += " & ".join([row["Model"]] + values) + " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n"
        latex += "\\caption{Results}\n\\end{table}"

        return latex

    def __str__(self) -> str:
        return self.to_string()
