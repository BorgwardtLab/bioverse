"""Discover Bioverse implementations for Sphinx documentation."""

from __future__ import annotations

import importlib
import importlib.resources
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import yaml

from bioverse.utilities import load
from bs4 import BeautifulSoup
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

PYTHON_IMPLEMENTATIONS: dict[str, tuple[str, str]] = {
    "adapters": ("bioverse.adapters", "bioverse.adapter.Adapter"),
    "processors": ("bioverse.processors", "bioverse.processor.Processor"),
    "metrics": ("bioverse.metrics", "bioverse.metric.Metric"),
    "tasks": ("bioverse.tasks", "bioverse.task.Task"),
    "samplers": ("bioverse.samplers", "bioverse.sampler.Sampler"),
    "transforms": ("bioverse.transforms", "bioverse.transform.Transform"),
}

YAML_IMPLEMENTATIONS = ("datasets", "benchmarks")
DOC_METADATA_KEYS = ("description", "citation")


def normalize_citations(citation: Any) -> list[str] | None:
    """Normalize ``citation`` YAML values to a list of non-empty strings."""
    if citation is None:
        return None
    if isinstance(citation, str):
        citation = citation.strip()
        return [citation] if citation else None
    if isinstance(citation, list):
        citations = [str(entry).strip() for entry in citation if str(entry).strip()]
        return citations or None
    raise ValueError(
        f"Invalid citation value {citation!r}. Expected a string or list of strings."
    )


def config_preview_yaml(config: dict[str, Any]) -> str:
    """Return YAML for docs preview, excluding documentation-only keys."""
    preview = {
        key: value for key, value in config.items() if key not in DOC_METADATA_KEYS
    }
    if not preview:
        return ""
    return yaml.dump(preview, sort_keys=False, allow_unicode=True).rstrip()


def _resolve_base_class(base_path: str) -> type:
    module_name, _, class_name = base_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def class_summary(qualname: str) -> str:
    """First paragraph of a class docstring for implementation page intros."""
    module_name, _, class_name = qualname.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception:
        return ""

    doc = inspect.getdoc(cls) or ""
    if not doc:
        return ""

    paragraph = doc.split("\n\n")[0]
    return re.sub(r"\s+", " ", paragraph).strip()


def _yaml_component_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(next(iter(value)))
    if isinstance(value, list):
        names = [_yaml_component_name(item) for item in value]
        return ", ".join(names)
    return str(value)


def summarize_yaml_config(submodule: str, config: dict[str, Any]) -> str:
    """Fallback summary when a YAML file has no ``description`` key."""
    if submodule == "benchmarks":
        dataset = config.get("dataset", "?")
        sampler = _yaml_component_name(config.get("sampler", "?"))
        task = _yaml_component_name(config.get("task", "?"))
        metric = _yaml_component_name(config.get("metric", "?"))
        return (
            f"Benchmark on dataset ``{dataset}`` with sampler ``{sampler}``, "
            f"task ``{task}``, and metric ``{metric}``."
        )
    if submodule == "datasets":
        if "adapter" in config:
            adapter = _yaml_component_name(config["adapter"])
            return f"Dataset built from adapter ``{adapter}``."
        if "parent" in config:
            parent = config["parent"]
            if isinstance(parent, list):
                parent = ", ".join(str(item) for item in parent)
            return f"Dataset derived from ``{parent}``."
        if "transforms" in config:
            return "Dataset defined by a transform pipeline."
    return ""


def discover_classes(package_name: str, base_class: type) -> list[dict[str, str]]:
    """Return public classes with docstring summaries."""
    package = importlib.import_module(package_name)
    discovered: dict[str, str] = {}

    for _finder, module_name, _ispkg in sorted(
        pkgutil.iter_modules(package.__path__), key=lambda item: item[1]
    ):
        if module_name.startswith("_"):
            continue

        module_path = f"{package_name}.{module_name}"
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, base_class)
                and obj is not base_class
            ):
                qualname = f"{module.__name__}.{name}"
                discovered[qualname] = class_summary(qualname)

    return [
        {"qualname": qualname, "summary": summary}
        for qualname, summary in sorted(discovered.items())
    ]


def discover_yaml_configs(submodule: str) -> list[dict[str, Any]]:
    """Return sorted YAML configuration metadata for datasets or benchmarks."""
    root = importlib.resources.files(f"bioverse.{submodule}")
    configs: list[dict[str, Any]] = []

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not str(path.name).endswith(".yaml"):
            continue

        with importlib.resources.as_file(path) as file_path:
            config = load(file_path)
            description = config.get("description") or summarize_yaml_config(
                submodule, config
            )
            citations = normalize_citations(config.get("citation"))
            configs.append(
                {
                    "name": Path(path.name).stem,
                    "filename": path.name,
                    "config": config,
                    "description": description,
                    "citations": citations,
                    "config_preview": config_preview_yaml(config),
                }
            )

    return configs


def build_docs_context() -> dict[str, Any]:
    """Build the Jinja context used by implementation reference pages."""
    context: dict[str, Any] = {}

    for key, (package_name, base_path) in PYTHON_IMPLEMENTATIONS.items():
        base_class = _resolve_base_class(base_path)
        context[key] = {
            "module": package_name,
            "classes": discover_classes(package_name, base_class),
        }

    for key in YAML_IMPLEMENTATIONS:
        context[key] = {"configs": discover_yaml_configs(key)}

    return context


PYTHON_IMPLEMENTATION_PAGES = {
    "adapters",
    "processors",
    "metrics",
    "tasks",
    "samplers",
    "transforms",
}


def flatten_implementation_toc(app: Sphinx, exc: Exception | None) -> None:
    """Show only class signature chips in the implementations page TOC."""
    if exc is not None or app.builder is None or app.builder.name != "html":
        return

    implementations_dir = Path(app.outdir) / "implementations"
    if not implementations_dir.is_dir():
        return

    intersect_attr = "x-intersect.margin.0%.0%.-70%.0%"

    for html_file in implementations_dir.glob("*.html"):
        if html_file.stem not in PYTHON_IMPLEMENTATION_PAGES:
            continue

        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
        sidebar_list = soup.select_one("#right-sidebar ul")
        content = soup.select_one("#content")
        if sidebar_list is None or content is None:
            continue

        for headerlink in content.select("a.headerlink"):
            parent_dt = headerlink.find_parent("dt", class_="sig")
            if parent_dt is None:
                continue

            parent_dl = parent_dt.find_parent("dl")
            property_tag = parent_dt.select_one("em.property > span.pre")
            is_class_signature = (
                parent_dl is not None
                and "py" in parent_dl.get("class", [])
                and "class" in parent_dl.get("class", [])
                and parent_dl.find("dt", recursive=False) is parent_dt
                and property_tag is not None
                and property_tag.get_text(strip=True) == "class"
            )
            if not is_class_signature and intersect_attr in headerlink.attrs:
                del headerlink.attrs[intersect_attr]

        sidebar_list.clear()
        for class_node in content.select("dl.py.class > dt.sig.sig-object.py"):
            property_tag = class_node.select_one("em.property > span.pre")
            if property_tag is None or property_tag.get_text(strip=True) != "class":
                continue

            name_span = class_node.select_one(".sig-name.descname")
            headerlink = class_node.select_one("a.headerlink")
            if name_span is None or headerlink is None or not headerlink.get("href"):
                continue

            list_item = soup.new_tag("li")
            anchor = soup.new_tag("a", href=headerlink["href"])
            anchor["class"] = "reference internal"
            anchor[":data-current"] = f"activeSection === '{headerlink['href']}'"

            code = soup.new_tag("code")
            code["class"] = "docutils literal notranslate"
            code_span = soup.new_tag("span")
            code_span["class"] = "pre"
            code_span.string = name_span.get_text()
            code.append(code_span)
            anchor.append(code)
            list_item.append(anchor)
            sidebar_list.append(list_item)

        html_file.write_text(str(soup), encoding="utf-8")
