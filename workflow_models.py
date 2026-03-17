"""
workflow_models.py
------------------
Extract all model file references (name, download URL, target directory) from
a ComfyUI workflow JSON, and optionally download them.

Supported formats
-----------------
UI format   ("workflow.json")
    Top-level keys: id, nodes[], links[], ...
    Model metadata lives in node.properties.models[]

API format  ("workflow_api.json")
    Top-level keys are integer-string node IDs: {"4": {"class_type": ..., "inputs": {...}}}
    No URL metadata is stored; only the model *filename* is available.

Subgraph / node-group formats
    definitions.subgraphs[].nodes[]   – modern inline subgraph nodes
    extra.groupNodes.{uuid}.nodes[]   – legacy group node format

Usage
-----
    from workflow_models import get_workflow_models, download_workflow_models

    with open("my_workflow.json") as f:
        workflow = json.load(f)

    models = get_workflow_models(workflow)
    results = download_workflow_models(workflow, models_dir="/path/to/ComfyUI/models")
    for r in results:
        print(r.status, r.dest)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelRef:
    """A single model file referenced by a workflow node."""

    name: str
    """Filename as stored on disk (e.g. 'ltx-video-2b-v0.9.safetensors')."""

    directory: str
    """ComfyUI folder-type key (e.g. 'checkpoints', 'loras', 'text_encoders').
    Maps directly to the keys in folder_paths.folder_names_and_paths."""

    url: str | None
    """Direct download URL embedded in the workflow, or None if absent.
    May be a HuggingFace resolve URL or a CivitAI model-version URL."""

    node_type: str
    """The class_type / node type that references this model
    (e.g. 'CheckpointLoaderSimple')."""

    node_id: int | str
    """The node's id field (int in UI format, string in API format)."""

    source: str
    """Where in the JSON this ref was found:
    'nodes', 'subgraph:<uuid>', or 'groupNode:<uuid>'."""

    def __str__(self) -> str:
        loc = f"{self.directory}/{self.name}"
        return f"[{self.node_type}] {loc}" + (f" <- {self.url}" if self.url else "")


@dataclass
class DownloadResult:
    """Outcome of a single model download attempt."""

    model: ModelRef
    dest: Path
    status: str          # "downloaded", "resumed", "skipped", "no_url", "error"
    error: str | None = None

    def __str__(self) -> str:
        if self.status == "error":
            return f"ERROR   {self.dest.name}: {self.error}"
        if self.status == "skipped":
            return f"SKIP    {self.dest}"
        if self.status == "no_url":
            return f"NO URL  {self.model.directory}/{self.model.name}"
        verb = "GET" if self.status == "downloaded" else "RESUME"
        return f"{verb}    {self.dest}"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_models_from_node(node: dict[str, Any], source: str) -> list[ModelRef]:
    refs: list[ModelRef] = []
    node_type = node.get("type") or node.get("class_type") or "<unknown>"
    node_id = node.get("id", "<unknown>")
    props = node.get("properties") or {}
    models = props.get("models")
    if not isinstance(models, list):
        return refs
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        directory = entry.get("directory")
        if not name or not directory:
            continue
        refs.append(ModelRef(
            name=name,
            directory=directory,
            url=entry.get("url") or None,
            node_type=node_type,
            node_id=node_id,
            source=source,
        ))
    return refs


def _collect_nodes_ui(workflow: dict[str, Any]) -> list[ModelRef]:
    refs: list[ModelRef] = []
    for node in workflow.get("nodes") or []:
        refs.extend(_extract_models_from_node(node, source="nodes"))
    definitions = workflow.get("definitions") or {}
    for subgraph in definitions.get("subgraphs") or []:
        sg_id = subgraph.get("id", "<unknown>")
        for node in subgraph.get("nodes") or []:
            refs.extend(_extract_models_from_node(node, source=f"subgraph:{sg_id}"))
    extra = workflow.get("extra") or {}
    for uuid, group in (extra.get("groupNodes") or {}).items():
        for node in (group.get("nodes") or []):
            refs.extend(_extract_models_from_node(node, source=f"groupNode:{uuid}"))
    return refs


def _collect_nodes_api(workflow: dict[str, Any]) -> list[ModelRef]:
    LOADER_TO_DIRECTORY: dict[str, list[str]] = {
        "CheckpointLoaderSimple":   ["checkpoints"],
        "CheckpointLoader":         ["checkpoints"],
        "unCLIPCheckpointLoader":   ["checkpoints"],
        "LoraLoader":               ["loras"],
        "LoraLoaderModelOnly":      ["loras"],
        "VAELoader":                ["vae"],
        "ControlNetLoader":         ["controlnet"],
        "DiffControlNetLoader":     ["controlnet"],
        "UNETLoader":               ["diffusion_models"],
        "CLIPLoader":               ["text_encoders"],
        "DualCLIPLoader":           ["text_encoders", "text_encoders"],
        "TripleCLIPLoader":         ["text_encoders", "text_encoders", "text_encoders"],
        "CLIPVisionLoader":         ["clip_vision"],
        "StyleModelLoader":         ["style_models"],
        "GLIGENLoader":             ["gligen"],
        "DiffusersLoader":          ["diffusers"],
        "UpscaleModelLoader":       ["upscale_models"],
        "HypernetworkLoader":       ["hypernetworks"],
        "PhotoMakerLoader":         ["photomaker"],
    }
    refs: list[ModelRef] = []
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs") or {}
        directories = LOADER_TO_DIRECTORY.get(class_type)
        if not directories:
            continue
        filenames = [v for v in inputs.values() if isinstance(v, str)]
        for directory, filename in zip(directories, filenames):
            refs.append(ModelRef(
                name=filename, directory=directory, url=None,
                node_type=class_type, node_id=node_id, source="api",
            ))
    return refs


def _is_api_format(workflow: dict[str, Any]) -> bool:
    if "nodes" in workflow:
        return False
    for v in workflow.values():
        if isinstance(v, dict) and "class_type" in v:
            return True
    return False


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def get_workflow_models(workflow: dict[str, Any]) -> list[ModelRef]:
    """Return all model references found in *workflow*, deduplicated and sorted."""
    refs = _collect_nodes_api(workflow) if _is_api_format(workflow) else _collect_nodes_ui(workflow)
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[ModelRef] = []
    for ref in refs:
        key = (ref.name, ref.directory, ref.url)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    unique.sort(key=lambda r: (r.directory, r.name, r.url or ""))
    return unique


def get_workflow_models_from_file(path: str | Path) -> list[ModelRef]:
    """Convenience wrapper: load *path* as JSON and call get_workflow_models."""
    with open(path, encoding="utf-8") as f:
        return get_workflow_models(json.load(f))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_CHUNK = 1 << 17   # 128 KiB read chunks
_BAR_WIDTH = 38


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress(label: str, done: int, total: int | None) -> None:
    if total:
        pct = done / total
        filled = int(_BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        line = f"\r  [{bar}] {pct:5.1%}  {_fmt_size(done)}/{_fmt_size(total)}  {label}"
    else:
        line = f"\r  {_fmt_size(done)} downloaded  {label}"
    sys.stderr.write(line)
    sys.stderr.flush()


def _build_headers(url: str, hf_token: str | None, civitai_token: str | None,
                   resume_from: int = 0) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": "workflow-model-downloader/1.0",
    }
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    if civitai_token and "civitai.com" in url:
        headers["Authorization"] = f"Bearer {civitai_token}"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    return headers


def _download_file(
    url: str,
    dest: Path,
    hf_token: str | None = None,
    civitai_token: str | None = None,
) -> str:
    """Download *url* to *dest*, resuming if a .part file exists.

    Returns the status string: 'downloaded' or 'resumed'.
    Raises urllib.error.URLError / OSError on failure.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    headers = _build_headers(url, hf_token, civitai_token, resume_from)
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=60) as resp:
        # Determine total size for progress display
        if resume_from and resp.status == 206:
            # Content-Range: bytes start-end/total
            cr = resp.headers.get("Content-Range", "")
            total: int | None = int(cr.split("/")[-1]) if "/" in cr else None
        else:
            resume_from = 0   # server didn't honour Range; start fresh
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl else None

        # If the complete file already exists at the right size, skip.
        if dest.exists() and total and dest.stat().st_size == total:
            return "skipped"

        label = dest.name
        mode = "ab" if resume_from else "wb"
        done = resume_from

        with open(part, mode) as fh:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _progress(label, done, total)

    sys.stderr.write("\n")
    part.rename(dest)
    return "resumed" if resume_from else "downloaded"


def download_workflow_models(
    workflow: dict[str, Any],
    models_dir: str | Path,
    hf_token: str | None = None,
    civitai_token: str | None = None,
) -> list[DownloadResult]:
    """Download every model referenced in *workflow* into *models_dir*.

    Parameters
    ----------
    workflow:
        Parsed workflow JSON dict (UI or API format).
    models_dir:
        Root of the ComfyUI models directory.  Each model is placed at
        ``models_dir / model.directory / model.name``.
    hf_token:
        Optional HuggingFace API token for gated model repos.
        Falls back to the ``HF_TOKEN`` environment variable.
    civitai_token:
        Optional CivitAI API token.
        Falls back to the ``CIVITAI_API_KEY`` environment variable.

    Returns
    -------
    list[DownloadResult]
        One entry per unique ModelRef.
    """
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    civitai_token = civitai_token or os.environ.get("CIVITAI_API_KEY")
    models_dir = Path(models_dir)
    refs = get_workflow_models(workflow)
    results: list[DownloadResult] = []

    for i, ref in enumerate(refs, 1):
        dest_dir = models_dir / ref.directory
        dest = dest_dir / ref.name
        prefix = f"[{i}/{len(refs)}]"

        if ref.url is None:
            print(f"{prefix} NO URL  {ref.directory}/{ref.name}", file=sys.stderr)
            results.append(DownloadResult(model=ref, dest=dest, status="no_url"))
            continue

        # Already fully downloaded?
        if dest.exists():
            print(f"{prefix} SKIP    {dest}", file=sys.stderr)
            results.append(DownloadResult(model=ref, dest=dest, status="skipped"))
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"{prefix} GET     {ref.url}", file=sys.stderr)

        try:
            status = _download_file(ref.url, dest, hf_token, civitai_token)
            # _download_file may return "skipped" if sizes matched mid-way
            if status == "skipped":
                print(f"{prefix} SKIP    {dest}", file=sys.stderr)
            results.append(DownloadResult(model=ref, dest=dest, status=status))
        except Exception as exc:
            part = dest.with_suffix(dest.suffix + ".part")
            print(f"\n{prefix} ERROR   {ref.name}: {exc}", file=sys.stderr)
            results.append(DownloadResult(model=ref, dest=dest, status="error",
                                          error=str(exc)))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List (and optionally download) model files for a ComfyUI workflow."
    )
    parser.add_argument("workflow", help="Path to workflow JSON file")
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing models into --models-dir",
    )
    parser.add_argument(
        "--models-dir", default=None, metavar="DIR",
        help="Root ComfyUI models/ directory (required with --download)",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace API token for gated repos (also reads HF_TOKEN env var)",
    )
    parser.add_argument(
        "--civitai-token", default=None, metavar="TOKEN",
        help="CivitAI API token (also reads CIVITAI_API_KEY env var)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print model list as JSON (list mode only)",
    )
    args = parser.parse_args()

    if args.download:
        if not args.models_dir:
            parser.error("--models-dir is required when using --download")
        with open(args.workflow, encoding="utf-8") as f:
            workflow = json.load(f)
        results = download_workflow_models(
            workflow,
            models_dir=args.models_dir,
            hf_token=args.hf_token,
            civitai_token=args.civitai_token,
        )
        errors = [r for r in results if r.status == "error"]
        if errors:
            sys.exit(1)
    else:
        models = get_workflow_models_from_file(args.workflow)
        if not models:
            print("No model references found.", file=sys.stderr)
            sys.exit(0)
        if args.json:
            print(json.dumps([
                {"name": m.name, "directory": m.directory, "url": m.url,
                 "node_type": m.node_type, "node_id": m.node_id, "source": m.source}
                for m in models
            ], indent=2))
        else:
            for m in models:
                url_part = f"\n    url: {m.url}" if m.url else ""
                print(
                    f"{m.directory}/{m.name}"
                    f"\n    node: {m.node_type} (id={m.node_id}, source={m.source})"
                    f"{url_part}"
                )


if __name__ == "__main__":
    _main()
