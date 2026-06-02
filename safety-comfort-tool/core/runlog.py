"""
core/runlog.py — Universal run-logging utility

Every pipeline run (Mode A action, Mode B/C invocation) emits a structured
record so that the user can later answer:
  • "Which parameters did I use for this output?"
  • "How many units were analyzed? How many had data issues?"
  • "When was this generated?"

Outputs (in the run's directory):
  • run_log_{timestamp}.json — machine-readable
  • run_log_{timestamp}.md   — human-readable summary

Both files capture: timestamp, mode, parameters, agent versions (file paths),
inputs, outputs, summary stats, errors. Use via the RunLogger context manager
or call write_log() directly.
"""
from __future__ import annotations
import os
import json
import sys
import platform
import traceback
from datetime import datetime
from typing import Optional, Any


class RunLogger:
    """
    Collect run metadata and emit json + md side-by-side.

    Usage:
        with RunLogger("ModeB", out_dir, params={"target_dong": "..."}) as log:
            log.note("Loaded 79 cells")
            log.add_artifact("cells_csv", "/path/to/cells.csv", "Per-cell I_phy")
            log.add_summary("I_phy_range", [-0.8, 3.4])
            ... (exceptions auto-captured)
    """

    def __init__(
        self,
        mode: str,                       # 'ModeA' | 'ModeB' | 'ModeC' | free string
        out_dir: str,
        *,
        params: Optional[dict] = None,
        title: Optional[str] = None,
    ):
        self.mode = mode
        self.out_dir = out_dir
        self.title = title or f"{mode} run"
        self.started = datetime.now()
        self.ended: Optional[datetime] = None
        self.params: dict = dict(params or {})
        self.notes: list[str] = []
        self.artifacts: list[dict] = []
        self.summary: dict = {}
        self.errors: list[dict] = []
        self.status: str = "running"

        os.makedirs(out_dir, exist_ok=True)
        ts = self.started.strftime("%Y%m%d_%H%M%S")
        self.json_path = os.path.join(out_dir, f"run_log_{ts}.json")
        self.md_path   = os.path.join(out_dir, f"run_log_{ts}.md")

    # -------------------------------------------------------------
    # Builders
    # -------------------------------------------------------------
    def note(self, text: str) -> None:
        self.notes.append(text)

    def add_artifact(self, key: str, path: Optional[str], description: str = "") -> None:
        size = None
        exists = bool(path) and os.path.exists(path)
        if exists:
            try:
                size = os.path.getsize(path)
            except OSError:
                pass
        self.artifacts.append({
            "key":         key,
            "path":        path,
            "exists":      exists,
            "size_bytes":  size,
            "description": description,
        })

    def add_summary(self, key: str, value: Any) -> None:
        # Coerce non-JSON-serializable types
        try:
            json.dumps(value, default=str)
            self.summary[key] = value
        except (TypeError, ValueError):
            self.summary[key] = str(value)

    def add_error(self, where: str, exc: BaseException) -> None:
        self.errors.append({
            "where":   where,
            "type":    type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })

    # -------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.add_error("context-manager", exc)
            self.status = "failed"
        else:
            self.status = "ok"
        self.write()
        return False  # do not suppress

    # -------------------------------------------------------------
    # Emit
    # -------------------------------------------------------------
    def write(self) -> tuple[str, str]:
        """Write both files. Returns (json_path, md_path)."""
        self.ended = datetime.now()
        elapsed_s = (self.ended - self.started).total_seconds()

        payload = {
            "title":      self.title,
            "mode":       self.mode,
            "status":     self.status,
            "started":    self.started.isoformat(timespec="seconds"),
            "ended":      self.ended.isoformat(timespec="seconds"),
            "elapsed_s":  round(elapsed_s, 2),
            "host":       platform.node(),
            "platform":   platform.platform(),
            "python":     sys.version.split()[0],
            "params":     self.params,
            "summary":    self.summary,
            "artifacts":  self.artifacts,
            "notes":      self.notes,
            "errors":     self.errors,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        # Markdown
        md = self._render_md(payload)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(md)

        return self.json_path, self.md_path

    # -------------------------------------------------------------
    # Markdown renderer
    # -------------------------------------------------------------
    @staticmethod
    def _render_md(p: dict) -> str:
        ok = "✅" if p["status"] == "ok" else "❌"
        lines = [
            f"# {p['title']}",
            "",
            f"- **Status**: {ok} {p['status']}",
            f"- **Mode**: `{p['mode']}`",
            f"- **Started**: {p['started']}",
            f"- **Ended**: {p['ended']}",
            f"- **Elapsed**: {p['elapsed_s']} s",
            f"- **Host**: {p['host']}  ·  Python {p['python']}",
            "",
            "## Parameters",
        ]
        if p["params"]:
            for k, v in p["params"].items():
                lines.append(f"- `{k}` = `{v}`")
        else:
            lines.append("- (none)")

        lines += ["", "## Summary"]
        if p["summary"]:
            for k, v in p["summary"].items():
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
                    # table-like list of records
                    lines.append(f"### {k}")
                    keys = list(v[0].keys())
                    lines.append("| " + " | ".join(keys) + " |")
                    lines.append("|" + "|".join("---" for _ in keys) + "|")
                    for row in v:
                        lines.append("| " + " | ".join(str(row.get(kk, "")) for kk in keys) + " |")
                else:
                    lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- (none)")

        lines += ["", "## Artifacts"]
        if p["artifacts"]:
            lines.append("| key | path | exists | size | description |")
            lines.append("|---|---|---|---|---|")
            for a in p["artifacts"]:
                sz = f"{a['size_bytes']:,}" if a["size_bytes"] else "-"
                ex = "✓" if a["exists"] else "✗"
                lines.append(f"| `{a['key']}` | `{a['path']}` | {ex} | {sz} | {a['description']} |")
        else:
            lines.append("- (none)")

        if p["notes"]:
            lines += ["", "## Notes"]
            for n in p["notes"]:
                lines.append(f"- {n}")

        if p["errors"]:
            lines += ["", "## Errors"]
            for e in p["errors"]:
                lines.append(f"### at `{e['where']}`")
                lines.append(f"- type: `{e['type']}`")
                lines.append(f"- message: {e['message']}")
                lines.append("```")
                lines.append(e["traceback"].rstrip())
                lines.append("```")

        return "\n".join(lines) + "\n"
