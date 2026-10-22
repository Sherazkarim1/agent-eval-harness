"""HTML report writer.

A single self-contained file: inline CSS, no JavaScript, no external assets.
It opens straight from disk or from a CI artifact viewer without a server, and
needs no charting dependency to install or break.

The presentation is a bespoke, print-friendly design system rather than a
component library: a small token set, a restrained type scale, hairline
dividers instead of stacked cards, and CSS-only micro-interactions (row hover,
focus rings, ``details`` disclosure). Dark mode follows the OS via
``prefers-color-scheme`` and a print stylesheet keeps it legible on paper.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from agent_eval.reporting.common import build_payload

#: Design tokens. Values live in CSS custom properties so light/dark schemes and
#: the print stylesheet can override them without touching markup.
_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --surface: #f7f8fa;
  --surface-2: #eef0f4;
  --border: #e4e7ec;
  --border-strong: #d3d8e0;
  --text: #14171c;
  --muted: #667085;
  --accent: #4338ca;
  --accent-soft: #eef0ff;
  --pass: #15803d;
  --pass-bg: #e7f6ec;
  --fail: #b91c1c;
  --fail-bg: #fdeaea;
  --err: #c2410c;
  --err-bg: #fdece1;
  --warn: #b45309;
  --warn-bg: #fbf3e2;
  --info: #1d4ed8;
  --info-bg: #e8effd;
  --radius: 10px;
  --radius-sm: 6px;
  --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
          "Liberation Mono", monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0e13;
    --surface: #12161d;
    --surface-2: #181d25;
    --border: #222834;
    --border-strong: #303845;
    --text: #e6e9ef;
    --muted: #97a1b0;
    --accent: #8f97ff;
    --accent-soft: #1a1f36;
    --pass: #4ade80;
    --pass-bg: #12251b;
    --fail: #f87171;
    --fail-bg: #2a1518;
    --err: #fb923c;
    --err-bg: #2a1c11;
    --warn: #fbbf24;
    --warn-bg: #2a2211;
    --info: #60a5fa;
    --info-bg: #14202f;
  }
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

code, .mono {
  font-family: var(--mono);
  font-size: 0.86em;
}

.wrap {
  max-width: 64rem;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4.5rem;
}

/* A single hairline accent rule anchors the document; no gradients. */
.topbar {
  height: 3px;
  background: var(--accent);
  border-radius: 0 0 2px 2px;
  print-color-adjust: exact;
  -webkit-print-color-adjust: exact;
}

header.report-head {
  padding: 1.75rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}

.head-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

h1 {
  margin: 0;
  font-size: 1.65rem;
  line-height: 1.2;
  letter-spacing: -0.02em;
  font-weight: 700;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 1.1rem;
  margin-top: 0.85rem;
  color: var(--muted);
  font-size: 0.82rem;
}
.meta code {
  background: var(--surface-2);
  padding: 0.1rem 0.35rem;
  border-radius: var(--radius-sm);
  color: var(--text);
}

.banner {
  border: 1px solid var(--border);
  border-left: 3px solid var(--warn);
  background: var(--warn-bg);
  color: var(--text);
  padding: 0.7rem 1rem;
  border-radius: var(--radius-sm);
  margin: 1.25rem 0 1.5rem;
  font-size: 0.85rem;
}
.banner strong { color: var(--warn); }

/* Verdict pill next to the title. */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  border: 1px solid transparent;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}
.pill::before {
  content: "";
  width: 0.42rem;
  height: 0.42rem;
  border-radius: 50%;
  background: currentColor;
}
.pill.pass { color: var(--pass); background: var(--pass-bg); }
.pill.fail { color: var(--fail); background: var(--fail-bg); }
.pill.err { color: var(--err); background: var(--err-bg); }

h2 {
  font-size: 1.02rem;
  letter-spacing: -0.01em;
  font-weight: 650;
  margin: 2.25rem 0 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
h2::before {
  content: "";
  width: 0.4rem;
  height: 1.02rem;
  border-radius: 2px;
  background: var(--accent);
  flex: none;
}

/* Metric strip: one hairline grid, never nested cards. */
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.stat {
  background: var(--bg);
  padding: 0.85rem 1rem;
}
.stat-label {
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 650;
}
.stat-value {
  font-size: 1.4rem;
  font-weight: 650;
  margin-top: 0.2rem;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.stat.lead .stat-value { color: var(--accent); }
.stat-value small { font-size: 0.6em; font-weight: 600; color: var(--muted); }

/* Score meters: a precise technical read, not decorative bars. */
.meters { border-top: 1px solid var(--border); }
.meter {
  display: grid;
  grid-template-columns: 9.5rem 1fr 4.5rem;
  align-items: center;
  gap: 0.9rem;
  padding: 0.5rem 0.6rem;
  border-bottom: 1px solid var(--border);
  border-radius: var(--radius-sm);
  transition: background-color 0.15s ease;
}
.meter:hover { background: var(--surface); }
.meter-name {
  font-size: 0.85rem;
  font-weight: 550;
  color: var(--text);
}
.meter-track {
  display: block;
  height: 0.5rem;
  background: var(--surface-2);
  border-radius: 999px;
  overflow: hidden;
}
.meter-fill {
  display: block;
  height: 100%;
  background: var(--accent);
  border-radius: 999px;
  transition: width 0.4s ease;
  print-color-adjust: exact;
  -webkit-print-color-adjust: exact;
}
.meter-value {
  text-align: right;
  font-family: var(--mono);
  font-size: 0.84rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.meter.lead .meter-name { font-weight: 700; }
.meter.lead .meter-value { color: var(--accent); font-size: 0.95rem; }

/* Two-column key/value facts (latency, tokens). */
.kv {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
  gap: 0.1rem 2rem;
  margin: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.35rem 1.1rem;
  background: var(--surface);
}
.kv > div {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.86rem;
}
.kv > div:last-child { border-bottom: 0; }
.kv dt { color: var(--muted); font-weight: 550; }
.kv dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.note { color: var(--muted); font-size: 0.78rem; font-style: italic; }

/* Tables. */
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
thead th {
  text-align: left;
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 650;
  padding: 0.65rem 0.9rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody td {
  padding: 0.6rem 0.9rem;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tbody tr:last-child td { border-bottom: 0; }
tbody tr { transition: background-color 0.12s ease; }
tbody tr:hover { background: var(--surface); }
td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
td .mono { color: var(--text); }
td .tags { color: var(--muted); font-size: 0.8rem; }

.sev-error { color: var(--fail); font-weight: 650; }
.sev-warning { color: var(--warn); font-weight: 650; }
.sev-info { color: var(--info); font-weight: 650; }

/* Expandable problem cases. */
details.problem {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  margin: 0.6rem 0;
  overflow: hidden;
}
details.problem summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.7rem 1rem;
  font-weight: 600;
  font-size: 0.9rem;
  transition: background-color 0.15s ease;
}
details.problem summary::-webkit-details-marker { display: none; }
details.problem summary:hover { background: var(--surface-2); }
details.problem summary::after {
  content: "+";
  margin-left: auto;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 1rem;
  line-height: 1;
  transition: transform 0.2s ease;
}
details.problem[open] summary::after { transform: rotate(45deg); }
details.problem[open] summary { border-bottom: 1px solid var(--border); }
.problem-body { padding: 0.9rem 1rem 1.1rem; }
.problem-body h3 {
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1rem 0 0.35rem;
}
.problem-body h3:first-child { margin-top: 0; }

pre {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.7rem 0.85rem;
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 0.78rem;
  line-height: 1.5;
  margin: 0.25rem 0 0;
}

ul.plain { margin: 0.25rem 0 0; padding-left: 1.1rem; }
ul.plain li { margin: 0.2rem 0; }
li.ok::marker { content: "\\2713  "; color: var(--pass); font-weight: 700; }
li.bad::marker { content: "\\2715  "; color: var(--fail); font-weight: 700; }

footer.report-foot {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem 1rem;
  justify-content: space-between;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 2px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition: none !important;
    animation: none !important;
    scroll-behavior: auto !important;
  }
}

@media print {
  :root {
    --bg: #ffffff;
    --surface: #ffffff;
    --surface-2: #f2f2f2;
    --border: #d8dbe0;
    --text: #000000;
    --muted: #555555;
    --accent: #000000;
  }
  body { background: #ffffff; font-size: 12px; }
  .wrap { max-width: none; padding: 0; }
  .meter-fill { background: #000000; }
  tbody tr:hover { background: transparent; }
  details.problem { break-inside: avoid; }
  a { color: #000000; text-decoration: none; }
}
"""


def _esc(value: Any) -> str:
    """HTML-escape any value."""
    return html.escape("" if value is None else str(value))


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format a score, preserving ``n/a`` for missing measurements."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _pill(status: str) -> str:
    """Status pill with a semantic colour and a status dot."""
    cls = {"passed": "pass", "failed": "fail", "errored": "err"}.get(status, status)
    return f'<span class="pill {cls}">{_esc(status)}</span>'


def _stat(label: str, value: str, cls: str = "") -> str:
    """One metric cell for the summary strip."""
    extra = f" {cls}" if cls else ""
    return (
        f'<div class="stat{extra}">'
        f'<div class="stat-label">{_esc(label)}</div>'
        f'<div class="stat-value">{_esc(value)}</div>'
        "</div>"
    )


def _meter(name: str, value: float | None, cls: str = "") -> str:
    """A labelled score meter; the fill width is inline so it needs no script."""
    pct = 0.0 if value is None else max(0.0, min(1.0, value)) * 100
    extra = f" {cls}" if cls else ""
    return (
        f'<div class="meter{extra}">'
        f'<span class="meter-name">{_esc(name)}</span>'
        f'<span class="meter-track">'
        f'<span class="meter-fill" style="width:{pct:.1f}%"></span>'
        f"</span>"
        f'<span class="meter-value">{_fmt(value)}</span>'
        "</div>"
    )


def _table(headers: list[str], rows: list[str], aligns: list[str] | None = None) -> str:
    """Render a bordered, hoverable table.

    Args:
        headers: Column headings.
        rows: Pre-rendered ``<td>`` fragments, one list item per row.
        aligns: Optional per-column CSS class, e.g. ``"num"`` for right alignment.
    """
    aligns = aligns or [""] * len(headers)
    head = "".join(
        f'<th class="{a}">{_esc(h)}</th>' if a else f"<th>{_esc(h)}</th>"
        for h, a in zip(headers, aligns, strict=True)
    )
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return (
        '<div class="table-wrap">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</div>"
    )


def _kv(pairs: list[tuple[str, str]]) -> str:
    """Render key/value facts as a compact two-column definition list."""
    items = "".join(f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in pairs)
    return f'<dl class="kv">{items}</dl>'


def _section(anchor: str, title: str, body: str) -> str:
    return f'<section id="{_esc(anchor)}"><h2>{_esc(title)}</h2>{body}</section>'


def render(payload: dict[str, Any]) -> str:
    """Render a report payload as a self-contained HTML document.

    Args:
        payload: The canonical payload from
            :func:`agent_eval.reporting.common.build_payload`.

    Returns:
        A complete HTML document.
    """
    aggregate = payload["aggregate"]
    scores = aggregate["scores"]
    latency = aggregate["latency_ms"]
    tokens = aggregate["tokens"]
    cost = aggregate["cost"]

    problem_count = payload["problem_count"]
    verdict = "passed" if problem_count == 0 else "failed"

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Evaluation - {_esc(payload['dataset']['ref'])}</title>",
        f"<style>{_CSS}</style></head><body>",
        '<div class="topbar"></div>',
        '<div class="wrap">',
    ]

    # Header: dataset identity, verdict, and run metadata.
    agent = payload["agent"]
    judge = payload["judge"]
    meta = [
        f"Run <code>{_esc(payload['run_id'])}</code>",
        f"Agent <code>{_esc(agent['name'])} {_esc(agent['version'])}</code>",
        f"Judge <code>{_esc(judge['name'])}</code>",
        f"<span>{_esc(payload['started_at'])}</span>",
    ]
    parts.append(
        '<header class="report-head">'
        '<div class="head-row">'
        f"<h1>{_esc(payload['dataset']['ref'])}</h1>"
        f"{_pill(verdict)}"
        "</div>"
        f'<div class="meta">{"".join(meta)}</div>'
        "</header>"
    )

    if payload["is_mock"]:
        parts.append(
            '<div class="banner"><strong>Mock run.</strong> These scores come from a '
            "mock agent and judge; they are simulated, not real measurements.</div>"
        )

    # Summary strip.
    parts.append(
        '<div class="stats">'
        + _stat("Total", str(aggregate["total"]))
        + _stat("Passed", str(aggregate["passed"]))
        + _stat("Failed", str(aggregate["failed"]))
        + _stat("Errored", str(aggregate["errored"]))
        + _stat("Pass rate", _pct(aggregate["pass_rate"]))
        + _stat("Overall", _fmt(scores["overall"]), cls="lead")
        + "</div>"
    )

    # Scores as meters.
    meters = "".join(
        [
            _meter("Overall", scores["overall"], cls="lead"),
            _meter("Deterministic", scores["deterministic"]),
            _meter("Judge", scores["judge"]),
            _meter("Trajectory", scores["trajectory"]),
            _meter("Performance", scores["performance"]),
        ]
    )
    parts.append(_section("scores", "Scores", f'<div class="meters">{meters}</div>'))

    # Latency & token facts.
    facts = _kv(
        [
            ("Average latency", f"{latency['average']} ms"),
            ("p50 latency", f"{latency['p50']} ms"),
            ("p95 latency", f"{latency['p95']} ms"),
            ("Max latency", f"{latency['max']} ms"),
            ("Input tokens", f"{tokens['input']:,}"),
            ("Output tokens", f"{tokens['output']:,}"),
            ("Total tokens", f"{tokens['total']:,}"),
            ("API cost", f"${cost['api_cost_usd']:.4f}"),
        ]
    )
    parts.append(_section("latency", "Latency & tokens", facts))

    # Cases table.
    case_rows: list[str] = []
    for case in payload["cases"]:
        case_score = None if case["score"] is None else case["score"]["overall"]
        tags = ", ".join(case["tags"])
        case_rows.append(
            f'<td><span class="mono">{_esc(case["case_id"])}</span></td>'
            f"<td>{_pill(case['status'])}</td>"
            f'<td class="num">{_fmt(case_score)}</td>'
            f'<td class="num">{case["latency_ms"]} ms</td>'
            f'<td class="tags">{_esc(tags)}</td>'
        )
    parts.append(
        _section(
            "cases",
            "Cases",
            _table(
                ["Case", "Status", "Score", "Latency", "Tags"],
                case_rows,
                aligns=["", "", "num", "num", ""],
            ),
        )
    )

    # Problem details.
    problems = [c for c in payload["cases"] if c["status"] != "passed"]
    if problems:
        blocks: list[str] = []
        for case in problems:
            blocks.append(_problem_block(case))
        parts.append(_section("problems", f"Problems ({len(problems)})", "".join(blocks)))

    if payload["truncated_problems"]:
        parts.append(
            f'<p class="note">{_esc(payload["truncated_problems"])} further problem(s) '
            "omitted; see the JSON report for the full set.</p>"
        )

    # Regression.
    regression = payload["regression"]
    if regression is not None:
        parts.append(_regression_section(regression))

    # Footer.
    foot_left = f'Run <span class="mono">{_esc(payload["run_id"])}</span>'
    foot_right = f"agent-eval &middot; schema v{_esc(payload['schema_version'])}"
    parts.append(
        f'<footer class="report-foot"><span>{foot_left}</span><span>{foot_right}</span></footer>'
    )

    parts.append("</div></body></html>")
    return "\n".join(parts) + "\n"


def _problem_block(case: dict[str, Any]) -> str:
    """Render one expandable problem case."""
    case_score = None if case["score"] is None else case["score"]["overall"]
    summary = (
        f"{_pill(case['status'])}"
        f'<span class="mono">{_esc(case["case_id"])}</span>'
        f'<span class="note">{_fmt(case_score)}</span>'
    )
    body: list[str] = []

    if case["input"]:
        body.append("<h3>Input</h3>")
        body.append(f"<pre>{_esc(case['input'])}</pre>")

    if case["error"]:
        body.append("<h3>Error</h3>")
        body.append(f"<pre>{_esc(case['error'])}</pre>")

    failed = [a for a in case["assertions"] if not a["passed"]]
    if failed:
        body.append("<h3>Failed assertions</h3>")
        items = "".join(
            f'<li class="bad"><code>{_esc(a["assertion"])}</code>: {_esc(a["message"])}</li>'
            for a in failed
        )
        body.append(f'<ul class="plain">{items}</ul>')

    passed = [a for a in case["assertions"] if a["passed"]]
    if passed:
        body.append("<h3>Passed assertions</h3>")
        items = "".join(f'<li class="ok"><code>{_esc(a["assertion"])}</code></li>' for a in passed)
        body.append(f'<ul class="plain">{items}</ul>')

    if case.get("output"):
        body.append("<h3>Output</h3>")
        body.append(f"<pre>{_esc(case['output'])}</pre>")

    trajectory = case.get("trajectory")
    if trajectory and trajectory["tool_calls"]:
        body.append("<h3>Tool calls</h3>")
        items = "".join(
            f"<li><code>{_esc(call['name'])}</code> {_esc(call['arguments'])}</li>"
            for call in trajectory["tool_calls"]
        )
        body.append(f'<ul class="plain">{items}</ul>')

    return (
        '<details class="problem">'
        f"<summary>{summary}</summary>"
        f'<div class="problem-body">{"".join(body)}</div>'
        "</details>"
    )


def _regression_section(regression: dict[str, Any]) -> str:
    """Render the regression comparison, or a note when no baseline exists."""
    if not regression["has_baseline"]:
        note = "<p>No baseline available, so no comparison was performed.</p>"
        return _section("regression", "Regression", note)

    delta = regression["score_delta"]
    delta_text = "n/a" if delta is None else f"{delta:+.4f}"
    body = [
        "<p>"
        f"Status <strong>{_esc(regression['status'])}</strong> vs baseline "
        f"<code>{_esc(regression['baseline_name'])}</code> "
        f"(score delta {_esc(delta_text)})."
        "</p>"
    ]
    if regression["findings"]:
        rows = [
            f'<td class="sev-{_esc(f["severity"])}">{_esc(f["severity"].upper())}</td>'
            f"<td>{_esc(f['kind'])}</td>"
            f"<td>{_esc(f['message'])}</td>"
            for f in regression["findings"]
        ]
        body.append(_table(["Severity", "Kind", "Detail"], rows))
    return _section("regression", "Regression", "".join(body))


def write(path: Path, record: Any, cases: list[Any], **kwargs: Any) -> Path:
    """Write an HTML report to ``path``.

    Args:
        path: Destination file; parent directories are created as needed.
        record: The run record.
        cases: Per-case results.
        **kwargs: Forwarded to :func:`build_payload`.

    Returns:
        The path written.
    """
    payload = build_payload(record, cases, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path
