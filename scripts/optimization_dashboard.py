# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimization dashboard — observe what each agent2 round did and how it performed.

A deliberately small FastAPI app (inspired by Train_engine's web monitor, minus
the recording / multi-agent chat machinery): it discovers every
``trajectory.jsonl`` written by the optimization orchestrator, renders the
performance curve, and shows per-round detail (speedup, SOL %, bottleneck,
Triton-config changes, and the generated kernel / prompt for that round).

Run:
    python scripts/optimization_dashboard.py --root . --port 8086

The data layer (``discover_runs`` / ``load_run``) is pure and unit-tested; the
HTTP layer is a thin wrapper that polls the filesystem so a live run updates in
place.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.searching.trajectory import read_trajectory

# Directories under the project root that may contain optimization artifacts.
# Each trajectory.jsonl marks one run; its parent dir holds the per-round
# kernels / prompts / strategy json.
_TRAJECTORY_GLOB = "**/trajectory.jsonl"


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Find every trajectory.jsonl under *root* and summarize each run.

    Returns a list of run summaries (most recently modified first):
        {id, dir, rounds, baseline_ms, best_ms, best_speedup, best_round, mtime}
    """
    root = Path(root)
    runs: list[dict[str, Any]] = []
    for traj in root.glob(_TRAJECTORY_GLOB):
        rows = read_trajectory(traj)
        if not rows:
            continue
        run_dir = traj.parent
        baseline_ms = next(
            (r.get("time_ms") for r in rows if r.get("kind") == "baseline"), None
        )
        round_rows = [r for r in rows if r.get("kind") == "round"]
        timed = [r for r in round_rows if r.get("time_ms")]
        best = min(timed, key=lambda r: r["time_ms"]) if timed else None
        best_ms = best["time_ms"] if best else None
        best_speedup = (
            baseline_ms / best_ms if baseline_ms and best_ms and best_ms > 0 else None
        )
        runs.append(
            {
                "id": _run_id(root, run_dir),
                "dir": str(run_dir),
                "rounds": len(round_rows),
                "baseline_ms": baseline_ms,
                "best_ms": best_ms,
                "best_speedup": best_speedup,
                "best_round": best.get("round") if best else None,
                "mtime": traj.stat().st_mtime,
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _run_id(root: Path, run_dir: Path) -> str:
    """A stable, URL-safe id derived from the run dir relative to root."""
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        rel = run_dir
    return str(rel).replace("/", "~") or "."


def _resolve_run_dir(root: Path, run_id: str) -> Path:
    return (Path(root) / run_id.replace("~", "/")).resolve()


def load_run(root: Path, run_id: str) -> dict[str, Any]:
    """Load one run's full trajectory plus the complete per-round attempt.

    The manager-level trajectory lives in the run dir; the full detail of what
    each round tried (diagnosis, prescription, prompt, LLM reply, reflexion, and
    the generated kernel) lives in the per-round worker artifact dir at
    ``<run_dir>/workers/*/r<N>/artifacts/``. We attach all of it so the dashboard
    can review every attempt, not just the winning time.
    """
    run_dir = _resolve_run_dir(root, run_id)
    # Containment check: never read outside the configured root.
    if Path(root).resolve() not in run_dir.parents and run_dir != Path(root).resolve():
        raise ValueError(f"run id escapes root: {run_id}")
    rows = read_trajectory(run_dir / "trajectory.jsonl")
    for r in rows:
        if r.get("kind") != "round":
            continue
        n = int(r.get("round"))
        art = _round_artifact_dir(run_dir, n)
        if art is None:
            continue
        # The per-round files are named by the WORKER's internal round counter
        # (always round001 — each manager round runs one worker round), NOT by
        # the manager round number. So discover the real prefix present in this
        # dir instead of assuming f"round{n:03d}".
        tag = _round_file_tag(art)
        # The prescription/diagnosis the LLM produced from the NCU metrics.
        r["strategy"] = _read_json_if_exists(art / f"{tag}_strategy.json")
        r["opt_prompt"] = _read_if_exists(art / f"{tag}_opt_prompt.txt", 16000)
        r["opt_reply"] = _read_if_exists(art / f"{tag}_opt_reply.txt", 16000)
        r["reasoning"] = _read_if_exists(art / f"{tag}_reasoning.txt", 40000)
        r["reflexion"] = _read_if_exists(art / f"{tag}_reflexion.txt", 8000)
        # The kernel this round produced (round_1 is the optimized candidate);
        # a failed round only has kernel_round_0 (the carried-over baseline).
        opt_kernel = art / "kernel_round_1.py"
        r["kernel_code"] = _read_if_exists(opt_kernel, 20000)
        # Round status: a round is "ok" only if it produced a verified optimized
        # kernel. Failed rounds still carry a diagnosis worth reviewing, plus a
        # structured failure record telling apart the failure modes.
        # This round's OWN measured time (bare new-kernel time), distinct from
        # the reverted best-so-far that trajectory carries. Shows what each round
        # actually achieved even when it was slower and got reverted.
        r["round_result"] = _read_json_if_exists(art / f"{tag}_result.json")
        r["failure"] = _read_json_if_exists(art / f"{tag}_failure.json")
        if opt_kernel.exists():
            r["status"] = "ok"
        elif r.get("failure"):
            # generation-stage failure: empty_response / no_code_block / request_error
            r["status"] = r["failure"].get("kind", "gen_failed")
        elif r.get("opt_reply") and r.get("strategy"):
            # produced a reply + diagnosis but the kernel didn't verify downstream
            r["status"] = "verify_failed"
        elif r.get("strategy"):
            r["status"] = "generated_no_valid_kernel"
        else:
            r["status"] = "no_diagnosis"
        # Real NCU numbers for this round's kernel, straight from the CSV, so the
        # review shows tensor/compute/dram/L1/occupancy — not just the summary.
        r["ncu"] = _extract_ncu_sol(art)
        # Quick "did it try TMA / pipelining" flags for review at a glance.
        kc = r.get("kernel_code") or ""
        r["used_tma"] = any(
            tok in kc
            for tok in (
                "make_tensor_descriptor",
                "tensor_descriptor",
                "_experimental_tma",
                "TensorDescriptor",
                "cp.async.bulk",
            )
        )
    return {"id": run_id, "dir": str(run_dir), "rows": rows}


# NCU columns worth surfacing per round (short label -> full metric name).
_NCU_COLS = {
    "tensor_sol": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "compute_sol": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "regs_per_thread": "launch__registers_per_thread",
}


def _extract_ncu_sol(art: Path) -> dict[str, float] | None:
    """Pull the key SOL columns from this round's NCU CSV, if present.

    Surfaces the real tensor/compute/dram/L1/occupancy numbers so the review
    reflects hardware truth (e.g. tensor cores starved at 16%), not the coarse
    trajectory summary.
    """
    import csv

    csvs = sorted(art.glob("remote_ncu_*/ncu_round_*.csv")) + sorted(
        art.glob("ncu_round_*.csv")
    )
    if not csvs:
        return None
    try:
        rows = list(csv.reader(csvs[0].open(encoding="utf-8", errors="replace")))
    except OSError:
        return None
    if not rows:
        return None
    hdr_i = max(range(len(rows)), key=lambda i: len(rows[i]))
    hdr = rows[hdr_i]
    idx = {label: hdr.index(col) for label, col in _NCU_COLS.items() if col in hdr}
    if not idx:
        return None
    kname = hdr.index("Kernel Name") if "Kernel Name" in hdr else None
    for row in rows[hdr_i + 1 :]:
        if len(row) < len(hdr):
            continue
        # Prefer the triton kernel row (skip pytorch at:: kernels).
        if kname is not None and row[kname].startswith(("at::", "void at::")):
            continue
        out: dict[str, float] = {}
        for label, i in idx.items():
            try:
                out[label] = float(row[i].replace(",", ""))
            except (ValueError, IndexError):
                pass
        if out:
            return out
    return None


def _round_artifact_dir(run_dir: Path, n: int) -> Path | None:
    """Find the per-round worker artifact dir for manager round *n*.

    greedy/beam layout: ``<run_dir>/workers/<worker>/r<N>/artifacts``.
    Returns the first match (greedy has a single worker) or None.
    """
    for cand in sorted(run_dir.glob(f"workers/*/r{n}/artifacts")):
        if cand.is_dir():
            return cand
    return None


def _round_file_tag(art: Path) -> str:
    """Return the ``roundNNN`` prefix actually used by files in *art*.

    Files are named by the worker's internal round counter (round001), not the
    manager round, so we read the real prefix off disk. Falls back to round001.
    """
    for f in sorted(art.glob("round*_*")):
        name = f.name
        if len(name) >= 8 and name[:5] == "round" and name[5:8].isdigit():
            return name[:8]
    return "round001"


def _read_json_if_exists(path: Path) -> Any | None:
    import json

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None



def _read_if_exists(path: Path, max_bytes: int) -> str | None:
    if not path.exists():
        return None
    data = path.read_text(encoding="utf-8", errors="replace")
    return data if len(data) <= max_bytes else data[:max_bytes] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KernelAgent Optimization Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
         background:#0d1117; color:#c9d1d9; }
  header { padding:10px 16px; background:#161b22; border-bottom:1px solid #30363d;
           display:flex; align-items:center; gap:12px; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .muted { color:#8b949e; }
  #wrap { display:flex; height:calc(100vh - 43px); }
  #runs { width:280px; border-right:1px solid #30363d; overflow-y:auto; flex:none; }
  .run { padding:10px 14px; border-bottom:1px solid #21262d; cursor:pointer; }
  .run:hover { background:#161b22; }
  .run.sel { background:#1f6feb22; border-left:3px solid #1f6feb; }
  .run b { color:#e6edf3; }
  .pill { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; }
  .up { background:#238636; color:#fff; } .down{ background:#6e7681; color:#fff; }
  #main { flex:1; overflow-y:auto; padding:16px; }
  canvas { background:#161b22; border:1px solid #30363d; border-radius:6px; width:100%; }
  table { width:100%; border-collapse:collapse; margin-top:14px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid #21262d; vertical-align:top; }
  th { color:#8b949e; font-weight:600; position:sticky; top:0; background:#0d1117; }
  tr.best td { background:#23863622; }
  tr.fail td { color:#f85149; }
  .chg { color:#d29922; }
  .ncu { color:#8b949e; font-size:11px; margin:2px 0; }
  details { margin-top:4px; } summary { cursor:pointer; color:#58a6ff; }
  pre { background:#161b22; border:1px solid #30363d; padding:8px; overflow:auto;
        max-height:340px; border-radius:6px; white-space:pre-wrap; }
  .kpi { display:flex; gap:18px; margin-bottom:12px; }
  .kpi div { background:#161b22; border:1px solid #30363d; border-radius:6px; padding:8px 14px; }
  .kpi b { font-size:20px; color:#e6edf3; display:block; }
</style>
</head>
<body>
<header>
  <h1>KernelAgent · Optimization Trajectory</h1>
  <span class="muted" id="status">loading…</span>
</header>
<div id="wrap">
  <div id="runs"></div>
  <div id="main"><p class="muted">Select a run on the left.</p></div>
</div>
<script>
let SEL = null;
const fmt = (x, d=3) => (x==null ? "—" : Number(x).toFixed(d));

async function loadRuns() {
  const r = await fetch("/api/runs"); const runs = await r.json();
  document.getElementById("status").textContent =
    runs.length + " run(s) · auto-refresh 3s";
  const el = document.getElementById("runs");
  el.innerHTML = runs.map(run => {
    const sp = run.best_speedup ? `<span class="pill up">${fmt(run.best_speedup,2)}×</span>`
                                : `<span class="pill down">—</span>`;
    return `<div class="run ${run.id===SEL?'sel':''}" onclick="selectRun('${run.id}')">
      <b>${run.id}</b> ${sp}<br/>
      <span class="muted">${run.rounds} rounds · base ${fmt(run.baseline_ms)}ms
      · best ${fmt(run.best_ms)}ms (r${run.best_round??'—'})</span></div>`;
  }).join("") || '<p class="muted" style="padding:14px">No runs found.</p>';
  // Auto-select the newest run that actually has rounds, not just a bare
  // baseline (an in-progress run shows 0 rounds and would look empty).
  if (!SEL && runs.length) {
    const withData = runs.find(r => r.rounds > 0) || runs[0];
    selectRun(withData.id);
  }
}

async function selectRun(id) {
  SEL = id; document.querySelectorAll(".run").forEach(d=>d.classList.toggle("sel",
    d.querySelector("b")?.textContent===id));
  const r = await fetch("/api/runs/" + encodeURIComponent(id));
  const data = await r.json(); render(data);
}

function render(data) {
  const rows = data.rows || [];
  const base = rows.find(x=>x.kind==="baseline");
  const rounds = rows.filter(x=>x.kind==="round");
  const baseMs = base ? base.time_ms : null;
  const timed = rounds.filter(x=>x.time_ms);
  const best = timed.length ? timed.reduce((a,b)=>b.time_ms<a.time_ms?b:a) : null;
  const main = document.getElementById("main");
  main.innerHTML = `
    <div class="kpi">
      <div>baseline<b>${fmt(baseMs)} ms</b></div>
      <div>best<b>${fmt(best?best.time_ms:null)} ms</b></div>
      <div>speedup<b>${best&&baseMs?fmt(baseMs/best.time_ms,2)+'×':'—'}</b></div>
      <div>pytorch ref<b>${fmt(base?base.pytorch_ms:null)} ms</b></div>
    </div>
    <canvas id="chart" height="240"></canvas>
    <table><thead><tr><th>r</th><th>this round (ms)</th><th>best (ms)</th><th>speedup</th>
      <th>Δ%</th><th>tensor SOL%</th><th>bottleneck</th><th>config Δ</th><th>status</th><th>attempts</th>
    </tr></thead><tbody>${rounds.map(rowHtml).join("")}</tbody></table>`;
  drawChart(baseMs, rounds);
}

function rowHtml(r) {
  const cfg = Object.entries(r.config_changes||{}).map(([k,v])=>
    `<span class="chg">${k}:${v}</span>`).join(" ") || "—";
  const cls = !r.verified ? "fail" : (r.is_best ? "best" : "");
  const det = (label,txt) => txt ? `<details><summary>${label}</summary><pre>${esc(txt)}</pre></details>` : "";
  // Diagnosis / prescription: pull the summary + recommended fixes out of the
  // strategy json so you can review what the agent decided and why.
  let diag = "";
  if (Array.isArray(r.strategy) && r.strategy.length) {
    diag = r.strategy.map(b => {
      const fixes = (b.root_causes||[]).flatMap(rc=>(rc.fixes||[]).map(f=>f.fix));
      return `<b>[${b.category}]</b> ${esc(b.summary||"")}` +
        (fixes.length ? `<br/><span class="chg">fix:</span> ${esc(fixes.join(" | "))}` : "");
    }).join("<hr style='border-color:#30363d'/>");
    diag = `<details><summary>diagnosis / prescription</summary><div style="padding:6px">${diag}</div></details>`;
  }
  const tma = r.used_tma ? `<span class="pill up">TMA</span>` : "";
  // Real NCU numbers for this round (hardware truth, not the coarse summary).
  let ncu = "";
  if (r.ncu) {
    const n = r.ncu;
    const parts = [];
    if (n.tensor_sol!=null) parts.push(`tensor ${fmt(n.tensor_sol,1)}%`);
    if (n.compute_sol!=null) parts.push(`sm ${fmt(n.compute_sol,1)}%`);
    if (n.dram_sol!=null) parts.push(`dram ${fmt(n.dram_sol,1)}%`);
    if (n.l1_sol!=null) parts.push(`L1 ${fmt(n.l1_sol,1)}%`);
    if (n.occupancy!=null) parts.push(`occ ${fmt(n.occupancy,1)}%`);
    if (n.regs_per_thread!=null) parts.push(`regs ${fmt(n.regs_per_thread,0)}`);
    ncu = `<div class="ncu">NCU: ${parts.join(" · ")}</div>`;
  }
  const statusBadge = {
    ok:"✅ ok",
    empty_response:"🚫 LLM empty",
    no_code_block:"🚫 no-code",
    request_error:"🔌 req-error",
    verify_failed:"❌ verify-fail",
    generated_no_valid_kernel:"⚠️ gen-fail",
    no_diagnosis:"— pending",
  }[r.status] || (r.verified?"✅":"❌");
  const failDetail = r.failure ? `<div class="ncu" style="color:#f85149">${esc(r.failure.detail||"")}</div>` : "";
  // This round's own measured time vs the reverted best-so-far, so repeated
  // "best" values don't hide that each round tried a new (slower) kernel.
  const rr = r.round_result;
  let thisRound = "—";
  if (rr && rr.new_time_ms != null) {
    const rev = rr.reverted ? ' <span class="chg">↩reverted</span>' : "";
    thisRound = `${fmt(rr.new_time_ms)}${rev}`;
  }
  const bestSoFar = rr && rr.best_so_far_ms != null ? fmt(rr.best_so_far_ms) : fmt(r.time_ms);
  const attempts = ncu + failDetail + diag +
    det("💭 thinking", r.reasoning) +
    det("kernel", r.kernel_code) + det("prompt", r.opt_prompt) +
    det("LLM reply", r.opt_reply) + det("reflexion", r.reflexion);
  return `<tr class="${cls}"><td>${r.round}</td><td>${thisRound}</td><td>${bestSoFar}</td>
    <td>${fmt(r.speedup_vs_baseline,2)}×</td><td>${fmt(r.improvement_pct,1)}</td>
    <td>${fmt(r.ncu?r.ncu.tensor_sol:r.combined_sol_pct,1)}</td><td>${r.bottleneck||"—"} ${tma}</td>
    <td>${cfg}</td><td>${statusBadge}</td><td>${attempts}</td></tr>`;
}

function drawChart(baseMs, rounds) {
  const c = document.getElementById("chart"); if (!c) return;
  const dpr = window.devicePixelRatio||1, W=c.clientWidth, H=240;
  c.width=W*dpr; c.height=H*dpr; const g=c.getContext("2d"); g.scale(dpr,dpr);
  const pts = rounds.filter(r=>r.time_ms).map(r=>({x:r.round,y:r.time_ms,best:r.is_best}));
  const ys = [baseMs,...pts.map(p=>p.y)].filter(v=>v); if(!ys.length) return;
  const xs = [0,...pts.map(p=>p.x)];
  const maxY=Math.max(...ys)*1.1, minY=0, maxX=Math.max(...xs,1), pad=40;
  const X=x=>pad+(W-2*pad)*(x/maxX), Y=y=>H-pad-(H-2*pad)*((y-minY)/(maxY-minY));
  g.strokeStyle="#30363d"; g.fillStyle="#8b949e"; g.font="10px monospace";
  for(let i=0;i<=4;i++){const v=maxY*i/4,yy=Y(v);g.beginPath();g.moveTo(pad,yy);
    g.lineTo(W-pad,yy);g.stroke();g.fillText(v.toFixed(1),4,yy+3);}
  if(baseMs){g.strokeStyle="#f85149";g.setLineDash([4,4]);g.beginPath();
    g.moveTo(pad,Y(baseMs));g.lineTo(W-pad,Y(baseMs));g.stroke();g.setLineDash([]);
    g.fillStyle="#f85149";g.fillText("baseline",W-pad-50,Y(baseMs)-4);}
  g.strokeStyle="#1f6feb";g.lineWidth=2;g.beginPath();
  pts.forEach((p,i)=>{const xx=X(p.x),yy=Y(p.y);i?g.lineTo(xx,yy):g.moveTo(xx,yy);});
  g.stroke();
  pts.forEach(p=>{g.fillStyle=p.best?"#238636":"#1f6feb";g.beginPath();
    g.arc(X(p.x),Y(p.y),p.best?5:3,0,7);g.fill();
    g.fillStyle="#8b949e";g.fillText("r"+p.x,X(p.x)-6,H-pad+14);});
}

const esc = s => (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
loadRuns(); setInterval(()=>{ loadRuns(); if(SEL) selectRun(SEL); }, 3000);
</script>
</body></html>"""


def build_app(root: Path):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="KernelAgent Optimization Dashboard")
    root = Path(root).resolve()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/runs")
    def api_runs() -> JSONResponse:
        return JSONResponse(discover_runs(root))

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> JSONResponse:
        try:
            return JSONResponse(load_run(root, run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".", help="Directory to scan for trajectory.jsonl files"
    )
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    app = build_app(Path(args.root))
    print(f"Optimization dashboard: http://{args.host}:{args.port}  (root={Path(args.root).resolve()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
