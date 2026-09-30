"""Static dashboard generator: reads the research memory, writes one self-contained HTML page
(host it on GitHub Pages / Vercel). No JS required."""
from __future__ import annotations
import html
import json
import os


def _summaries(mem) -> list:
    return [json.loads(r["detail"]) for r in
            mem.db.execute("SELECT detail FROM events WHERE kind='generation_summary' ORDER BY id")]


def _lineage_str(mem, sid) -> str:
    return " → ".join(f"#{i:03d}" for i in mem.lineage(sid))


def build_context(mem, cfg, limits) -> dict:
    sums = _summaries(mem)
    last = sums[-1] if sums else {}
    top = []
    for sid in last.get("top", []):
        ex = mem.db.execute("SELECT metrics, learning_efficiency, lesson FROM experiments "
                            "WHERE strategy_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        st = mem.get_strategy(sid)
        m = json.loads(ex["metrics"]) if ex else {}
        top.append({"id": sid, "family": st["family"], "stage": st["stage"], "lineage": _lineage_str(mem, sid),
                    "ret": m.get("val_return", 0.0), "dd": m.get("val_drawdown", 0.0),
                    "le": ex["learning_efficiency"] if ex else 0.0, "lesson": ex["lesson"] if ex else ""})
    hyp = mem.db.execute("SELECT text FROM hypotheses ORDER BY id DESC LIMIT 1").fetchone()
    st = mem.db.execute("SELECT novelty, reason, regime FROM states ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "counts": mem.counts(), "last": last, "top": top,
        "mutations": sum(s["mutations"] for s in sums), "crossovers": sum(s["crossovers"] for s in sums),
        "discarded": sum(s["discarded"] for s in sums), "generation": last.get("generation", -1) + 1,
        "state": dict(st) if st else {}, "hypothesis": hyp["text"] if hyp else "-",
        "risk_budget": last.get("risk_budget", 1.0), "limits": limits, "cfg": cfg,
    }


def render(ctx: dict) -> str:
    e = html.escape
    st, last = ctx["state"], ctx["last"]
    rows = "".join(
        f"<tr><td>#{t['id']:03d}</td><td>{e(t['family'])}</td><td>{e(t['stage'])}</td>"
        f"<td>{t['ret']:+.1%}</td><td>{t['dd']:.1%}</td><td>{t['le']:.1f}</td><td class=l>{e(t['lineage'])}</td></tr>"
        for t in ctx["top"]) or "<tr><td colspan=7>no data yet</td></tr>"
    L = ctx["limits"]
    lesson = e(ctx["top"][0]["lesson"]) if ctx["top"] else "-"
    return f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>RUFFLUX Evolution Lab</title>
<style>
:root{{--bg:#1a0a1f;--card:#2a1233;--pink:#ff5fb8;--soft:#ffb3dd;--txt:#f6e4f1}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 10%,#3a1546,var(--bg));color:var(--txt);
font:15px/1.5 ui-monospace,Menlo,monospace;padding:20px;max-width:980px;margin:auto}}
h1{{color:var(--pink);letter-spacing:.2em;margin:0}}h2{{color:var(--soft);font-size:13px;letter-spacing:.15em;margin:0 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin:16px 0}}
.card{{background:var(--card);border:1px solid #5a2a6b;border-radius:14px;padding:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:4px 6px;text-align:left;border-bottom:1px solid #4a2058}}
.l{{color:var(--soft)}}.blob{{width:120px;height:110px;margin:6px auto 10px;border-radius:52% 48% 55% 45%/55% 50% 50% 45%;
background:radial-gradient(circle at 50% 50%,#fff 0 12%,var(--pink) 45%,#b02a80 100%);box-shadow:0 0 30px var(--pink);
position:relative;border:2px solid #fff}}
.eye{{position:absolute;left:34px;top:32px;width:50px;height:46px;border-radius:50%;background:#fff}}
.eye i{{position:absolute;left:16px;top:14px;width:20px;height:20px;border-radius:50%;background:#c0157f}}
.dot{{color:var(--pink)}}.dim{{opacity:.6}}
</style>
<h1>RUFFLUX</h1><div class=dim>EVOLUTION LAB · V0.1 · simulation only · paper trading not connected</div>
<div class=dim>{e(ctx.get('banner',''))}</div>
<div class=grid>
<div class=card><h2>RUFFLUXY</h2><div class=blob><div class=eye><i></i></div></div>
<div><span class=dot>●</span> observing &nbsp;<span class=dot>●</span> thinking<br><span class=dot>●</span> experimenting &nbsp;<span class=dot>●</span> learning</div></div>
<div class=card><h2>ENVIRONMENT</h2>Market state regime: {e(str(st.get('regime','-')))}<br>Novelty: {st.get('novelty',0):.2f}<br>
<span class=dim>{e(str(st.get('reason','')))}</span></div>
<div class=card><h2>CURRENT GENERATION</h2>Generation: {ctx['generation']}<br>Strategies: {last.get('strategies','-')}<br>
Survivors: {last.get('survivors','-')}<br>Experiments: {last.get('experiments','-')}</div>
<div class=card><h2>EVOLUTION</h2>Mutations: {ctx['mutations']}<br>Crossovers: {ctx['crossovers']}<br>Discarded: {ctx['discarded']}<br>
Holdout peeks: {ctx['counts']['holdout_peeks']}</div>
<div class=card><h2>SYSTEM HEALTH</h2>Capital: simulated<br>Risk (learning) budget: {ctx['risk_budget']:.0%}<br>
Hard limits: max DD {L.max_drawdown:.0%} · daily loss {L.max_daily_loss:.0%} · pos {L.max_position_frac:.0%}<br>Active experiments: {last.get('experiments','-')}</div>
</div>
<div class=card><h2>TOP STRATEGIES (out-of-sample, latest generation)</h2>
<table><tr><th>ID<th>Family<th>Stage<th>Perf<th>Risk<th>Learn-eff<th>Lineage</tr>{rows}</table></div>
<div class=grid><div class=card><h2>LATEST OBSERVATION</h2>{e(str(st.get('reason','-')))}</div>
<div class=card><h2>LATEST HYPOTHESIS</h2>{e(ctx['hypothesis'])}<br><span class=dim>Latest lesson: {lesson}</span></div></div>
</html>"""


def write_dashboard(mem, cfg, limits, path: str, banner: str = "") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ctx = build_context(mem, cfg, limits)
    ctx["banner"] = banner
    with open(path, "w") as f:
        f.write(render(ctx))


def write_index(results: dict, gov, out_dir: str) -> None:
    """Landing page: halt status + one line per symbol."""
    e = html.escape
    os.makedirs(out_dir, exist_ok=True)
    status = (f"<b style='color:#ff5f5f'>HALTED</b>: {e(', '.join(gov.halt_reasons))}" if gov.halted
              else "running (simulation only; no orders placed)")
    lines = "".join(
        f"<li><a href='{e(s)}.html'>{e(s)}</a> — " +
        (f"gen {r['generation']} · novelty {r['novelty']:.2f} · {e(r['regime'])} · survivors {r['survivors']}" if "generation" in r
         else e(str(r.get("error") or r.get("skipped") or "")) ) + "</li>"
        for s, r in results.items())
    page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>RUFFLUX</title><body style='background:#1a0a1f;color:#f6e4f1;font:16px ui-monospace,monospace;padding:24px'>"
            f"<h1 style='color:#ff5fb8;letter-spacing:.2em'>RUFFLUX</h1><p>System: {status}</p><ul>{lines}</ul></body>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)
