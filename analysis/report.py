"""Turn a run directory into one HTML page you can actually act on.

    python -m analysis.report runs/baseline

Writes `report.html` next to `results.jsonl`, plus a rendered replay for every
game it links (losses and draws by default — those are the ones worth watching).
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from analysis import replay as replay_mod
from analysis import stats as stats_mod
from analysis import viewer
from arena import rating

CAUSE_HINTS = {
    "early_rush": "defend_margin / defend_horizon / first_expand_turn",
    "out_expanded": "expand weights; castle_gather_min_army too greedy",
    "out_gathered": "castle economy; gather_start_turn",
    "blundered": "we were ahead and lost the general — defence, deathtouch guard",
    "timeout": "time_budget_ms, or the turn loop got slow",
    "draw": "deathtouch_prep_turn, attack_margin too timid",
}


def _bar(label: str, n: int, total: int, colour: str) -> str:
    pct = 100.0 * n / max(total, 1)
    hint = CAUSE_HINTS.get(label, "")
    return (f'<tr><td>{html.escape(label)}</td><td class=num>{n}</td>'
            f'<td class=barcell><span class=bar style="width:{pct:.1f}%;background:{colour}"></span></td>'
            f'<td class=hint>{html.escape(hint)}</td></tr>')


def _curve_svg(curve: list[list[float]]) -> str:
    if len(curve) < 2:
        return ""
    max_land = max(max(r[1], r[2]) for r in curve) or 1
    max_army = max(max(r[3], r[4]) for r in curve) or 1
    n = len(curve)

    def path(idx: int, scale: float) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{i / (n - 1) * 600:.1f} {158 - r[idx] / scale * 150:.1f}"
            for i, r in enumerate(curve))
    return (
        '<svg viewBox="0 0 600 160" preserveAspectRatio="none" class=curve>'
        f'<path d="{path(1, max_land)}" fill="none" stroke="#ff6b6b" stroke-width="2"></path>'
        f'<path d="{path(2, max_land)}" fill="none" stroke="#4dabf7" stroke-width="2"></path>'
        f'<path d="{path(3, max_army)}" fill="none" stroke="#ff6b6b" stroke-width=1 stroke-dasharray="4 3"></path>'
        f'<path d="{path(4, max_army)}" fill="none" stroke="#4dabf7" stroke-width=1 stroke-dasharray="4 3"></path>'
        '</svg>'
        f'<div class=muted>land solid / army dashed &middot; us red, them blue &middot; '
        f'peak land {max_land:.0f}, peak army {max_army:.0f}</div>')


def build(run_dir: str | Path, render_all: bool = False, every: int = 2) -> Path:
    run = Path(run_dir)
    results = stats_mod.load_results(run)
    agg = stats_mod.aggregate(results)
    summ = rating.summary(agg["wins"], agg["draws"], agg["losses"])
    test = rating.sprt(agg["wins"], agg["draws"], agg["losses"])

    a_spec = next((r["spec0"] if r["a_seat"] == 0 else r["spec1"]) for r in results)
    b_spec = next((r["spec1"] if r["a_seat"] == 0 else r["spec0"]) for r in results)

    rows = []
    for i, r in enumerate(results):
        cause = stats_mod.classify(r)
        link = ""
        path = r.get("replay_path")
        if path and (render_all or cause not in ("win",)):
            rep_file = run / path
            if rep_file.exists():
                out_html = rep_file.with_suffix(".html")
                if not out_html.exists():
                    viewer.render(replay_mod.load(rep_file), out_html, every)
                link = f'<a href="{html.escape(str(out_html.relative_to(run)))}">watch</a>'
        a = r["a_seat"]
        rows.append(
            f'<tr class="r-{cause}"><td class=num>{i}</td><td class=num>{r["seed"]}</td>'
            f'<td>{r["h"]}x{r["w"]}</td><td>{"A=p0" if a == 0 else "A=p1"}</td>'
            f'<td class="res {r["a_result"]}">{r["a_result"]}</td><td>{cause}</td>'
            f'<td class=num>{r["turns"]}</td>'
            f'<td class=num>{r["land"][a]}/{r["land"][1 - a]}</td>'
            f'<td class=num>{r["castles"][a]}</td>'
            f'<td class=num>{r["max_ms"][a]:.0f}</td><td>{link}</td></tr>')

    causes = agg["causes"]
    total = agg["games"]
    colours = {"win": "#37b24d", "draw": "#868e96", "early_rush": "#f76707",
               "out_expanded": "#e03131", "out_gathered": "#d6336c",
               "blundered": "#ae3ec9", "timeout": "#f59f00"}
    cause_rows = "".join(_bar(k, causes[k], total, colours.get(k, "#495057"))
                         for k in sorted(causes, key=lambda k: -causes[k]))

    sizes = "".join(
        f'<tr><td>{k}</td><td class=num>{v["wins"]}/{v["games"]}</td>'
        f'<td class=num>{v["rate"]*100:.0f}%</td></tr>'
        for k, v in agg["by_size"].items())

    first_castle = (f'{agg["first_castle_turn"]:.0f}' if agg["first_castle_turn"] else "never")

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>arena report — {html.escape(a_spec)} vs {html.escape(b_spec)}</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0 auto; max-width:1100px; padding:24px; background:#12141a; color:#e6e6e6;
        font:13px/1.5 ui-monospace,Menlo,monospace; }}
 h1 {{ font-size:17px; margin:0 0 4px }} h2 {{ font-size:13px; margin:26px 0 8px; color:#8b93a7;
       text-transform:uppercase; letter-spacing:.08em }}
 .muted {{ color:#8b93a7 }}
 .headline {{ font-size:26px; margin:10px 0 }}
 .verdict {{ display:inline-block; padding:3px 9px; border-radius:4px; background:#1b1f2a;
             border:1px solid #303646 }}
 table {{ border-collapse:collapse; width:100%; margin-top:6px }}
 th,td {{ padding:3px 8px; border-bottom:1px solid #1e222c; text-align:left }}
 th {{ color:#8b93a7; font-weight:600 }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums }}
 .barcell {{ width:45% }} .bar {{ display:inline-block; height:11px; border-radius:2px }}
 .hint {{ color:#6c7689; font-size:12px }}
 .win {{ color:#37b24d }} .loss {{ color:#e03131 }} .draw {{ color:#868e96 }}
 a {{ color:#4dabf7 }}
 svg.curve {{ width:100%; height:160px; background:#0d0f14; border:1px solid #262a35 }}
 .grid {{ display:flex; gap:32px; flex-wrap:wrap }} .grid>div {{ flex:1; min-width:260px }}
</style>
<h1>{html.escape(a_spec)} <span class=muted>vs</span> {html.escape(b_spec)}</h1>
<div class=muted>{total} games &middot; colours swapped every game &middot; run <code>{html.escape(str(run))}</code></div>
<div class=headline>{agg['wins']}W {agg['draws']}D {agg['losses']}L
  <span class=muted style="font-size:16px">&nbsp;score {summ['score']:.3f}</span></div>
<div>elo <b>{summ['elo']:+.1f}</b> <span class=muted>[{summ['elo_lo']:+.1f}, {summ['elo_hi']:+.1f}]</span>
 &middot; <span class=verdict>SPRT: {html.escape(test['verdict'])} (llr {test['llr']:+.2f},
 bounds [{test['lower']:.2f}, {test['upper']:.2f}])</span></div>

<h2>How games ended</h2>
<table><tr><th>outcome</th><th class=num>n</th><th></th><th>knob to look at</th></tr>{cause_rows}</table>

<h2>Average game</h2>
<div class=grid>
 <div><table>
  <tr><td>mean length</td><td class=num>{agg['mean_turns']:.0f} turns</td></tr>
  <tr><td>castles built</td><td class=num>{agg['castles']:.1f}</td></tr>
  <tr><td>first castle</td><td class=num>turn {first_castle}</td></tr>
  <tr><td>games with a castle</td><td class=num>{agg['castle_games']}/{total}</td></tr>
  <tr><td>mean move time</td><td class=num>{agg['mean_ms']:.2f} ms</td></tr>
  <tr><td>slowest move</td><td class=num>{agg['max_ms']:.1f} ms</td></tr>
  <tr><td>faults</td><td class=num>{agg['faults']}</td></tr>
 </table></div>
 <div><table><tr><th>board</th><th class=num>wins</th><th class=num>rate</th></tr>{sizes}</table></div>
</div>

<h2>Mean land and army</h2>
{_curve_svg(agg['curve'])}

<h2>Games</h2>
<table>
<tr><th class=num>#</th><th class=num>seed</th><th>size</th><th>seat</th><th>result</th>
    <th>cause</th><th class=num>turns</th><th class=num>land</th><th class=num>castles</th>
    <th class=num>ms</th><th></th></tr>
{''.join(rows)}
</table>
"""
    out = run / "report.html"
    out.write_text(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--all", action="store_true", help="render replays for wins too")
    ap.add_argument("--every", type=int, default=2, help="replay frame stride")
    args = ap.parse_args()
    print(build(args.run_dir, args.all, args.every))


if __name__ == "__main__":
    main()
