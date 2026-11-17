"""gta6-twitch-ccu-forecast-01 entry: a comparables-ratio forecaster (Python 3.13, stdlib only).

Model, re-derived for v2. A launch day's first hour is a weak anchor by itself: the unlock
rolls at midnight local time (New Zealand first, then Europe, then the US coasts), so new
regions keep joining the audience for most of the UTC day and the curve's *shape* is the
predictable part. The case file carries dozens of past launches with full hourly curves.
For each past launch c I form the release-day log-ratios

    r_c(h) = ln( v_c(h) / v_c(1) ),   h = 2..24,

and forecast this case's hour h as its own hour-1 print times the exponential of a weighted
quantile of those ratios:

    q_tau(h) = p1 * exp( Q_tau( { r_c(h) } ) ).

Weights: a Gaussian kernel on the log-scale gap between the comparable's hour 1 and this
case's hour 1 (similar-scale launches count most), times mild resemblance factors for
console line-ups and midnight-local unlocks, and a mild discount for launches whose day
was distorted by paid early access. Each hour's empirical log-quantiles are then shrunk a
little toward the persistence forecast (ratio 1 with the baseline's random-walk spread),
with strength alpha = n_eff / (n_eff + K) from the hour's effective sample size, so a thin
or lopsided hour degrades gracefully instead of extrapolating noise.

What the leave-one-out backtest over the 38 practice comparables changed in v2 (and why):
the v1 shrinkage (K = 6) pulled late hours hard toward ratio 1, but real launch curves keep
growing into the evening unlock waves -- v1's median sat below the actual by +0.21..+0.35
log units over hours 18-24, and cutting K to 1.5 was worth ~0.10 of mean pinball loss in
LOO (0.544 -> 0.446) with the big-launch subset unchanged. The resemblance factors were
likewise softened (console 1.5 -> 1.25, midnight 1.25 -> 1.1, early-access 0.5 -> 0.75)
and the scale bandwidth narrowed (1.5 -> 1.25): at v1 strength the hand-set multipliers,
not the data, were deciding which curves counted.

Safety: output validity outranks forecast quality. The model path is wrapped so any
surprise (missing fields, unusable comparables) falls back to the persistence forecast,
which itself sits in front of a constant last resort. Every input file always produces an
output file of exactly the required shape: 23 rows, hours 2..24, five non-decreasing
quantiles each in (0, 1e9].

Run by the grader as `python3 -P -s /bot/main.py`; locally:
    MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, as the starter does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The persistence spread, in natural-log units per square-root hour (the house baseline's).
SIGMA = 0.15

# --- Model constants (v2, tuned by the leave-one-out backtest) ------------------------------
# Gaussian kernel bandwidth, in log units, for "similar scale to this case's hour 1".
SCALE_BANDWIDTH = 1.25
# Weight multipliers for launches resembling this one (softened from v1; see docstring).
CONSOLE_FACTOR = 1.25       # platforms include a console (this launch is console-only)
MIDNIGHT_FACTOR = 1.1       # unlock rolled at midnight local time, as this one does
EARLY_ACCESS_FACTOR = 0.75  # release day distorted by paid early access: trust it less
# Shrinkage: alpha = n_eff / (n_eff + SHRINK_K); K pseudo-observations of persistence.
SHRINK_K = 1.5
# Output bounds: every quantile must lie in (0, 1e9]; keep a small positive floor.
Q_FLOOR = 0.01
Q_CAP = 1_000_000_000.0


def round2(x: float):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round(x * 100) / 100).

    A whole number comes back as an int, so the JSON stays compact like the baseline's.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def parse_ts(text: str) -> float:
    """An ISO-8601 UTC timestamp ('...Z') to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


# --- Persistence forecast (the fallback; the house baseline's own rule) ----------------------

def persistence_rows(p1: float, hours: list) -> list:
    rows = []
    for h in hours:
        spread = SIGMA * math.sqrt(h - 1)
        rows.append({"hour": h, "q": [round2(p1 * math.exp(z * spread)) for z in Z]})
    return rows


# --- Comparables model -----------------------------------------------------------------------

def comparable_weight(comp: dict) -> float:
    """How much a past launch's curve should count for this case, before scale similarity."""
    w = 1.0
    platforms = " ".join(str(p) for p in comp.get("platforms") or [])
    if any(tag in platforms for tag in ("PS", "Xbox", "Switch")):
        w *= CONSOLE_FACTOR
    unlock = str(comp.get("unlock") or "").lower()
    if "midnight local" in unlock:
        w *= MIDNIGHT_FACTOR
    if "early-access-contaminated" in unlock or "early access" in unlock:
        w *= EARLY_ACCESS_FACTOR
    return w


def day_one_ratios(comp: dict):
    """(v1, {hour: log-ratio}) for a comparable's release day, or None if unusable.

    The release day is the 24 hourly points starting at the comparable's day_start; a point
    exactly one hour apart is expected, and any hour the source never sampled is simply absent.
    """
    try:
        t0 = parse_ts(comp["day_start"])
        by_time = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                by_time[parse_ts(point["from"])] = float(viewers)
        v1 = by_time.get(t0)
        if not v1:
            return None
        ratios = {}
        for h in range(2, 25):
            v = by_time.get(t0 + (h - 1) * 3600)
            if v:
                ratios[h] = math.log(v / v1)
        if not ratios:
            return None
        return v1, ratios
    except Exception:
        return None


def weighted_quantile(samples: list, tau: float) -> float:
    """The tau-quantile of [(value, weight)], linear on the midpoint empirical CDF."""
    ordered = sorted(samples)
    total = sum(w for _, w in ordered)
    if total <= 0:
        raise ValueError("empty sample")
    # Midpoint CDF positions: the i-th point sits at (cumw_i - w_i / 2) / total.
    points = []
    cum = 0.0
    for value, w in ordered:
        cum += w
        points.append((value, (cum - 0.5 * w) / total))
    if tau <= points[0][1]:
        return points[0][0]
    if tau >= points[-1][1]:
        return points[-1][0]
    for (x0, f0), (x1, f1) in zip(points, points[1:]):
        if f0 <= tau <= f1:
            if f1 == f0:
                return x1
            return x0 + (x1 - x0) * (tau - f0) / (f1 - f0)
    return points[-1][0]


def model_rows(case: dict, p1: float, hours: list) -> list:
    """The comparables-ratio forecast. Raises on anything unusable; the caller falls back."""
    game_name = str((case.get("game") or {}).get("name") or "")
    log_p1 = math.log(p1)
    per_hour = {h: [] for h in hours}
    for comp in case.get("comparables") or []:
        if str(comp.get("name") or "") == game_name:
            continue  # a case never forecasts from itself
        built = day_one_ratios(comp)
        if built is None:
            continue
        v1, ratios = built
        gap = (math.log(v1) - log_p1) / SCALE_BANDWIDTH
        w = comparable_weight(comp) * math.exp(-0.5 * gap * gap)
        if w <= 0:
            continue
        for h, r in ratios.items():
            if h in per_hour:
                per_hour[h].append((r, w))
    if not any(per_hour[h] for h in hours):
        raise ValueError("no comparable contributed a ratio")

    rows = []
    for h in hours:
        samples = per_hour[h]
        spread = SIGMA * math.sqrt(h - 1)
        if samples:
            total_w = sum(w for _, w in samples)
            sq_w = sum(w * w for _, w in samples)
            n_eff = (total_w * total_w / sq_w) if sq_w > 0 else 0.0
            alpha = n_eff / (n_eff + SHRINK_K) if n_eff > 0 else 0.0
        else:
            alpha = 0.0
        qs = []
        for tau, z in zip(LEVELS, Z):
            persistence_log = z * spread  # log-ratio the baseline implies at this level
            if alpha > 0.0:
                emp_log = weighted_quantile(samples, tau)
                log_q = alpha * emp_log + (1.0 - alpha) * persistence_log
            else:
                log_q = persistence_log
            qs.append(p1 * math.exp(log_q))
        rows.append({"hour": h, "q": clean_quantiles(qs)})
    return rows


def clean_quantiles(qs: list) -> list:
    """Force the row into the valid region: clamp to (0, 1e9], non-decreasing, rounded."""
    out = []
    prev = Q_FLOOR
    for q in qs:
        q = min(max(float(q), Q_FLOOR), Q_CAP)
        q = max(q, prev)
        out.append(round2(q))
        prev = float(out[-1])
    return out


# --- Case handling ---------------------------------------------------------------------------

def first_print(case: dict) -> float:
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    return max(1.0, float(first["viewers"]))


def forecast_case(case: dict) -> dict:
    hours = [row["hour"] for row in case.get("forecast") or []]
    if sorted(hours) != list(range(2, 25)):
        hours = list(range(2, 25))
    try:
        p1 = first_print(case)
    except Exception:
        # No readable hour-1 print: the only valid answer left is a constant file.
        return {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in hours]}
    try:
        return {"hours": model_rows(case, p1, hours)}
    except Exception:
        return {"hours": persistence_rows(p1, hours)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
