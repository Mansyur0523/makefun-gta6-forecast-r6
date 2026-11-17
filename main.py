#!/usr/bin/env python3
"""gta6-twitch-ccu-forecast-01 -- round 6 entry.

Forecasts the Twitch concurrent-viewer curve of a big game launch's release
day (hours 2..24, five quantiles an hour) from everything recorded up to the
end of hour 1. Python 3 standard library only, no network, deterministic.

Model (built from scratch for this entry)
----------------------------------------
Work in log-ratios to hour 1:  lr_h = ln(viewers_h / viewers_1).

1. Per-hour ridge.  For every comparable past launch in the case file we
   extract its release-day curve.  A ridge regression (lambda 10) is fit per
   hour on eight features of the launch -- log hour-1 viewers, log hour-1
   channels, log(eve peak / hour 1), the slope into hour 1, the viewers at
   midnight 48 h and 72 h before release relative to hour 1, and log(hour-1
   channels / eve peak channels) -- and evaluated at the target launch.
   The 48 h / 72 h features are the "pre-release maturity" signal: a
   category that already had a real audience days before release has less
   headroom left on the day itself, so its day runs flatter relative to
   hour 1; a category that was dead until the last day ramps harder.
   (A missing pre-release point counts as a dead category, -7 in log.)

2. Shape / level split.  The ridge curve's level (its mean over hours
   2..24) is kept, but its shape is blended with the mean shape of the
   console-only comparables (console launches rise more gently than the
   PC-led average).  The blend weight depends on how far the eve peak
   spiked above hour 1: w = clip(0.5 - 0.8 * (logever - 0.6), 0, 0.8), so a
   launch whose eve hype collapsed hard into hour 1 keeps more of the
   ridge's steep shape.

3. Unlock bumps.  Two smooth bumps are added to the median curve, both
   tied to the release day's unlock clock: +0.5 * exp(-((h-6)/2)^2) for the
   US unlock wave (US Eastern midnight is 05:00Z, Pacific 08:00Z) and
   +0.1 * exp(-((h-20)/3)^2) for the US evening prime-time peak.

4. Spread.  A fixed log-spread of 0.22 around the median, with the
   standard-normal quantiles (-1.645, -0.674, 0, 0.674, 1.645).  On the
   practice set this gives a 90% coverage of about 0.90.

Target history: the target's own pre-release hours are taken from
history_hourly.  On the final case the recorder only starts two days
before release, so hours older than a day are filled in from the
comparable carrying the target game's own category history (same Twitch
category), which the case file ships as one of the comparables.  Practice
cases have a full 14-day history, so the fallback never triggers there.

Fallbacks keep the output valid on degenerate input: comparables without a
complete release day are skipped; with no usable comparable at all the
median is flat at hour 1 (persistence) with a wider spread.

Run:  python3 main.py
  reads  $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
  (default /output), one <id>.out per <id>.in.  About 0.1 s per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SPREAD = 0.22
RIDGE_LAMBDA = 10.0
DEAD_LOG = -7.0  # log-ratio used when a pre-release hour has no point

# Unlock bumps (median log-ratio added at hour h).
SURGE_AMP, SURGE_CENTER, SURGE_WIDTH = 0.5, 6.0, 2.0
PRIME_AMP, PRIME_CENTER, PRIME_WIDTH = 0.1, 20.0, 3.0
# Shape-blend weight w = clip(W0 - W_SLOPE * (logever - W_REF), 0, W_MAX).
W0, W_SLOPE, W_REF, W_MAX = 0.5, 0.8, 0.6, 0.8


def parse_ts(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hour_offset(moment, day_start):
    """Whole hours from day_start to moment (may be negative)."""
    return int((moment - day_start).total_seconds() // 3600)


def solve_ridge(rows, targets, weights, lam):
    """Ridge regression by Gaussian elimination on the normal equations.

    rows: list of feature vectors (intercept included).  The intercept is
    not penalised; every other coefficient gets +lam on the diagonal.
    """
    n = len(rows[0])
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for x, y, w in zip(rows, targets, weights):
        for i in range(n):
            b[i] += w * x[i] * y
            for j in range(n):
                a[i][j] += w * x[i] * x[j]
    for i in range(n):
        a[i][i] += lam if i > 0 else 1e-9
    for i in range(n):
        piv = a[i][i]
        for j in range(i + 1, n):
            f = a[j][i] / piv
            for k in range(i, n):
                a[j][k] -= f * a[i][k]
            b[j] -= f * b[i]
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sol[i] = (b[i] - sum(a[i][j] * sol[j] for j in range(i + 1, n))) / a[i][i]
    return sol


def hourly_map(points, day_start, key_from="from"):
    """{hour offset from day_start: (viewers, channels)} for hourly points."""
    out = {}
    for p in points or []:
        try:
            dh = hour_offset(parse_ts(p[key_from]), day_start)
        except (KeyError, ValueError):
            continue
        viewers = p.get("viewers")
        if viewers is None:
            continue
        out[dh] = (viewers, p.get("channels") or 0)
    return out


def viewers_at(hours, offset, default=None):
    point = hours.get(offset)
    if point is None or not point[0]:
        return default
    return point[0]


def launch_features(hours, v1, c1):
    """The eight ridge features for one launch, from its hourly map.

    hours: {offset: (viewers, channels)} covering the eve and release day.
    v1/c1: hour-1 viewers/channels.  Offsets are hours from day_start:
    hour 1 is offset 0, the eve is offsets -24..-1.
    """
    eve_peak = 0
    eve_peak_channels = 0
    for off in range(-24, 0):
        point = hours.get(off)
        if point is None:
            continue
        if point[0] > eve_peak:
            eve_peak = point[0]
        if point[1] > eve_peak_channels:
            eve_peak_channels = point[1]
    logever = math.log(max(1, eve_peak) / max(1, v1))
    last_eve = viewers_at(hours, -1)
    slope = math.log(max(1, v1) / last_eve) if last_eve else 0.0

    def log_midnight(offset):
        value = viewers_at(hours, offset)
        return math.log(value / max(1, v1)) if value else DEAD_LOG

    chratio = math.log(max(1, c1) / max(1, eve_peak_channels))
    return [
        1.0,
        math.log(max(1, v1)),
        math.log(max(1, c1)),
        logever,
        slope,
        log_midnight(-48),
        log_midnight(-72),
        chratio,
    ]


def comparable_curves(case):
    """Extract every comparable launch with a complete release day.

    Returns a list of dicts: log-ratio curve lr[0..23] (hour 1..24),
    feature vector, console-only flag, and eve12 shape (unused by the
    final model, kept out).
    """
    out = []
    for comp in case.get("comparables", []) or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, ValueError):
            continue
        hours = hourly_map(comp.get("hourly", []), day_start)
        day = [hours.get(k) for k in range(0, 24)]
        if any(point is None for point in day):
            continue
        v1 = max(1, day[0][0])
        c1 = day[0][1] or 1
        day_viewers = [max(1, point[0]) for point in day]
        lr = [math.log(v / v1) for v in day_viewers]
        out.append(
            {
                "lr": lr,
                "features": launch_features(hours, v1, c1),
                "console_only": "PC" not in (comp.get("platforms") or ""),
            }
        )
    return out


def target_context(case):
    """Hour-1 print plus the target launch's features.

    The target's pre-release hours come from history_hourly; hours older
    than the recorded history are filled from the comparable that carries
    this game's own category history (same game name), when one exists.
    """
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    v1 = max(1, known["viewers"])
    c1 = known.get("channels") or 1
    day_start = parse_ts(case["day_start"])
    hours = hourly_map(case.get("history_hourly", []), day_start)
    # Fill old hours from the game's own pre-release series if present.
    if not any(off <= -25 for off in hours):
        name = (case.get("game") or {}).get("name", "").lower()
        for comp in case.get("comparables", []) or []:
            comp_name = (comp.get("name") or "").lower()
            if name and (name in comp_name or comp_name.startswith(name)):
                try:
                    comp_start = parse_ts(comp["day_start"])
                except (KeyError, ValueError):
                    continue
                old = hourly_map(comp.get("hourly", []), comp_start)
                for off, point in old.items():
                    if off <= -25 and off not in hours:
                        hours[off] = point
                break
    features = launch_features(hours, v1, c1)
    return v1, c1, features


def median_curve(comps, target_features):
    """Median log-ratio curve (hours 2..24) for the target launch."""
    if not comps:
        return [0.0] * 23
    # 1. Per-hour ridge on the eight features.
    ridge = []
    for h in range(1, 24):
        rows = [d["features"] for d in comps]
        targets = [d["lr"][h] for d in comps]
        w = solve_ridge(rows, targets, [1.0] * len(comps), RIDGE_LAMBDA)
        ridge.append(sum(a * b for a, b in zip(w, target_features)))
    # 2. Shape / level split against the console-only mean shape.
    level = sum(ridge) / 23
    ridge_shape = [x - level for x in ridge]
    console = [d for d in comps if d["console_only"]]
    shape_pool = console if console else comps
    mean_curve = [sum(d["lr"][h] for d in shape_pool) / len(shape_pool) for h in range(1, 24)]
    mean_level = sum(mean_curve) / 23
    pool_shape = [x - mean_level for x in mean_curve]
    logever = target_features[3]
    w_shape = min(W_MAX, max(0.0, W0 - W_SLOPE * (logever - W_REF)))
    shape = [(1.0 - w_shape) * a + w_shape * b for a, b in zip(ridge_shape, pool_shape)]
    med = [level + s for s in shape]
    # 3. Unlock bumps: US unlock wave + US evening prime time.
    out = []
    for h, m in zip(range(2, 25), med):
        m += SURGE_AMP * math.exp(-((h - SURGE_CENTER) / SURGE_WIDTH) ** 2)
        m += PRIME_AMP * math.exp(-((h - PRIME_CENTER) / PRIME_WIDTH) ** 2)
        out.append(m)
    return out


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    v1, _c1, features = target_context(case)
    med = median_curve(comparable_curves(case), features)
    med_by_hour = {h: med[h - 2] for h in range(2, 25)}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        q = []
        for z in Z:
            value = v1 * math.exp(med_by_hour[h] + z * SPREAD)
            value = min(max(value, 1.0), 1e9)
            q.append(round(value, 2))
        # Rounding can only equalise neighbours; keep the row sorted.
        q.sort()
        hours.append({"hour": h, "q": q})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
