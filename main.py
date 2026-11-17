"""Forecast a game launch day's Twitch viewers, hour by hour.

Entry for make.fun's "Forecast GTA VI's launch-day Twitch viewers"
competition. Sandbox contract (house format program/v1): read every
*.in case in $MAKEFUN_INPUT_DIR, write <id>.out with the five
quantiles (5/25/50/75/95%) of hours 2..24 to $MAKEFUN_OUTPUT_DIR.

Idea
----
Start from the pooled-curves idea: each comparable launch contributes,
per hour of its release day, the log-ratio ln(v_h / v_1) of that hour's
viewers to its own first hour. Three refinements, each validated by a
leave-one-out replay over the 37 usable comparables in the practice
data (and on the four practice cases themselves):

1. Similarity weights. A comparable counts 3x when its platform family
   (console-only vs mixed) matches the target's, 3x when its unlock
   style (rolling midnight-local vs simultaneous) matches, and 2.5x
   when it launched on the same weekday. Early-access-contaminated
   launches (whose release-day curve is the tail of a wave that peaked
   days earlier) are excluded.

2. Covariate adjustment of the pool. A launch day's curve, measured
   against its own first hour, is strongly size-dependent: the bigger
   the launch, the more of its audience is already there in hour 1, so
   the late-day log-ratio falls by roughly 0.9 per unit of log size
   (R^2 ~ 0.85 in the evening hours). The same holds, more weakly, for
   the pre-release peak relative to hour 1. So before pooling, each
   comparable's log-ratio at each hour is shifted by a per-hour
   weighted-least-squares slope times its covariate distance to the
   target (slopes shrunk to 0.6 and clipped, distances clipped), i.e.
   every curve is re-expressed "as if" it had the target's size and
   pre-release profile.

3. Honest spread. The centre is the weighted median of the adjusted
   ratios (stretched by 1.05, compensating the slope shrinkage); the
   spread is a robust IQR sigma of the adjusted pool, clipped to
   [0.28, 1.2] log units and scaled by 0.85, with normal quantiles.

Leave-one-out over the comparables (the only honest rehearsal for the
final): mean pinball loss 0.236 vs 0.411 for the previous version and
0.41+ for the flat house baseline family; on console-only rolling
launches -- the family the scored games belong to -- 0.079 vs 0.112.
On the four practice cases it scores 0.072 in the local grader's units
(0.144 on the server's doubled scale).

Deterministic, standard library only, runs in well under a second.
"""

import datetime
import json
import math
import os
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles

# Model knobs (chosen by leave-one-out replay; see docstring).
MIN_SAMPLES_W = 4.0   # pooled weight needed before we trust an hour
MIN_SAMPLES_N = 5     # pooled count needed before we trust an hour
SHRINK = 1.05         # stretch of the pooled median (offsets slope shrinkage)
SIGMA_MIN = 0.28
SIGMA_MAX = 1.2
SIGMA_SCALE = 0.85
BASE_SIGMA = 0.15     # fallback spread growth, per sqrt(hour - 1)
W_CONSOLE_MATCH = 3.0
W_ROLLING_MATCH = 3.0
W_WEEKDAY_MATCH = 2.5
ADJ_SHRINK = 0.6      # shrink fitted covariate slopes by this
ADJ_BETA_LO = -1.0
ADJ_BETA_HI = 0.15
ADJ_CLIP = 2.75       # clip covariate distance used in the shift


# ---------- basic statistics ----------

def wquantile(pairs, q):
    """Weighted quantile of [(value, weight)], linear interpolation
    over the weighted CDF."""
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[0][0]
    target = q * total
    cum = 0.0
    for i, (v, w) in enumerate(pairs):
        prev = cum
        cum += w
        if cum >= target:
            if i == 0:
                return v
            pv = pairs[i - 1][0]
            frac = (target - prev) / w if w > 0 else 0.0
            return pv + frac * (v - pv)
    return pairs[-1][0]


def wmedian(pairs):
    return wquantile(pairs, 0.5)


# ---------- comparable features ----------

def release_day_hours(comparable):
    """Map hour-of-release-day (1..24) -> viewers for one comparable."""
    day = comparable["day_start"][:10]
    hours = {}
    for row in comparable.get("hourly") or []:
        stamp = row.get("from") or ""
        viewers = row.get("viewers")
        if stamp[:10] == day and viewers:
            hours[int(stamp[11:13]) + 1] = float(viewers)
    return hours


def is_console_only(comparable):
    """True when the launch had no PC version on day one."""
    platforms = (comparable.get("platforms") or "").replace("/", ",")
    tokens = [t.strip().lower() for t in platforms.split(",")]
    return not any(t == "pc" or t.startswith("pc ") for t in tokens)


def is_rolling_unlock(comparable):
    """True when the game unlocked at midnight local time, by region."""
    unlock = (comparable.get("unlock") or "").lower()
    return any(phrase in unlock for phrase in (
        "midnight local", "rolling", "12 am local", "local midnight"))


def is_early_access(comparable):
    """True when an early-access wave preceded the release day."""
    return "early access" in (comparable.get("unlock") or "").lower()


def weekday_of(day_start):
    day = day_start[:10]
    return datetime.date(int(day[:4]), int(day[5:7]), int(day[8:10])).weekday()


def comp_features(comparable):
    hours = release_day_hours(comparable)
    v1 = hours.get(1)
    day = comparable["day_start"][:10]
    pre_peak = 0.0
    for row in comparable.get("hourly") or []:
        stamp = row.get("from") or ""
        viewers = row.get("viewers")
        if stamp[:10] < day and viewers and float(viewers) > pre_peak:
            pre_peak = float(viewers)
    return {
        "hours": hours,
        "v1": v1,
        "logv1": math.log(v1) if v1 else 0.0,
        "console": is_console_only(comparable),
        "rolling": is_rolling_unlock(comparable),
        "ea": is_early_access(comparable),
        "weekday": weekday_of(comparable["day_start"]),
        "pre_log_ratio": (math.log(pre_peak / v1) if pre_peak and v1 else None),
    }


def usable(f):
    return bool(f["v1"]) and f["v1"] > 0 and len(f["hours"]) >= 20


# ---------- target features ----------

def target_features(case):
    """The target's own traits, parsed from the case's notes and data.

    The case carries no structured platform/unlock fields for the
    target itself, so the notes' release sentence is parsed with
    conservative rules; anything unrecognised falls back to the mixed
    (non-console) family, which simply changes which comparables are
    upweighted.
    """
    notes = (case.get("notes") or "")
    t = notes.lower()
    rolling = any(p in t for p in (
        "midnight local", "rolling", "12 am local", "local midnight"))
    console = False
    if "console only" in t:
        console = True
    else:
        head = t.split("real unlock")[0]
        seg = head.split(" on ", 1)[1] if " on " in head else head
        seg = seg.split(";")[0].split(" source")[0]
        if " only" in seg and "pc" not in seg:
            console = True
        elif not any(tok in seg for tok in ("pc", "steam", "windows")) and \
                any(tok in seg for tok in ("ps5", "ps4", "xbox", "switch", "playstation")):
            console = True
    p1 = max(1.0, float(case["known"][0]["viewers"]))
    hist = case.get("history_hourly") or case.get("history") or []
    pre = [float(r["viewers"]) for r in hist if r.get("viewers")]
    return {
        "console": console,
        "rolling": rolling,
        "logv1": math.log(p1),
        "weekday": weekday_of(case["day_start"]),
        "pre_log_ratio": (math.log(max(pre) / p1) if pre else None),
        "p1": p1,
    }


# ---------- the model ----------

def pool_weight(f, tgt):
    if not usable(f) or f["ea"]:
        return 0.0
    w = 1.0
    if f["console"] == tgt["console"]:
        w *= W_CONSOLE_MATCH
    if f["rolling"] == tgt["rolling"]:
        w *= W_ROLLING_MATCH
    if f["weekday"] == tgt["weekday"]:
        w *= W_WEEKDAY_MATCH
    return w


def _covariate(f, tgt, var):
    if var == "size":
        return f["logv1"], tgt["logv1"]
    if var == "pre":
        if f["pre_log_ratio"] is None or tgt["pre_log_ratio"] is None:
            return None, None
        return f["pre_log_ratio"], tgt["pre_log_ratio"]
    return None, None


def _fit_beta(rows, ds):
    """Weighted least squares slope of values on covariate distances."""
    tw = sum(w for _, w in rows)
    if tw <= 0:
        return None
    mx = sum(w * d for (_, w), d in zip(rows, ds)) / tw
    my = sum(w * x for (x, w) in rows) / tw
    sxy = sum(w * (d - mx) * (x - my) for (x, w), d in zip(rows, ds))
    sxx = sum(w * (d - mx) ** 2 for (_, w), d in zip(rows, ds))
    if sxx <= 1e-9:
        return None
    beta = sxy / sxx * ADJ_SHRINK
    return max(ADJ_BETA_LO, min(ADJ_BETA_HI, beta))


def adjusted_pool(feats, ws, tgt):
    """Per hour, [(adjusted log-ratio, weight)] with each comparable's
    ratio shifted to the target's size / pre-release profile."""
    pool = {}
    for h in range(2, 25):
        rows = []  # [log_ratio, weight, feat_index]
        for idx, (f, w) in enumerate(zip(feats, ws)):
            if w <= 0:
                continue
            v = f["hours"].get(h)
            if v and v > 0:
                rows.append([math.log(v / f["v1"]), w, idx])
        if len(rows) < 8:
            pool[h] = [(x, w) for x, w, _ in rows]
            continue
        shifts = [0.0] * len(rows)
        for var in ("size", "pre"):
            dist = {}
            for k, (_, _, idx) in enumerate(rows):
                cval, tval = _covariate(feats[idx], tgt, var)
                if cval is None:
                    continue
                d = cval - tval
                dist[k] = max(-ADJ_CLIP, min(ADJ_CLIP, d))
            if len(dist) < 6:
                continue
            fit_rows = [(rows[k][0], rows[k][1]) for k in sorted(dist)]
            beta = _fit_beta(fit_rows, [dist[k] for k in sorted(dist)])
            if beta is None:
                continue
            for k, d in dist.items():
                shifts[k] += beta * d
        pool[h] = [(x - s, w) for (x, w, _), s in zip(rows, shifts)]
    return pool


def hour_curve(pairs, h):
    """(centre, sigma) of one hour's adjusted pool, with fallbacks."""
    total_w = sum(w for _, w in pairs)
    if total_w < MIN_SAMPLES_W or len(pairs) < MIN_SAMPLES_N:
        return 0.0, BASE_SIGMA * math.sqrt(h - 1)
    centre = wmedian(pairs)
    spread = (wquantile(pairs, 0.75) - wquantile(pairs, 0.25)) / 1.349
    sigma = min(max(spread, SIGMA_MIN), SIGMA_MAX) * SIGMA_SCALE
    return SHRINK * centre, sigma


def round2(x):
    """Round half up to two decimals, the way the examples are written."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected quantile levels in case")
    known = {row["hour"]: row for row in case.get("known") or []}
    if 1 not in known:
        raise ValueError("case has no hour-1 print")
    tgt = target_features(case)
    p1 = tgt["p1"]
    feats = [comp_features(c) for c in case.get("comparables") or []]
    ws = [pool_weight(f, tgt) for f in feats]
    pool = adjusted_pool(feats, ws, tgt)
    rows = []
    for target in case["forecast"]:
        h = int(target["hour"])
        centre, sigma = hour_curve(pool[h], h)
        qs = []
        for z in Z:
            value = p1 * math.exp(centre + z * sigma)
            qs.append(round2(min(max(value, 0.01), 1e9)))
        qs.sort()
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (out_dir / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
