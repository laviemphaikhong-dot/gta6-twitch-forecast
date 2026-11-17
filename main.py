"""Forecast a game launch day's Twitch viewers, hour by hour.

Written for make.fun's "Forecast GTA VI's launch-day Twitch viewers"
competition. The program runs once per case in an offline sandbox:
it reads the case file(s) in the input directory and writes, for each
case, the five quantiles (5/25/50/75/95%) of the viewer count of every
hour from hour 2 to hour 24 of release day.

Idea
----
The naive guess "every hour equals hour 1" ignores the shape of a
launch day: a console game unlocks at midnight local time, region by
region, so the audience keeps building as Europe and then the US come
online, peaks in the transatlantic evening and fades late at night.
Every case carries the full hourly history of dozens of past big
launches ("comparables"). For each comparable we measure how its own
release day evolved relative to its own first hour, and we pool those
log-ratios per hour of day. The median ratio is the typical curve; the
spread of the ratios (a robust IQR-based sigma) is the honest
uncertainty.

The pool is weighted, not filtered: a launch that resembles the target
(console-only, rolling midnight-local unlock - the family the scored
games belong to) counts four times, a mixed launch once. A
leave-one-out replay over the launches in the practice data, scored
with the contest's own pinball loss, prefers this weighting and a
near-unshrunk median; on the console-rolling launches it scores about
a quarter better than the flat "every hour equals hour 1" forecast.
The pooled sigma is still clipped, so a handful of wild launches
cannot make the bands absurd.

Everything is deterministic, standard library only, and fast: a case
holds a few dozen comparables of a few hundred hourly rows each.
"""

import json
import math
import os
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles

# Model knobs (tuned on a leave-one-out replay of past launches).
MIN_SAMPLES = 8     # pooled ratios needed before we trust an hour
SHRINK = 0.95       # pull the pooled median this far toward 1.0x
SIGMA_MIN = 0.28    # never claim to be surer than this (log units)
SIGMA_MAX = 1.2     # never claim to be less sure than this
BASE_SIGMA = 0.15   # fallback spread growth, per sqrt(hour - 1)
W_CONSOLE = 2       # pool weight of a console-only launch
W_ROLLING = 2       # pool weight of a rolling midnight-local unlock


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile(sorted_values, q):
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def release_day_hours(comparable):
    """Map hour-of-release-day (1..24) -> viewers for one comparable.

    The comparable's hourly rows span two weeks before its release day
    to three days after; only the rows whose date matches its own
    day_start belong to the release day itself.
    """
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


def pool_weight(comparable):
    weight = 1
    if is_console_only(comparable):
        weight *= W_CONSOLE
    if is_rolling_unlock(comparable):
        weight *= W_ROLLING
    return weight


def pooled_log_ratios(comparables):
    """Per forecast hour, the weighted list of ln(v_h / v_1)."""
    pool = {h: [] for h in range(2, 25)}
    for comparable in comparables:
        hours = release_day_hours(comparable)
        first = hours.get(1)
        if not first or first <= 0 or len(hours) < 20:
            continue
        weight = pool_weight(comparable)
        for h in range(2, 25):
            value = hours.get(h)
            if value and value > 0:
                pool[h].extend([math.log(value / first)] * weight)
    return pool


def hour_curve(pool, h):
    """(median log ratio, sigma) for one hour, with safe fallbacks."""
    samples = sorted(pool.get(h) or [])
    fallback_sigma = BASE_SIGMA * math.sqrt(h - 1)
    if len(samples) < MIN_SAMPLES:
        return 0.0, fallback_sigma
    centre = median(samples)
    spread = (quantile(samples, 0.75) - quantile(samples, 0.25)) / 1.349
    sigma = min(max(spread, SIGMA_MIN), SIGMA_MAX)
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
    p1 = max(1.0, float(known[1]["viewers"]))
    pool = pooled_log_ratios(case.get("comparables") or [])
    rows = []
    for target in case["forecast"]:
        h = int(target["hour"])
        centre, sigma = hour_curve(pool, h)
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
