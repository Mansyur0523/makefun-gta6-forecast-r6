# makefun-gta6-forecast-r6

[![bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast-r6/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast-r6/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme)

Entry for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme)
bounty on [make.fun](https://make.fun): forecast Grand Theft Auto VI's
launch-day Twitch concurrent viewers, hour by hour (hours 2..24 of
2026-11-19 UTC), as five quantiles an hour, from everything recorded up to
the end of the first hour. Scored by mean pinball loss on the log scale.

## How it works

`main.py` (Python 3 standard library only, no network) reads one case file
and writes one forecast. The model, built from scratch for this entry:

1. **Per-hour ridge on comparable launches.** Every comparable past launch
   in the case file contributes its release-day curve, as log-ratios to its
   hour-1 print. A ridge regression (λ = 10) is fit per hour on eight
   features: log hour-1 viewers, log hour-1 channels, log(eve peak /
   hour 1), the slope into hour 1, the viewers at midnight 48 h and 72 h
   before release relative to hour 1, and log(hour-1 channels / eve peak
   channels).
2. **Pre-release maturity.** The 48 h / 72 h features are the key
   discriminator the earlier rounds missed: a category that already had a
   real audience days before release (warm) has less headroom left on the
   day itself and runs flatter relative to hour 1; a category that was dead
   until the last day ramps much harder. This is what separates launches
   with nearly identical hour-1 prints but opposite day corrections.
3. **Shape / level split.** The ridge curve's level is kept, but its shape
   is blended with the mean shape of the console-only comparables
   (console launches rise more gently than the PC-led average). The blend
   weight depends on how far the eve peak spiked above hour 1, so a launch
   whose eve hype collapsed hard into hour 1 keeps more of the ridge's
   steep shape.
4. **Unlock bumps.** Two smooth bumps on the median curve, tied to the
   release day's unlock clock: the US unlock wave (US Eastern midnight
   05:00Z, Pacific 08:00Z) and the US evening prime-time peak.
5. **Spread.** A fixed log-spread of 0.22 around the median (standard
   normal quantiles).

The target's own pre-release hours come from `history_hourly`; on the
final case the recorder only starts two days before release, so hours older
than a day are filled from the comparable carrying the target game's own
category history, which the case file ships as one of the comparables.

## Validation (practice set, black-box scorer)

| case | loss | typical miss |
|------|------|--------------|
| p00 Marvel's Wolverine | 0.1076 | 19.15% |
| p01 Red Dead Redemption 2 | 0.0902 | 15.59% |
| p02 God of War Ragnarök | 0.0767 | 11.22% |
| p03 Marvel's Spider-Man 2 | 0.1401 | 22.50% |
| **MEAN** | **0.1036** | **17.04%** |

90% coverage 0.902. Round 4's entry scored 0.1139 on the same set.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case.
Deterministic (two runs write byte-identical output), about 0.1 s per case.

## License

MIT — see [LICENSE](LICENSE).
