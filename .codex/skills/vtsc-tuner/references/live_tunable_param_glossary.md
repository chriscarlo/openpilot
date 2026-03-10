# VTSC Live-Tuneable Parameter Glossary (Plain English)

Use this file when the user asks what a tuning key actually changes.

Scope:
- `VTSCExpertModeEnabled`
- `VTSC.Expert.*` keys added for live tuning
- `VTSCHUD.*` rally co-pilot HUD keys added for live tuning

Hot-apply behavior:
- VTSC controller expert keys refresh at 5 Hz (`vision_turn_params.py`, `PARAM_REFRESH_S = 0.2`).
- Rally co-pilot HUD keys refresh at ~5 Hz (`hud.cc`, refresh on `frame % 4`).
- No reboot/service restart required for these keys to take effect.

## Master Gate

| Key | What it adjusts |
| --- | --- |
| `VTSCExpertModeEnabled` | Master gate for expert live-tuning values. `0` means built-in defaults are used for `VTSC.Expert.*` and `VTSCHUD.*`; `1` means your live values are applied. |

## VTSC Expert Keys

| Key | What it adjusts | If value increases |
| --- | --- | --- |
| `VTSC.Expert.FreewayCurvEps` | Curvature cutoff for treating road as effectively straight in freeway fail-open logic. | More road sections are considered straight. |
| `VTSC.Expert.FreewayMinVisibleM` | Minimum visible distance required before freeway fail-open can activate. | Requires longer visible distance to fail-open. |
| `VTSC.Expert.FreewayMinConf` | Minimum confidence required for freeway fail-open and enough-vision checks. | Requires stronger confidence before relaxing occlusion behavior. |
| `VTSC.Expert.HighwayMinMps` | Speed boundary used by occlusion logic to split lower-speed versus highway behavior. | Highway branch starts at higher speed. |
| `VTSC.Expert.VTurnHoldMinVMps` | Minimum speed for freeway cap-hold behavior. | Hold engages only at higher speed. |
| `VTSC.Expert.VTurnHoldDeltaMps` | Minimum cap drop needed to trigger cap-hold. | Only larger cap drops get held. |
| `VTSC.Expert.VTurnHoldS` | Hold duration (clean-vision path). | Hold persists longer. |
| `VTSC.Expert.VTurnHoldSOccluded` | Hold duration when occlusion-triggered hold is used. | Occlusion hold persists longer. |
| `VTSC.Expert.EnteringPredLatAccTh` | Predicted lateral acceleration threshold required as turn evidence for occluded hold trigger. | Needs stronger turn evidence to trigger. |
| `VTSC.Expert.TrajectoryPhaseAdvanceS` | Global timing phase advance for interpreting model trajectory. | VTSC reacts earlier in time. |
| `VTSC.Expert.SteerFallbackModelKappaMax` | Max model curvature still treated as "model says straight" for steering fallback entry. | Steering fallback can activate in more cases. |
| `VTSC.Expert.SteerFallbackMinKappa` | Minimum steering-derived curvature required before fallback is accepted. | Harder for steering fallback to activate. |
| `VTSC.Expert.SteerFallbackMinVMps` | Minimum speed at which steering fallback is allowed. | Fallback only active at higher speeds. |
| `VTSC.Expert.SevereOvershootSpeedScaleMin` | Minimum speed scaling under severe/low-confidence overshoot handling. | Less conservative severe scaling. |
| `VTSC.Expert.HiddenTurnEnabled` | Enables hidden-turn early deceleration helper logic. | `1` enables feature, `0` disables feature. |
| `VTSC.Expert.HiddenTurnVMaxMps` | Max ego speed where hidden-turn helper is allowed. | Hidden-turn helper remains active at higher speeds. |
| `VTSC.Expert.HiddenTurnTHS` | Short-horizon time window used to compute available distance in hidden-turn checks. | More available distance is assumed. |
| `VTSC.Expert.HiddenTurnDeltaVMps` | Required speed deficit (or shed amount) to trigger hidden-turn intervention. | Harder to trigger hidden-turn intervention. |
| `VTSC.Expert.HiddenTurnMinOccS` | Minimum occlusion duration before hidden-turn logic can engage. | Must stay occluded longer before trigger. |
| `VTSC.Expert.HiddenTurnAvailScale` | Multiplier on available distance in hidden-turn margin test. | Margin test becomes more permissive. |
| `VTSC.Expert.HiddenTurnPhaseS` | Early-occlusion phase window where hidden-turn logic is active. | Hidden-turn logic stays active longer. |
| `VTSC.Expert.HiddenTurnHeadingWinS` | Heading lookahead window used by hidden-turn straightness estimation. | Uses longer heading window. |
| `VTSC.Expert.HiddenTurnVisHeadingMaxRad` | Max heading change still treated as straight enough for hidden-turn tightening. | More tolerant of heading change. |
| `VTSC.Expert.LowSpeedMarginMaxVMps` | Max speed where low-speed positive-margin override can suppress crawl-like behavior. | Override applies to higher speeds. |
| `VTSC.Expert.LowSpeedMarginCurvThresh` | Curvature threshold for "effectively straight" in low-speed margin override. | Override applies on curvier segments. |
| `VTSC.Expert.OcclBypassHeadwayVFloorMps` | Speed floor used in lead-headway computation for occlusion bypass. | Computed headway shrinks more at crawl speeds; bypass triggers easier. |
| `VTSC.Expert.OcclBypassLowSpeedVMps` | Low-speed threshold for close-lead occlusion bypass path. | Close-lead bypass applies at higher speeds. |
| `VTSC.Expert.OcclBypassLeadDRelMaxM` | Max lead distance for "close lead" qualification in low-speed bypass path. | Farther leads still qualify as close. |
| `VTSC.Expert.ConfidenceEnterSevere` | Confidence threshold to enter severe occlusion state. | Enters severe state sooner. |
| `VTSC.Expert.ConfidenceExitToPartial` | Confidence threshold to recover from severe/lost back to partial. | Needs stronger recovery before exiting severe/lost. |

## Rally Co-Pilot HUD Keys (`VTSCHUD.*`)

| Key | What it adjusts | If value increases |
| --- | --- | --- |
| `VTSCHUD.KappaShowMin` | Curvature threshold to initially show the rally co-pilot curve HUD. | HUD appears only on stronger curves. |
| `VTSCHUD.KappaHoldMin` | Curvature hysteresis threshold to keep HUD visible once shown. | HUD drops out sooner as curvature decreases. |
| `VTSCHUD.CurveHoldNewDistMinM` | While in-curve, min new-curve distance before accepting a new geometry snapshot. | More aggressive geometry lock; fewer mid-curve geometry swaps. |
| `VTSCHUD.GeometryEpsilonM` | Tolerance for deciding whether incoming geometry changed enough to reset ego-advance interpolation. | Less sensitive to small geometry jitter. |
| `VTSCHUD.FadeInAlpha` | Fade-in response coefficient. | Faster fade-in. |
| `VTSCHUD.FadeOutAlpha` | Fade-out response coefficient. | Faster fade-out. |
| `VTSCHUD.Scale` | Global rally co-pilot widget scale. | Entire widget renders larger. |
| `VTSCHUD.BottomSafePxAtScale1` | Bottom safe-area clearance. | Widget is positioned farther above the bottom edge. |
| `VTSCHUD.PadPxAtScale1` | Internal padding inside widget bounds. | More internal whitespace. |
| `VTSCHUD.GapPxAtScale1` | Vertical gap between top label, curve area, and bottom labels. | Larger section spacing. |
| `VTSCHUD.TopHeightPxAtScale1` | Height allocated for top speed label row. | Top row gets taller. |
| `VTSCHUD.BottomHeightPxAtScale1` | Height allocated for bottom distance/time row. | Bottom row gets taller. |
| `VTSCHUD.MinCurveAreaPxAtScale1` | Minimum curve-area size required before drawing strip-map geometry. | More likely to suppress drawing in tight layouts. |
| `VTSCHUD.RoadMainWidthPxAtScale1` | Base road backbone thickness. | Backbone appears thicker. |
| `VTSCHUD.GlowWidthPxAtScale1` | Glow stroke width around the road backbone. | Glow appears wider. |
| `VTSCHUD.OutlineWidthPxAtScale1` | Dark outline stroke width around backbone. | Outline appears thicker. |
| `VTSCHUD.MainStrokeWidthPxAtScale1` | Bright main backbone stroke width. | Main line appears thicker. |
| `VTSCHUD.DistanceLabelSepPxAtScale1` | Horizontal spacing between distance and time labels. | Labels are spaced farther apart. |
| `VTSCHUD.SpeedFontPxAtScale1` | Speed label font size. | Top speed text appears larger. |
| `VTSCHUD.BottomFontPxAtScale1` | Bottom distance/time font size. | Bottom text appears larger. |
