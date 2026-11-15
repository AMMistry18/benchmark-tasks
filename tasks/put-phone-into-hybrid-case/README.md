# Put a smartphone into its hybrid case the right way round and snap all four corners under the lip

Put the phone in the case screen up, camera in the camera cutout, all four corners under the lip. Leave it on the mat.

About three in four smartphone owners use a case, and fitting a phone into a new or cleaned case is an everyday
job. Hybrid cases (a rigid clear back with a soft TPU bumper round it) are among the best-selling kinds. Their
makers' fitting steps are the same: line the phone's top edge up with the case's top edge so the camera lines up
with the cutout, push the top in first, then press the rest down until the bottom corners snap into place, and check
that all corners and edges are seated (Spigen Ultra Hybrid / Ultra Hybrid MagFit instruction sheets for iPhone 15,
16 and 17, as reproduced on manuals.plus; Spigen's own install guides at install.spigen.com).

The skill is a two-stage snap fit on a thin, smooth slab: turn the phone a quarter turn so its camera block meets the
cutout, set it on the lip, push the top end down until its corners slip under the lip, then press the bottom end down
corner by corner until those corners snap under too. Each press must land on the glass near the corner, inside the
lip, straight down and firm enough to push the lip back. Turned end for end, the camera block sits on the solid back,
props that end up and it can never seat.

## Run and score

From this task folder, with the official YAM Arena environment installed:

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=put_phone_into_hybrid_case.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Put the phone in the case screen up, camera in the camera cutout, all four corners under the lip. Leave it on the mat." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ./demo-run
```

For a model trial replace `--policy yam_scripted` with `--policy dream`. The output of `--scorer task_score` is the
score. One trial allows 2,400 control steps (120 s at 20 Hz). The scripted demo plan lasts 56.3 s (a second pass over the top-end corners comes last); on the GPU it reaches
full score with `success` termination at about 657 steps (33 s), after the first bottom-corner press snaps the bottom
end in and slides the top end fully under its lip; it uses only the authored layout and elapsed time and never reads
object state. Wall-clock time per trial on one GPU: about 4-6 minutes (about 0.2 s per step plus start-up and video).
Seeds: the start layout is fixed (`object_jitter=0`); the phone and the case are free bodies, the mat is fixed.

Offline tests: `pytest test_put_phone_into_hybrid_case.py` (scorer layouts: goal, every partial tier, the cheats).

## Scene (metres, world frame: x forward, y left, z up; table top z = 0.7475; arm bases at (0, +-0.305))

| Object | Real item | Model | Pose / size |
|---|---|---|---|
| mat | Grey felt desk mat, 3 mm | fixed box | 0.26 x 0.50 x 0.003, centre (0.18, 0.00), on the table |
| case | Spigen Ultra Hybrid for Apple iPhone 15 (non-MagFit), Navy Blue; polycarbonate back, TPU bumper; listed 151.6 x 75.4 x 13 mm (5.97 x 2.97 x 0.51 in incl. camera guard), about 31 g | free mechanism: back + side walls (base, 23 g) + 4 sprung corner latches (2 g each) | centre of its underside (0.170, 0.000) on the mat, long axis along y (yaw +90 deg), camera cutout at the +y end, far (+x) side |
| phone | Apple iPhone 15, 147.6 x 71.6 x 7.80 mm, 171 g (Apple technical specifications); aluminium frame, glass front and back | free body: bevelled convex hull + camera block + screen | screen up, centre (0.170, -0.170, mat + 3.9 mm), long axis along x, top end toward +x |

### Case (case frame: x along the length toward the camera end, origin at the centre of the underside; mm)

| Feature | Size | Source |
|---|---|---|
| Footprint | 151.4 x 75.4 | listed 151.6 x 75.4 |
| Back (polycarbonate) | 1.2 thick, with a 32 x 32 camera cutout, x 41.8..73.8, y -35.8..-3.8 | estimate from photos |
| Pocket | 148.0 x 72.0 (0.2 play per side and end) | estimate |
| Side walls (TPU) | 1.7 thick, 10.3 tall, no lip modelled | estimate |
| End lips, four corner sections | each from the corner to 0.5 mm short of the end's middle (35.4 long); underside 9.3 above the case underside, top 10.3; edge 1.0 over the screen's end edge; 45 deg lead-in on the inner top edge (0.8 x 0.8) | estimate from photos |
| Corner latch | the end-wall section with its lip slides outward along x, 0..2.5 mm; spring 1.5 N/mm with the rest 1.3 mm past the closed stop (1.95 N closed, 3.75 N fully open); damping 5 N s/m | stand-in for the TPU bumper's flex (see below) |

### Phone (phone frame: x toward the top end, y to the screen's left, z out of the screen; origin at the body centre)

| Feature | Size | Source |
|---|---|---|
| Body | 147.6 x 71.6 x 7.80, 171 g | Apple |
| Edge | 1.5 mm x 45 deg bevel top and bottom (stand-in for the rounded frame); plan corners square | estimate |
| Camera block | 28 x 28, 1.0 proud (modelled within the 1.2 mm case back; the case's raised camera guard is not modelled), centred 16 mm from the top edge and 16 mm from the right edge (seen from the screen) | estimate from photos |

## The snap force: what is known and what is assumed

- Fitting method (verified): Spigen's instruction sheets for the Ultra Hybrid on iPhone 15/16/17 say to line the
  phone's top edge up with the case's top edge (camera matching the cutout), push the phone in starting from the top,
  then work down to snap the bottom corners into place, and check that all corners and edges are seated. Only the
  TPU bumper flexes; the polycarbonate back is rigid.
- Force (not published): no maker, retailer or test lab figure for the force to press a phone into a hybrid case was
  found. The only related figure found is a reviewer's retention test (about 11 lb, roughly 50 N, to dislodge an
  iPhone from a Spigen hybrid PC/TPU case; method not stated), which shows a seated phone is held far more firmly
  than a few newtons.
- Our estimate: 10-20 N to press one end in (two corners), from a cantilever estimate of the 1.2-1.5 mm TPU end wall
  flexing outward about 1 mm, with TPU-on-glass friction on the lead-in. This is an estimate, not a measurement.
- Simulation: each corner latch needs about 6-7 N straight down on the glass near its corner (3.75 N spring at full
  travel over a 45 deg lead-in with friction 0.3-0.375), so about 12-14 N per end, pressed one corner at a time.
- Robot: a one-pad press commanded 20-30 mm past first contact gave 9-14 N on a 3 N/mm sprung plate at the far
  corner positions and about 12.5 N at 20 mm at the near positions (probe on the unmodified simulator). The demo
  commands about 21 mm past first contact.
- How to check a real case: press each corner down with a push-pull force gauge fitted with a 14 mm rubber pad,
  10 mm in from the end and 6 mm in from the side, and read the peak as the corner snaps under. Use cases whose
  peak per corner is within the robot's budget (about 8-10 N); a case needing more is out of scope.

## Real-trial safety (pressing on glass)

- Use a non-working iPhone 15 display dummy (same outline and glass front) ballasted to 171 g, or a phone with a
  sacrificial tempered-glass protector counted in its thickness; never a phone in use.
- Cover the fingertips with 1 mm rubber pads. The demo commands each press about 21 mm past first contact, which gives roughly 10-14 N on the glass, spread over a 14 x 3.5 mm pad.
- Keep the e-stop in reach, stand clear of the arms' reach, and stop the trial if a press lands on the lip or the
  phone is about to fall off the table edge.

## Real-world parts list

- Spigen Ultra Hybrid case for iPhone 15 (Navy Blue), new or cleaned; replace it if the lip tears.
- Apple iPhone 15 display dummy or phone (see safety), switched off.
- Grey felt desk mat, 3 mm, cut to 260 x 500 mm, taped down.
- Caliper with a depth rod, a magnifier or phone camera for the lip-over-glass check, a push-pull force gauge
  (0-50 N) for the snap force check.

## Scoring (same as task.json and the code)

Judged once both grippers have let go (centre point between the pads at least 30 mm from the phone's outline) and
the phone and case have both been still for 1 s. Right way round: screen up within 10 deg and the phone's top end at
the camera-cutout end within 10 deg. A corner is seated when the lip lies over the glass by at least 0.3 mm, the
glass at that corner is between 1.0 and 2.3 mm below the lip top (no higher than the lip's underside, and in the
pocket), and the corner is no more than 1 mm past the pocket's end or side wall. 1.0: right way round, four corners
seated, the case flat (under 3 deg) on the mat and within 0.05 m of its start. 0.75: three seated, or four with the
case moved or tipped. 0.5: two seated. 0.25: one seated. 0.1: right way round, resting on or in the case, none
seated. 0.0: anything else. Sticky zeros: screen down at any time, the phone dropped (falling faster than 0.8 m/s),
the phone or case off the mat.

Quick real-trial check per corner: look straight down with a phone camera or magnifier to see the lip edge lying on
the glass (a 0.3 mm feeler blade slid along the glass must stop under the lip), then set the caliper's depth rod on
the lip top and read the glass beside it (1.0-2.3 mm = in; less = riding on the lip).

## Assets and licences

All geometry is built in code from primitives and procedural meshes (kit modules `yam_parts.py`, `yam_joints.py`,
`yam_mesh.py`, included unmodified). No third-party meshes, textures, scans or fonts are used.
