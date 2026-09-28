# Garment-integrated right-hand prosthesis — design spec

Date: 2026-09-28
Status: draft, awaiting review
Photos referenced: `prosthesis/IMG_1002–1036.HEIC`

This is the roadmap and architecture spec. Each track (S, M, V) gets its own
implementation plan once this spec is approved.

---

## 1. Purpose

Martin has a congenital right transcarpal deficiency (no palm), active wrist
flexion/extension, and limited forearm rotation. He wants **one prosthesis,
integrated into a long-sleeved garment, with minimal visible difference from
the left arm**, that supports:

1. **Handshake** — the primary social goal.
2. **Push-ups.**
3. **Pull-ups** (overhand grip).
4. **Violin bowing** — the right arm holds the bow; bow control comes from the
   wrist alone (Martin has tested this).

Later: active multi-grip control via EMG pattern recognition.

### Constraints

- No surgery.
- No gel electrodes in the final device (dry electrodes from day one).
- DIY, low budget; about 6 h/week of build time.
- The existing custom myoelectric prosthesis stays as the daily fallback until
  the new system passes a full day of wear.

### Ruling principle: human appearance first

When requirements conflict, the option that keeps the right arm visually and
kinematically closest to the left wins. Strength, violin precision and
push-ups must be achieved **inside a human-shaped envelope** mirrored from the
left arm.

### Success criteria

| Goal | Pass when |
|---|---|
| Handshake | 10 handshakes with different people; honest feedback that it looks and feels natural; 2 h wear with no skin problems |
| Push-ups | 3 × 10 fist push-ups, shoulders level, no pain, redness gone within 20 min |
| Pull-ups | 10 s dead hang without pain, then a first full overhand pull-up |
| Violin | Full-bow strokes on open strings, straight bow, even tone |
| Symmetry | Right fingertip within ±1 cm of left; hand mirrored from the left; mass budget met (§4.5) |

---

## 2. Body data

From photos with a tape measure, 2026-09-28. Zero point is the elbow crease.
Accuracy about ±1–2 cm (camera angle, zero landmark).

| Measure | Right | Left |
|---|---|---|
| Elbow crease → stub tip / wrist crease | ≈ 23.5 cm | ≈ 27 cm |
| Elbow crease → middle fingertip | — | ≈ 45 cm |
| Hand length (wrist crease → middle fingertip) | — | ≈ 18 cm |
| Wrist pivot | ≈ 20–21 cm (estimated, Phase 0 measures it) | ≈ 27 cm |
| Stub width | ≈ 5.7 cm at 15 cm; ≈ 4.2 cm from 20 cm to the tip | — |
| Forearm rotation | Limited, ≈ 60–80° total, reaches full pronation | Normal |

Observations:

- **No distal widening.** The stub tip is no wider than the segment behind it,
  so a cap cannot hold tension by shape.
- **Stub tip** has palm-type (glabrous) skin, two small nubs, and deep skin
  folds on the palm side when the wrist flexes. Palm skin tolerates pressure;
  the nubs do not. The nubs cannot be moved, so they are not a control
  channel.
- **Pivot offset ≈ 6 cm.** A hand mounted directly on the stub would bend
  mid-forearm.
- **Right forearm muscle bulk** looks close to the left.
- **Rotation** does not need a precise number: the rotator (§4.4) is
  adjustable.

---

## 3. Architecture (option B: shell + linkage)

```
shirt body / shoulders ─ light elastic, daily weight only
upper sleeve           ─ stretchy, not in the load path
band above elbow bones ─ main anchor for pull-ups
elbow                  ─ 2 non-stretch tapes crossing the elbow pivot
forearm sleeve         ─ compression knit, EMG snap electrodes (Track S)
        │
FOREARM SHELL (anchored to the forearm; outer shape = mirrored left wrist)
        │    stub moves freely inside ─ linkage ─ push-pull rod
        │                                               │
ROTATOR (360°, indexed, hollow) ◄── rod passes through its centre
        │
WRIST FLEX JOINT at the anatomical line (≈ 27 cm)
        │
HAND (metal skeleton, finger lock, silicone glove)
```

The stub is the **joystick, not the mount**. The shell, anchored to the
forearm, carries the hand and all task loads. The stub only drives the wrist
joint through the linkage.

Why B over direct mounting on the stub:

1. The hand pivots at the anatomical wrist line (appearance).
2. Push-up and pull-up loads go into the shell and forearm, not through stub
   skin or the wrist joint.
3. With a 1:1 linkage the wrist feels the hand's weight at its true lever
   (≈ 0.24 Nm for a 350 g hand) instead of the 6 cm longer one (≈ 0.46 Nm).
4. Bow control stays as Martin tested it, and the linkage gives an exact
   wrist-angle signal for EMG gating.

---

## 4. Components

### 4.1 Garment (long-sleeved base layer)

- **Function:** tension only. It never carries compression or wrist torque.
- **Load line:** band just above the elbow bones → two non-stretch tapes along
  the inner and outer seams, crossing the elbow at its pivot (length stays
  constant through flexion) → clips on the shell.
- **Band:** adjustable closure (Velcro or dial). Relief channel over the
  ulnar-nerve groove behind the inner elbow bump; load on the bony bumps
  either side.
- **Rule:** stretch everywhere except the load line.
- **Symmetry:** the left sleeve copies the right sleeve's panels and seams.
  Tapes are 1–2 mm webbing. The electronics pod sits under the arm near the
  elbow.
- **Comfort:** thin technical fabric; mesh at the armpit and inner elbow.
- **Washing:** shell, hand and pod unclip. Keep 2–3 shirts with an identical
  clip and snap pattern.
- **PoC alternative:** the existing Iceross Upper-X liner, locked to the shell
  by its distal pin, if Phase 0 shows the wrist moves comfortably inside it
  and the maker confirms the pin's load rating.

### 4.2 Forearm shell

- Outer shape = mirrored left distal forearm and wrist (from scans).
- Internal clearance for the stub's flexion arc. About 3 cm of stub beyond
  the pivot swinging ±40° moves the tip about ±2 cm.
- Soft relief pockets over both nubs; no rigid edge across the palm-side
  flexion crease.
- Compression (push-ups) bears on the forearm taper and the palm-type skin
  around the nubs.
- Build order: clear PETG test shell first (white skin = too much pressure),
  then composite or Kydex with an aluminium insert. No FDM plastic in the
  pull-up load path.
- Target mass ≤ 200 g.

### 4.3 Wrist linkage

- Stub flexion/extension drives hand flexion/extension about a joint at the
  anatomical wrist line.
- **Ratio 1:1** as the baseline: it keeps the tested bow control and a wrist
  torque at or below normal. Use 1:2 only if the shell clearance cannot fit
  the left forearm's thickness. That trade doubles the wrist torque and halves
  precision.
- Flexion is transmitted as an axial push-pull rod through the hollow centre
  of the rotator, so it works at every rotation setting.
- An angle sensor on the linkage feeds Track S (EMG gating).

### 4.4 Wrist rotator

- 360°, **indexed lock** (pin in holes every 15–30°), set with the left hand.
  Not friction: friction slips under pull-up twist.
- Default: Martin's comfortable mid-rotation maps to a thumb-up hand
  (handshake neutral).
- Settings per task: handshake neutral; pull-ups matched to the bar; push-ups
  comfortable fist; violin bow-hold angle.
- Chin-ups are out of scope: rotating the hand does not supinate the forearm,
  so the biceps advantage does not appear.

### 4.5 Hand

- **Geometry:** mirrored left hand. Alginate cast → plaster copy → scan →
  mirror.
- **Skeleton:** metal. Laser-cut stainless or aluminium plates; titanium
  printing optional.
- **Fingers:**
  - v1: one shared lock for the four fingers, and a fixed opposed thumb on a
    friction joint.
  - v2: an individual ratchet per finger (Point Digit style), which gives
    passive multi-grip.
- **Poses:**

  | Pose | Task | Lock |
  |---|---|---|
  | Fist | Push-ups | Locked closed |
  | Open hook, thumb out | Pull-ups | Locked; the bar can always be lifted out, never trapped |
  | Relaxed, spring-loaded | Handshake | Off |
  | Bow grip | Violin | Locked; silicone gives compliance |

- **Lock switch:** on/off slider on the back of the hand, operated by the left
  hand or against the thigh, so a handshake cannot trap the other person.
- **Skin:** replaceable platinum-cure silicone glove, thicker at the knuckles
  and palm, colour-matched to the left hand.
- **Mass:** hand + rotator + distal linkage ≤ 350 g. Heavy parts go in the
  palm near the wrist; fingers are light (hollow or lattice).
- **Noise:** no audible clicks from the lock.

### 4.6 Sensing (Track S — not required for goals 1–4)

- 4 × MyoWare v1 on the **RAW** output (not SIG).
- Dry stainless (316) or gold-plated snap electrodes on a neoprene band. No
  nickel.
- Shared reference electrode on the bony point of the elbow.
- Battery powered; ≥ 1 kHz per channel (2 kHz used). Firmware in **Rust**,
  no Arduino framework, on two rigs sending identical frames:
  - **First rig: the Arduino Uno R3 already owned** (Rust via avr-hal).
    10-bit ADC with AREF wired to 3.3 V, no IMUs. Answers S1 and S3a
    at no hardware cost.
  - **Second rig, only after S3a passes:** ESP32-C6 (Seeed XIAO ESP32C6,
    esp-hal) with an external MCP3208 12-bit SPI ADC (the ESP32's own ADC is
    too noisy and nonlinear for this measurement). Two BNO055 IMUs, one on
    the band and one on the stub. Needed for S3b and S4.
- Recorded classes include **isometric rotation attempts**. Rotation is
  blocked, so these contractions do not move the hand, which makes them
  candidate grip commands that avoid the free-wrist conflict.
- Result decides whether an active hand (v3) is pursued.

---

## 5. Loads

BW = body weight (Phase 0 measures it). Per-arm figures.

| Task | Load | Path | Design value |
|---|---|---|---|
| Push-ups | Compression, ≈ 0.35 BW | Floor → knuckles → skeleton → rotator → shell → forearm taper + palm skin | 1.5 × static |
| Pull-ups | Tension, 0.5 BW static | Bar → hooked fingers → skeleton → rotator → shell → tapes → band → upper-arm bone | Design 0.75 BW·g; proof test 1.5 BW·g (80 kg → ≈ 590 N design, ≈ 1180 N proof) |
| Handshake | Small | Hand weight → linkage → wrist | Wrist torque ≤ 0.3 Nm with forearm horizontal |
| Violin | Very small | Bow → hand → linkage → wrist | Precision over strength |

---

## 6. Phases and gates

### Phase 0 — measure and test (week 1)

1. Body weight.
2. Right wrist pivot: flex fully, measure from the elbow crease to the bend
   crease.
3. Wrist flexion/extension range in degrees (phone angle app).
4. Wrist extension strength: luggage scale at distance d, torque = F × d.
5. Nub sensation (they cannot move): touch, pressure and vibration — are
   they clearly felt? If yes, they are a candidate site for grip-force
   feedback (small vibration motor in the shell pocket).
6. Liner-only test: Iceross liner without the socket; flex, extend and rotate
   20× each. Comfortable, no pinching?
7. Towel push-up test (stop-gap until the hand exists).
8. Bare-arm lifting-hook hang test: low bar, feet on a scale, ≤ 30 % BW,
   stop at any pain.
9. String test: load line to the back vs to an above-elbow band, through the
   pull-up range.
10. Scans: right forearm and stub (SILIKONY, Praha 9), left forearm and hand;
    alginate cast of the left hand.
11. Email the prosthesis maker: replacement battery, limb scan and CAD files,
    electrode pinout, liner pin load rating.

**Gate:** all values recorded; scans in hand.

### Track S — EMG proof of concept (weeks 1–8)

| Step | Weeks | Work | Gate |
|---|---|---|---|
| S1 | 1–2 | Build the dry band and logger | Left forearm: clean bursts, flat rest, same gesture after re-donning |
| S2 | 3 | Record: rest, wrist flex, wrist ext, pronation attempt, supination attempt, co-contraction, imagined fist, imagined pinch × 5 reps × 3 arm positions × 2 conditions (still / moving); both arms; log the first minutes after donning | Data complete |
| S3 | 4–6 | From scratch: MAV, WL, ZC, SSC features per 200 ms → LDA (NumPy) → small MLP; accuracy vs class count; cross-position; false triggers while moving, with and without IMU gating | **S3a** (Uno): rest + 3 commands ≥ 90 % within every position. **S3b** (ESP32 rig with IMUs): false triggers ≤ 2 % with gating. Both → active hand viable |
| S4 | 7–8 | Real-time on the microcontroller plus live biofeedback; 10 min/day practice, log the learning curve | Learning curve recorded |

### Track M — mechanics (weeks 9–30)

| Step | Weeks | Result | Gate |
|---|---|---|---|
| M1 | 9–13 | Garment v0 (compression shirt + sewn webbing + band) and PETG test shell | 1 h wear, no pressure marks; string/load path confirmed |
| M2 | 13–17 | Linkage + rotator on the shell, dummy hand mass | Stub drives the wrist joint 1:1 at every rotation setting; torque at the stub ≤ 0.3 Nm |
| M3 | 17–23 | Hand v1: metal skeleton, shared lock, fixed thumb | Off-body proof test passed; fist push-ups and dead hang, then first pull-up |
| M4 | 23–30 | Hand v2: springs, lock switch, silicone glove | Handshake criterion (§1) |
| M5 | later | Individual ratchets, or a motor if gate S3 passed | — |

### Track V — violin (bow grip tuned in weeks 28–32)

- Contact the OHMI Trust and find a teacher open to an adaptive setup; lessons
  can start any time.
- Tune the bow-grip pose and rotator angle on hand v2.
- Gate: the violin criterion in §1.

**Interim:** towel push-ups, bare-arm hooks (if they passed), and the existing
prosthesis (with a new battery) for daily use.

---

## 7. Safety

- **EMG rig:** battery only. Laptop unplugged from the charger, or a USB
  isolator. Nothing mains-powered connected while wired up. Forearm electrodes
  only.
- **Before hanging body weight:** off-body proof test at 2× design load
  (ratchet strap + crane scale), then progressive on-body loading with feet on
  a scale.
- **Pull-up pose is an open hook.** The bar must always lift out, and the
  left hand must be able to free the right under load.
- **Skin:** check after every session; redness lasting > 20 min, blisters,
  numbness or tingling → stop and relieve that spot.
- **Ulnar nerve:** relief channel in the elbow band.
- **Electrodes:** nickel-free only.
- **Hang test:** partial load (≤ 30 % BW) until more is known; wrist ligaments
  give way before muscles.
- **Shoulder:** build the right shoulder up progressively (band-assisted →
  slow lowering → full).
- **Old battery:** do not force-charge the deeply drained pack; have the
  provider replace it.

---

## 8. Risks

| Risk | Early signal | Response |
|---|---|---|
| Wrist uncomfortable inside the liner | Phase 0 test 6 | Garment band path only |
| Shell clearance too bulky for symmetry | M1/M2 | 1:2 linkage ratio (costs torque and precision) |
| Dry electrodes noisy on hairy skin | S1 | Trim hair, more band pressure, larger domes |
| Congenital EMG patterns do not separate | S3 | Fewer classes, IMU gating, daily practice; passive hand remains |
| Skin breakdown under load | Redness > 20 min | Larger contact area, liner, relief pockets |
| Hand over mass budget | M3 weigh-in | Lighter finger lattice; move mass toward the wrist |
| Time slips | > 2 weeks behind | Run one track at a time |

---

## 9. Out of scope

- Surgery of any kind.
- Gel electrodes in the final device.
- Chin-ups (supinated grip).
- A motorised hand before gate S3 passes.
- A product for anyone other than Martin.

---

## 10. Existing assets

- Custom myoelectric prosthesis: Össur Iceross Upper-X liner (distal pin
  thread), printed lattice frame, 2 dry electrodes, dead battery.
- 4 × MyoWare v1 sensors; a pack of Medico MSGLT-09RT gel electrodes (old).
- Diploma thesis: 2-class classification on a microcontroller using the SIG
  output (no raw data kept).

---

## 11. References

- Kryger et al. 2011, pattern recognition in congenital transradial limb
  defects — https://doi.org/10.1177/0309364611420905
- Kaluf et al. 2021, congenital amputees controlling multi-DOF myoelectric
  prostheses — https://doi.org/10.1016/j.jhsa.2021.08.011
- Tacca et al. 2024, wearable high-density EMG sleeve —
  https://doi.org/10.1038/s41598-024-64458-x
- Point Digit II, ratcheting prosthetic finger —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7980476/
- Adrian Anantawan, violinist with a passive bow device —
  https://www.bostonglobe.com/2024/08/15/arts/adrian-anantawan-violinist-adaptive-music-technology/
- OHMI Trust, adaptive string instruments — https://www.ohmi.org.uk/strings.html
- e-NABLE bow holder —
  https://enablingthefuture.org/upper-limb-prosthetics/bow-holder-device/
- SILIKONY s.r.o. (3D scanning, skin silicones), Špitálská 885/2a, Praha 9 —
  https://www.silikonysro.cz/3d-sken
- MindRove 8-channel armband (fallback sensing option) —
  https://mindrove.com/armband/
- BioAmp EXG Pill (8-channel upgrade option) —
  https://docs.upsidedownlabs.tech/hardware/bioamp/bioamp-exg-pill/index.html
