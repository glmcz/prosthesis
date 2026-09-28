# Phase 0 + Track S (EMG proof of concept) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the arm (Phase 0), then prove or refute that dry-electrode EMG from Martin's right forearm can drive at least three grip commands reliably (Track S): logger, recordings, from-scratch analysis, and on-chip classification.

**Architecture:** First rig: the Arduino Uno R3 you already own, running Rust (avr-hal), samples four MyoWare RAW outputs at 2 kHz each (10-bit, AREF 3.3 V) and streams CRC-checked binary frames over its USB-serial bridge at 1 Mbaud. Only if the accuracy gate S3a passes does a second rig follow: an ESP32-C6 with an MCP3208 (12-bit) and two BNO055 IMUs, same frames, for IMU gating and on-chip classification. A NumPy-only Python package parses the stream, filters it with biquads, extracts Hudgins features, labels guided recordings, and evaluates LDA/MLP classifiers with hold-out-by-rep cross-validation. The same filter → features → LDA chain is re-implemented in `no_std` Rust, checked against Python-generated golden data, and runs on the chip.

**Tech Stack:** Uno rig: Rust nightly-2025-04-27 + avr-hal (pinned rev), ravedude/avrdude · ESP32 rig: Rust 1.95 + esp-hal 1.2, espflash 4.6 · both: heapless 0.9 · Python 3.12, uv, NumPy ≥ 2, pyserial, pytest · Arduino Uno R3, 4 × MyoWare v1, stainless snap electrodes; later Seeed XIAO ESP32C6, MCP3208, 2 × BNO055.

**Spec:** `prosthesis/docs/2026-09-28-roadmap-design.md` — §6 (Phase 0, Track S), §4.6 (sensing), §7 (safety).

## Global Constraints

- No gel electrodes: dry 316 stainless or gold-plated snap electrodes only, never nickel-plated.
- MyoWare **RAW** output, not SIG. MyoWare boards are powered from **3.3 V**, so no input can exceed the 3.3 V ADC reference (Uno: AREF wired to 3.3 V; ESP32 rig: MCP3208 VREF).
- Both rigs send the same frames on the same 12-bit / 3.3 V scale: the Uno shifts its 10-bit readings left by 2. The host code is identical for both.
- Money follows evidence: the ESP32-C6 rig (Task 16) is built only after S3a passes on the Uno.
- 2 kHz per channel (spec minimum 1 kHz). Decisions every 50 ms on 200 ms windows: `WIN = 400`, `HOP = 100`.
- Filtering: 4th-order 20 Hz high-pass + 50 Hz notch (EU mains).
- Python analysis is from scratch: NumPy only. No SciPy, scikit-learn or PyTorch.
- Firmware is Rust, no Arduino. Logic that can run on the laptop lives in the `no_std` crates `frame`, `sensors`, `dsp` and is tested there.
- The frame format is byte-identical in Python and Rust; both test suites check the same golden vectors.
- Safety (spec §7): battery power only while electrodes are on skin — laptop **unplugged from its charger**, nothing mains-powered connected, electrodes on the forearm only.
- Gates: **S1** re-donning cosine similarity ≥ 0.90 for every class · **S3a** rest + 3 commands ≥ 90 % in every arm position (Uno) · **S3b** false triggers ≤ 2 % with IMU gating (ESP32 rig) · **S4** on-chip agreement ≥ 95 %, learning curve recorded.
- Session recordings are never committed (gitignored); back them up by hand. Arm photos (`*.HEIC`) are never committed.

## Review Focus

1. **Windows from one hold in both training and test.** Windows overlap (200 ms every 50 ms), so a random split leaks the test hold into training and inflates accuracy. Every split holds out whole reps. Test: Task 14, `test_cross_validation_does_not_leak_windows_of_the_same_hold`, proven red under a random split.
2. **The chip's u32 microsecond clock wraps every 71.6 min**, possibly mid-session. Times must stay monotonic. Tests: Task 3 (`U32Unwrapper`), Task 6 (`test_rate_survives_clock_wrap`).
3. **Garbage, split or corrupted bytes on USB.** The parser must resync, count bad CRCs and never emit a corrupt frame. Tests: Task 3.
4. **A disconnected electrode (constant channel)** must not cause division by zero or a singular covariance. Tests: Task 12.
5. **Train/serve skew.** Live features (streaming) or on-chip features (float32 Rust) must match offline ones. Tests: Task 15 (streaming = offline for any chunking), Task 17 (golden features *and* decisions, proven red for an off-by-one window), Task 18 (agreement gate on real hardware).

---

## File structure

All paths are relative to `prosthesis/`. Python commands run from `prosthesis/emg`, Rust tests from `prosthesis/emg/firmware`, firmware builds from `prosthesis/emg/firmware/uno-logger` or `.../logger`.

| Path | Responsibility |
|---|---|
| `data/phase0.md` | Phase 0 measurements and test results |
| `emg/emgpoc/protocol.py` | Frame format, CRC, incremental parser, clock unwrapping |
| `emg/emgpoc/filters.py` | Biquad high-pass/notch, streaming filter chain |
| `emg/emgpoc/streamstats.py` | Stream health: rate, lost frames, level, noise |
| `emg/emgpoc/display.py` | Terminal bars |
| `emg/emgpoc/serialport.py` | Find/open the ESP32-C6 USB serial port |
| `emg/emgpoc/features.py` | MAV, WL, ZC, SSC on sliding windows |
| `emg/emgpoc/session.py` | Recording protocol: classes, schedule, window labels |
| `emg/emgpoc/gating.py` | Quaternion maths, wrist speed, IMU gate |
| `emg/emgpoc/sessionio.py` | Collect frames into a session, save/load `.npz` |
| `emg/emgpoc/dataset.py` | Sessions → labelled feature table; re-donning similarity |
| `emg/emgpoc/standardize.py`, `lda.py`, `mlp.py`, `models.py` | Classifiers from scratch, pipeline, model files |
| `emg/emgpoc/evaluate.py`, `report.py` | Cross-validation, S3 metrics, S3 report |
| `emg/emgpoc/streaming.py`, `practice.py` | Live features; biofeedback practice log |
| `emg/emgpoc/export_rs.py`, `onchip.py` | Rust source generation; on-chip agreement check |
| `emg/scripts/*.py` | Thin command-line wrappers (hardware I/O) |
| `emg/firmware/frame`, `sensors`, `dsp` | `no_std` Rust crates, tested on the laptop |
| `emg/firmware/uno-logger` | Arduino Uno R3 binary (first rig) |
| `emg/firmware/logger` | ESP32-C6 binary (second rig, after S3a) |

## Schedule (spec §6, about 6 h/week)

| Week | Tasks |
|---|---|
| 1 | 1–5, Phase 0 (Task 2) alongside |
| 2 | 6–10 on the Uno, gate S1 |
| 3 | 11 (recording) |
| 4–6 | 12–14, gate S3a |
| 7 | 15, daily practice starts |
| 8–9 | 16 only if S3a passed: ESP32-C6 rig, gate S3b |
| 10 | 17–18, gate S4 |

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`, `emg/pyproject.toml`, `emg/emgpoc/__init__.py`, `emg/tests/test_smoke.py`

**Interfaces:** Produces the `emgpoc` package and the `uv run pytest` entry point used by every later task.

- [ ] **Step 1: Make `prosthesis/` a git repository**

```bash
cd prosthesis && git init
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
*.HEIC
.DS_Store
__pycache__/
emg/.venv/
emg/data/sessions/
emg/data/sessions_unsettled/
emg/firmware/target/
emg/firmware/logger/target/
emg/firmware/uno-logger/target/
```

- [ ] **Step 3: Write `emg/pyproject.toml`**

```toml
[project]
name = "emgpoc"
version = "0.1.0"
description = "EMG proof of concept for the garment-integrated prosthesis (Track S)"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "pyserial>=3.5"]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["emgpoc"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the package marker and a smoke test**

`emg/emgpoc/__init__.py`:

```python
"""EMG proof of concept: logger protocol, signal processing, classifiers."""
```

`emg/tests/test_smoke.py`:

```python
import emgpoc


def test_package_imports():
    assert emgpoc.__doc__
```

- [ ] **Step 5: Run the tests**

Run: `cd emg && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add .gitignore docs emg/pyproject.toml emg/uv.lock emg/emgpoc emg/tests
git commit -m "Scaffold the EMG proof-of-concept project"
```

---

### Task 2: Phase 0 — measure and test (manual)

**Files:**
- Create: `data/phase0.md`

**Interfaces:** Produces the body data that spec §2 marks as estimates, and the go/no-go facts for Track M. No code.

- [ ] **Step 1: Create the record sheet `data/phase0.md`**

```markdown
# Phase 0 results

Date:
Zero point for lengths: elbow crease

| # | Measure / test | Result | Notes |
|---|---|---|---|
| 1 | Body weight (kg) | | |
| 2 | Right wrist pivot: elbow crease → bend crease, wrist fully flexed (cm) | | |
| 3 | Wrist flexion / extension range (deg) | | phone angle app on the stub |
| 4 | Wrist extension strength: F (N) at d (m) → torque F × d (Nm) | | luggage scale, strap at d |
| 5 | Nub sensation: touch / pressure / vibration clearly felt? | | nubs cannot move |
| 6 | Liner-only test: flex, extend, rotate 20× each — comfortable? pinching? | | Iceross without socket |
| 7 | Towel push-ups: reps, pain, redness fade time (min) | | |
| 8 | Lifting-hook hang: max load reached (% BW), hold (s), fade time (min) | | ≤ 30 % BW |
| 9 | String test: length change, back anchor vs above-elbow band | | |
| 10 | Scans: right forearm + stub, left forearm + hand (files) | | SILIKONY, Praha 9 |
| 11 | Alginate cast of the left hand done | | |
| 12 | Maker email sent / answered: battery, scan + CAD, pinout, liner pin rating | | |
```

- [ ] **Step 2: Measurements 1–5**

Weigh in the morning. For the pivot (2), flex the right wrist fully, mark the bend crease with a skin pen, and measure from the elbow crease with the arm flat on the table. For strength (4), loop a strap around the stub at a measured distance d from the pivot, pull with the luggage scale while pushing the stub up against it, and read the peak.

- [ ] **Step 3: Tests 6–9**

**Safety:** hang test at no more than 30 % of body weight (feet on a bathroom scale, bar low enough to stand). Stop at any pain in the wrist joint; ligaments fail before muscles. Check the skin after every test and write the time until redness fades.

- [ ] **Step 4: Scans, cast, maker email (10–12)**

Book SILIKONY (+420 777 312 846, Thursdays 13–18, about 650 Kč per 30 min). Ask for `.stl` of both forearms: right including the stub, wrist neutral; left including the hand. Make the alginate cast of the left hand in a relaxed handshake pose, and pour plaster the same day.

- [ ] **Step 5: Commit the sheet**

```bash
git add data/phase0.md
git commit -m "Record Phase 0 measurements"
```

**Gate Phase 0:** every row filled.

---

### Task 3: Frame protocol (Python)

**Files:**
- Create: `emg/emgpoc/protocol.py`
- Test: `emg/tests/test_protocol.py`

**Interfaces:**
- Produces: `SYNC`, `EMG_TYPE`, `IMU_TYPE`, `CLASS_TYPE`, `EMG_LEN = 18`, `IMU_LEN = 26`, `CLASS_LEN = 11`, `QUAT_SCALE`, `ADC_VOLTS_PER_COUNT`; `crc8(data: bytes) -> int`; frozen dataclasses `EmgFrame(seq, t_us, ch: tuple[int,int,int,int])`, `ImuFrame(seq, t_us, q_band, q_stub)`, `ClassFrame(seq, t_us, class_id)`; `Frame` union; `encode_emg`, `encode_imu`, `encode_class -> bytes`; `FrameParser.feed(bytes) -> list[Frame]` with counters `bad_crc`, `skipped_bytes`; `U32Unwrapper.__call__(t: int) -> int`; `count_seq_gaps(seqs: list[int]) -> int`.

- [ ] **Step 1: Write the failing test**

```python
from emgpoc.protocol import (
    ClassFrame,
    EmgFrame,
    FrameParser,
    ImuFrame,
    U32Unwrapper,
    count_seq_gaps,
    crc8,
    encode_class,
    encode_emg,
    encode_imu,
)

# Golden vectors, shared with firmware/frame/src/lib.rs
GOLDEN_EMG = bytes.fromhex("a55a010100e8030000010002000300040046")
GOLDEN_IMU = bytes.fromhex("a55a020200d0070000004000000000000000000040000000002b")
GOLDEN_CLASS = bytes.fromhex("a55a030300b80b0000020f")


def test_crc8_smbus_check_value():
    assert crc8(b"123456789") == 0xF4


def test_encoders_match_golden_vectors():
    assert encode_emg(1, 1000, (1, 2, 3, 4)) == GOLDEN_EMG
    assert encode_imu(2, 2000, (16384, 0, 0, 0), (0, 16384, 0, 0)) == GOLDEN_IMU
    assert encode_class(3, 3000, 2) == GOLDEN_CLASS


def test_parses_each_frame_type():
    frames = FrameParser().feed(GOLDEN_EMG + GOLDEN_IMU + GOLDEN_CLASS)
    assert frames == [
        EmgFrame(1, 1000, (1, 2, 3, 4)),
        ImuFrame(2, 2000, (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ClassFrame(3, 3000, 2),
    ]


def test_frame_split_across_feeds_including_sync():
    parser = FrameParser()
    data = GOLDEN_EMG
    assert parser.feed(data[:1]) == []
    assert parser.feed(data[1:7]) == []
    assert parser.feed(data[7:]) == [EmgFrame(1, 1000, (1, 2, 3, 4))]


def test_leading_garbage_is_skipped_and_counted():
    parser = FrameParser()
    frames = parser.feed(b"\x00\xff\x13" + GOLDEN_EMG)
    assert frames == [EmgFrame(1, 1000, (1, 2, 3, 4))]
    assert parser.skipped_bytes == 3


def test_corrupted_frame_is_dropped_and_next_frame_survives():
    bad = bytearray(GOLDEN_EMG)
    bad[9] ^= 0x01  # flip a payload bit, CRC no longer matches
    parser = FrameParser()
    frames = parser.feed(bytes(bad) + GOLDEN_CLASS)
    assert frames == [ClassFrame(3, 3000, 2)]
    assert parser.bad_crc == 1


def test_unknown_type_is_skipped():
    parser = FrameParser()
    frames = parser.feed(b"\xa5\x5a\x7f" + GOLDEN_EMG)
    assert frames == [EmgFrame(1, 1000, (1, 2, 3, 4))]


def test_count_seq_gaps_handles_u16_wraparound():
    assert count_seq_gaps([10, 11, 12]) == 0
    assert count_seq_gaps([10, 13]) == 2
    assert count_seq_gaps([65534, 65535, 0, 1]) == 0
    assert count_seq_gaps([65535, 2]) == 2


def test_unwrapper_is_monotonic_across_u32_wrap():
    u = U32Unwrapper()
    out = [u(t) for t in [4_294_967_000, 4_294_967_295, 100, 5_000]]
    assert out == [4_294_967_000, 4_294_967_295, 4_294_967_396, 4_294_972_296]
    assert all(b > a for a, b in zip(out, out[1:]))


def test_small_backwards_jitter_is_not_a_wrap():
    u = U32Unwrapper()
    assert [u(t) for t in [1_000, 990]] == [1_000, 990]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.protocol'`

- [ ] **Step 3: Implement**

```python
"""Binary frame protocol shared with the ESP32-C6 firmware (firmware/frame).

Every frame: 0xA5 0x5A | type u8 | seq u16 LE | t_us u32 LE | payload | crc8
The CRC covers type..payload (not the sync bytes).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

SYNC = b"\xa5\x5a"
EMG_TYPE = 0x01
IMU_TYPE = 0x02
CLASS_TYPE = 0x03
EMG_LEN = 18
IMU_LEN = 26
CLASS_LEN = 11
QUAT_SCALE = 16384.0  # BNO055 quaternion: 1 LSB = 2^-14
ADC_VOLTS_PER_COUNT = 3.3 / 4095  # MCP3208, 12-bit, 3.3 V reference

_FRAME_LEN = {EMG_TYPE: EMG_LEN, IMU_TYPE: IMU_LEN, CLASS_TYPE: CLASS_LEN}


def crc8(data: bytes) -> int:
    """CRC-8/SMBUS: poly 0x07, init 0x00, no reflection, no final xor."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class EmgFrame:
    seq: int
    t_us: int
    ch: tuple[int, int, int, int]


@dataclass(frozen=True)
class ImuFrame:
    seq: int
    t_us: int
    q_band: tuple[float, float, float, float]  # w, x, y, z
    q_stub: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClassFrame:
    seq: int
    t_us: int
    class_id: int


Frame = EmgFrame | ImuFrame | ClassFrame


def _wrap(seq: int, t_us: int) -> tuple[int, int]:
    return seq & 0xFFFF, t_us & 0xFFFFFFFF


def encode_emg(seq: int, t_us: int, ch: tuple[int, int, int, int]) -> bytes:
    body = struct.pack("<BHI4H", EMG_TYPE, *_wrap(seq, t_us), *ch)
    return SYNC + body + bytes([crc8(body)])


def encode_imu(seq: int, t_us: int, q_band_raw: tuple[int, ...], q_stub_raw: tuple[int, ...]) -> bytes:
    body = struct.pack("<BHI8h", IMU_TYPE, *_wrap(seq, t_us), *q_band_raw, *q_stub_raw)
    return SYNC + body + bytes([crc8(body)])


def encode_class(seq: int, t_us: int, class_id: int) -> bytes:
    body = struct.pack("<BHIB", CLASS_TYPE, *_wrap(seq, t_us), class_id)
    return SYNC + body + bytes([crc8(body)])


class FrameParser:
    """Incremental parser: feed arbitrary byte chunks, get complete valid frames back."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.bad_crc = 0
        self.skipped_bytes = 0

    def feed(self, data: bytes) -> list[Frame]:
        self._buf += data
        frames: list[Frame] = []
        while True:
            start = self._buf.find(SYNC)
            if start < 0:
                # keep a trailing 0xA5: the sync may be split across chunks
                keep = 1 if self._buf[-1:] == SYNC[:1] else 0
                self.skipped_bytes += len(self._buf) - keep
                del self._buf[: len(self._buf) - keep]
                return frames
            if start:
                self.skipped_bytes += start
                del self._buf[:start]
            if len(self._buf) < 3:
                return frames
            length = _FRAME_LEN.get(self._buf[2])
            if length is None:
                self.skipped_bytes += 1
                del self._buf[:1]
                continue
            if len(self._buf) < length:
                return frames
            body = bytes(self._buf[2 : length - 1])
            if crc8(body) != self._buf[length - 1]:
                self.bad_crc += 1
                self.skipped_bytes += 1
                del self._buf[:1]
                continue
            frames.append(_decode(body))
            del self._buf[:length]


def _decode(body: bytes) -> Frame:
    kind = body[0]
    if kind == EMG_TYPE:
        _, seq, t_us, *ch = struct.unpack("<BHI4H", body)
        return EmgFrame(seq, t_us, tuple(ch))
    if kind == IMU_TYPE:
        _, seq, t_us, *q = struct.unpack("<BHI8h", body)
        qf = tuple(v / QUAT_SCALE for v in q)
        return ImuFrame(seq, t_us, qf[:4], qf[4:])
    _, seq, t_us, class_id = struct.unpack("<BHIB", body)
    return ClassFrame(seq, t_us, class_id)


class U32Unwrapper:
    """Turns the firmware's wrapping u32 micros() into a monotonic int (wraps every ~71.6 min)."""

    def __init__(self) -> None:
        self._offset = 0
        self._last: int | None = None

    def __call__(self, t: int) -> int:
        if self._last is not None and t < self._last - 2**31:
            self._offset += 2**32
        self._last = t
        return t + self._offset


def count_seq_gaps(seqs: list[int]) -> int:
    """Frames lost between consecutive received u16 sequence numbers (handles wraparound)."""
    return sum(((cur - prev) & 0xFFFF) - 1 for prev, cur in zip(seqs, seqs[1:]))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_protocol.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/protocol.py emg/tests/test_protocol.py
git commit -m "Add CRC-checked binary frame protocol with resyncing parser"
```

---

### Task 4: Rust `frame` crate

**Files:**
- Create: `emg/firmware/Cargo.toml`, `emg/firmware/frame/Cargo.toml`, `emg/firmware/frame/src/lib.rs`

**Interfaces:**
- Consumes: the golden vectors from Task 3 (identical bytes).
- Produces: `frame::{crc8, encode_emg(seq: u16, t_us: u32, ch: &[u16; 4]) -> [u8; 18], encode_imu(seq, t_us, q_band: &[i16; 4], q_stub: &[i16; 4]) -> [u8; 26], encode_class(seq, t_us, class_id: u8) -> [u8; 11], EMG_CHANNELS}` — used by both loggers.

- [ ] **Step 1: Write the workspace manifest `emg/firmware/Cargo.toml`**

`sensors` joins the members in Task 16, `dsp` in Task 17. The two loggers stay outside the workspace: each has its own target and toolchain.

```toml
# Host-testable no_std building blocks. The ESP32-C6 binary lives in logger/ (own target config).
[workspace]
resolver = "2"
members = ["frame"]
exclude = ["logger", "uno-logger"]

[workspace.package]
edition = "2021"
version = "0.1.0"
```

- [ ] **Step 2: Write the crate manifest `emg/firmware/frame/Cargo.toml`**

```toml
[package]
name = "frame"
edition.workspace = true
version.workspace = true

[dependencies]
```

- [ ] **Step 3: Write the failing test: `lib.rs` containing only its test module**

```rust
#![no_std]

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors, identical to emg/tests/test_protocol.py
    const GOLDEN_EMG: [u8; EMG_LEN] = [
        0xa5, 0x5a, 0x01, 0x01, 0x00, 0xe8, 0x03, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x46,
    ];
    const GOLDEN_IMU: [u8; IMU_LEN] = [
        0xa5, 0x5a, 0x02, 0x02, 0x00, 0xd0, 0x07, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x2b,
    ];
    const GOLDEN_CLASS: [u8; CLASS_LEN] = [0xa5, 0x5a, 0x03, 0x03, 0x00, 0xb8, 0x0b, 0x00, 0x00, 0x02, 0x0f];

    #[test]
    fn crc8_smbus_check_value() {
        assert_eq!(crc8(b"123456789"), 0xF4);
    }

    #[test]
    fn emg_frame_matches_golden() {
        assert_eq!(encode_emg(1, 1000, &[1, 2, 3, 4]), GOLDEN_EMG);
    }

    #[test]
    fn imu_frame_matches_golden() {
        assert_eq!(encode_imu(2, 2000, &[16384, 0, 0, 0], &[0, 16384, 0, 0]), GOLDEN_IMU);
    }

    #[test]
    fn class_frame_matches_golden() {
        assert_eq!(encode_class(3, 3000, 2), GOLDEN_CLASS);
    }
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd firmware && cargo test -q`
Expected: compile errors such as `error[E0425]: cannot find function \`crc8\` in this scope`

- [ ] **Step 5: Implement: replace `emg/firmware/frame/src/lib.rs` completely**

```rust
//! Binary frame encoder. Must stay byte-identical with emgpoc/protocol.py.
//!
//! Every frame: 0xA5 0x5A | type u8 | seq u16 LE | t_us u32 LE | payload | crc8
//! The CRC-8/SMBUS covers type..payload (not the sync bytes).
#![no_std]

pub const SYNC: [u8; 2] = [0xA5, 0x5A];
pub const EMG: u8 = 0x01;
pub const IMU: u8 = 0x02;
pub const CLASS: u8 = 0x03;
pub const EMG_LEN: usize = 18;
pub const IMU_LEN: usize = 26;
pub const CLASS_LEN: usize = 11;
pub const EMG_CHANNELS: usize = 4;

pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 };
        }
    }
    crc
}

/// Writes sync + type + seq + t_us, returns the index where the payload starts.
fn header(out: &mut [u8], kind: u8, seq: u16, t_us: u32) -> usize {
    out[..2].copy_from_slice(&SYNC);
    out[2] = kind;
    out[3..5].copy_from_slice(&seq.to_le_bytes());
    out[5..9].copy_from_slice(&t_us.to_le_bytes());
    9
}

fn finish(out: &mut [u8], len: usize) {
    out[len - 1] = crc8(&out[2..len - 1]);
}

pub fn encode_emg(seq: u16, t_us: u32, ch: &[u16; EMG_CHANNELS]) -> [u8; EMG_LEN] {
    let mut out = [0u8; EMG_LEN];
    let mut n = header(&mut out, EMG, seq, t_us);
    for v in ch {
        out[n..n + 2].copy_from_slice(&v.to_le_bytes());
        n += 2;
    }
    finish(&mut out, EMG_LEN);
    out
}

pub fn encode_imu(seq: u16, t_us: u32, q_band: &[i16; 4], q_stub: &[i16; 4]) -> [u8; IMU_LEN] {
    let mut out = [0u8; IMU_LEN];
    let mut n = header(&mut out, IMU, seq, t_us);
    for v in q_band.iter().chain(q_stub) {
        out[n..n + 2].copy_from_slice(&v.to_le_bytes());
        n += 2;
    }
    finish(&mut out, IMU_LEN);
    out
}

pub fn encode_class(seq: u16, t_us: u32, class_id: u8) -> [u8; CLASS_LEN] {
    let mut out = [0u8; CLASS_LEN];
    let n = header(&mut out, CLASS, seq, t_us);
    out[n] = class_id;
    finish(&mut out, CLASS_LEN);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors, identical to emg/tests/test_protocol.py
    const GOLDEN_EMG: [u8; EMG_LEN] = [
        0xa5, 0x5a, 0x01, 0x01, 0x00, 0xe8, 0x03, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x46,
    ];
    const GOLDEN_IMU: [u8; IMU_LEN] = [
        0xa5, 0x5a, 0x02, 0x02, 0x00, 0xd0, 0x07, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x2b,
    ];
    const GOLDEN_CLASS: [u8; CLASS_LEN] = [0xa5, 0x5a, 0x03, 0x03, 0x00, 0xb8, 0x0b, 0x00, 0x00, 0x02, 0x0f];

    #[test]
    fn crc8_smbus_check_value() {
        assert_eq!(crc8(b"123456789"), 0xF4);
    }

    #[test]
    fn emg_frame_matches_golden() {
        assert_eq!(encode_emg(1, 1000, &[1, 2, 3, 4]), GOLDEN_EMG);
    }

    #[test]
    fn imu_frame_matches_golden() {
        assert_eq!(encode_imu(2, 2000, &[16384, 0, 0, 0], &[0, 16384, 0, 0]), GOLDEN_IMU);
    }

    #[test]
    fn class_frame_matches_golden() {
        assert_eq!(encode_class(3, 3000, 2), GOLDEN_CLASS);
    }
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -q`
Expected: `test result: ok. 4 passed`

- [ ] **Step 7: Commit**

```bash
git add emg/firmware/Cargo.toml emg/firmware/Cargo.lock emg/firmware/frame
git commit -m "Add no_std frame encoder, byte-identical with the Python protocol"
```

---

### Task 5: Biquad filters

**Files:**
- Create: `emg/emgpoc/filters.py`
- Test: `emg/tests/test_filters.py`

**Interfaces:**
- Produces: `Biquad(b0, b1, b2, a1, a2)`; `highpass(fs, f0, q) -> Biquad`; `notch(fs, f0, q=30) -> Biquad`; `default_stages(fs=2000.0) -> list[Biquad]` (HP 20 Hz ×2, notch 50 Hz); `FilterChain(stages, n_channels).process(x: ndarray (n, C)) -> ndarray` with state kept between calls.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from emgpoc.filters import FilterChain, default_stages

FS = 2000.0


def sine(freq: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return np.sin(2 * np.pi * freq * t)[:, None]


def tail_amplitude(y: np.ndarray) -> float:
    return float(np.max(np.abs(y[-int(FS * 0.5) :])))  # after the filter settles


def run(x: np.ndarray) -> np.ndarray:
    return FilterChain(default_stages(FS), n_channels=x.shape[1]).process(x)


def test_motion_band_is_removed():
    assert tail_amplitude(run(sine(5.0))) < 0.05


def test_emg_band_passes():
    assert 0.9 < tail_amplitude(run(sine(150.0))) < 1.1


def test_mains_50hz_is_notched():
    assert tail_amplitude(run(sine(50.0))) < 0.05


def test_dc_offset_is_removed():
    y = run(np.full((int(FS * 2), 1), 1.65))  # MyoWare RAW idles near Vs/2
    assert tail_amplitude(y) < 1e-3


def test_chunked_equals_whole():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 4))
    whole = run(x)
    chain = FilterChain(default_stages(FS), n_channels=4)
    cuts = [0, 7, 100, 101, 1999, 3000]
    chunked = np.vstack([chain.process(x[a:b]) for a, b in zip(cuts, cuts[1:])])
    np.testing.assert_allclose(chunked, whole, atol=1e-12)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_filters.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.filters'`

- [ ] **Step 3: Implement**

```python
"""Biquad IIR filters from scratch (RBJ Audio EQ Cookbook), streaming-capable."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Biquad:
    """Normalised coefficients (a0 = 1)."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def highpass(fs: float, f0: float, q: float = 1 / math.sqrt(2)) -> Biquad:
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    a0 = 1 + alpha
    return Biquad((1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0, -2 * cw / a0, (1 - alpha) / a0)


def notch(fs: float, f0: float, q: float = 30.0) -> Biquad:
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    a0 = 1 + alpha
    return Biquad(1 / a0, -2 * cw / a0, 1 / a0, -2 * cw / a0, (1 - alpha) / a0)


def default_stages(fs: float = 2000.0) -> list[Biquad]:
    """4th-order high-pass at 20 Hz (motion artefacts) + 50 Hz mains notch (EU)."""
    return [highpass(fs, 20.0), highpass(fs, 20.0), notch(fs, 50.0)]


class FilterChain:
    """Cascade of biquads applied to every channel independently.

    State persists between process() calls, so feeding a signal in chunks gives
    the same output as feeding it at once (needed for real-time use).
    """

    def __init__(self, stages: list[Biquad], n_channels: int) -> None:
        self.stages = stages
        self._z = np.zeros((len(stages), 2, n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (n_samples, n_channels). Returns filtered float64 array of the same shape."""
        y = np.asarray(x, dtype=np.float64).copy()
        for s, f in enumerate(self.stages):
            z1, z2 = self._z[s]
            for i in range(len(y)):  # transposed direct form II
                xi = y[i].copy()
                yi = f.b0 * xi + z1
                z1 = f.b1 * xi - f.a1 * yi + z2
                z2 = f.b2 * xi - f.a2 * yi
                y[i] = yi
            self._z[s, 0], self._z[s, 1] = z1, z2
        return y
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_filters.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/filters.py emg/tests/test_filters.py
git commit -m "Add from-scratch biquad high-pass and mains notch filters"
```

---

### Task 6: Uno R3 rig, Uno logger, stream health (S1 electrical check)

**Files:**
- Create: `emg/firmware/uno-logger/Cargo.toml`, `emg/firmware/uno-logger/rust-toolchain.toml`, `emg/firmware/uno-logger/.cargo/config.toml`, `emg/firmware/uno-logger/Ravedude.toml`, `emg/firmware/uno-logger/src/main.rs`, `emg/emgpoc/streamstats.py`, `emg/emgpoc/display.py`, `emg/emgpoc/serialport.py`, `emg/scripts/check_stream.py`, `emg/scripts/scope.py`
- Test: `emg/tests/test_streamstats.py`, `emg/tests/test_display.py`

**Interfaces:**
- Consumes: `frame` (Task 4); `protocol` (Task 3); `filters` (Task 5).
- Produces: the Uno logger, streaming EMG frames at 2 kHz per channel at 1 Mbaud (no IMU frames on this rig); `StreamStats(emg_frames, imu_frames, emg_rate_hz, emg_lost, imu_lost, ch_mean_v, ch_p2p_v)`; `stream_stats(frames) -> StreamStats`; `bar(value, full_scale, width=40) -> str`; `open_stream(port: str | None) -> serial.Serial` (finds a Uno or an ESP32-C6, 1 Mbaud).

- [ ] **Step 1: Get the parts (about €15; the Uno and MyoWare boards you already have)**

| Part | Qty | Note |
|---|---|---|
| 316 stainless snap studs, 12–16 mm cap | 10 | 8 signal + 1 reference + spare. Not nickel |
| Snap electrode cable | 2 | reference lead + spare |
| Neoprene strap with Velcro, 5–7 cm wide | 1 | the band |
| Jumper wires | 1 set | |

- [ ] **Step 2: Wire it**

| From | To |
|---|---|
| Uno **3.3V** pin | every MyoWare `+`, and the Uno **AREF** pin |
| Uno GND | every MyoWare `−` |
| MyoWare 1–4 `RAW` | Uno A0–A3 |

**Safety, the chip:** with AREF wired to 3.3 V, never upload an Arduino sketch that calls `analogRead()` with the default reference: it shorts the internal 5 V reference against AREF and can damage the ATmega. The Rust firmware selects AREF before the ADC is enabled, and AREF is also the chip's reset default. Power the MyoWare boards from **3.3 V**, never 5 V, so RAW never exceeds AREF. Four MyoWare boards draw about 36 mA of the 3.3 V pin's roughly 50 mA, so no IMUs on this rig.

- [ ] **Step 3: Build the dry band**

Punch the four MyoWare boards' electrode positions through the neoprene. Snap the stainless studs through from the skin side into the boards' sockets. Boards sit outside, spaced evenly around the forearm, 3–4 cm below the elbow crease, lengthwise along the muscle. Tie all four reference pads to one snap-cable lead ending in a stud on the bony point of the elbow. Mark on the band where it aligns with the ulna ridge, so it goes back on the same way.

- [ ] **Step 4: Write the failing tests**

`emg/tests/test_streamstats.py`:

```python
import pytest

from emgpoc.protocol import ADC_VOLTS_PER_COUNT, EmgFrame, ImuFrame
from emgpoc.streamstats import stream_stats


def emg_stream(n: int, drop: set[int] = frozenset(), start_us: int = 0) -> list[EmgFrame]:
    return [
        EmgFrame(i & 0xFFFF, (start_us + 500 * i) & 0xFFFFFFFF, (2048, 2048 + (i % 2) * 10, 0, 4095))
        for i in range(n)
        if i not in drop
    ]


def test_rate_levels_and_noise():
    st = stream_stats(emg_stream(2001) + [ImuFrame(0, 0, (1, 0, 0, 0), (1, 0, 0, 0))])
    assert st.emg_rate_hz == pytest.approx(2000.0)
    assert st.emg_lost == 0 and st.imu_frames == 1
    assert st.ch_mean_v[0] == pytest.approx(2048 * ADC_VOLTS_PER_COUNT)
    assert st.ch_p2p_v[1] == pytest.approx(10 * ADC_VOLTS_PER_COUNT)
    assert st.ch_p2p_v[0] == 0.0


def test_lost_frames_are_counted():
    assert stream_stats(emg_stream(100, drop={10, 11, 50})).emg_lost == 3


def test_rate_survives_clock_wrap():
    st = stream_stats(emg_stream(2001, start_us=4_294_967_295 - 100_000))
    assert st.emg_rate_hz == pytest.approx(2000.0)


def test_too_few_frames_is_an_error():
    with pytest.raises(ValueError):
        stream_stats(emg_stream(1))
```

`emg/tests/test_display.py`:

```python
from emgpoc.display import bar


def test_bar_scales_and_clips():
    assert bar(0.5, 1.0, width=4) == "##.."
    assert bar(5.0, 1.0, width=4) == "####"
    assert bar(-1.0, 1.0, width=4) == "...."
    assert bar(1.0, 0.0, width=4) == "...."
```

- [ ] **Step 5: Run them to verify they fail**

Run: `uv run pytest tests/test_streamstats.py tests/test_display.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.streamstats'`

- [ ] **Step 6: Implement**

`emg/emgpoc/streamstats.py`:

```python
"""Health check of a live stream: rate, lost frames, per-channel level and noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import ADC_VOLTS_PER_COUNT, EmgFrame, Frame, ImuFrame, U32Unwrapper, count_seq_gaps


@dataclass(frozen=True)
class StreamStats:
    emg_frames: int
    imu_frames: int
    emg_rate_hz: float
    emg_lost: int
    imu_lost: int
    ch_mean_v: tuple[float, ...]  # RAW idles near Vs/2 = 1.65 V when powered from 3.3 V
    ch_p2p_v: tuple[float, ...]  # peak-to-peak: at rest this is the noise floor


def stream_stats(frames: list[Frame]) -> StreamStats:
    emg = [f for f in frames if isinstance(f, EmgFrame)]
    imu = [f for f in frames if isinstance(f, ImuFrame)]
    if len(emg) < 2:
        raise ValueError(f"need at least 2 EMG frames, got {len(emg)}")
    unwrap = U32Unwrapper()
    t = np.array([unwrap(f.t_us) for f in emg], dtype=np.int64)
    volts = np.array([f.ch for f in emg], dtype=np.float64) * ADC_VOLTS_PER_COUNT
    return StreamStats(
        emg_frames=len(emg),
        imu_frames=len(imu),
        emg_rate_hz=(len(emg) - 1) / ((t[-1] - t[0]) / 1e6),
        emg_lost=count_seq_gaps([f.seq for f in emg]),
        imu_lost=count_seq_gaps([f.seq for f in imu]),
        ch_mean_v=tuple(float(v) for v in volts.mean(axis=0)),
        ch_p2p_v=tuple(float(v) for v in np.ptp(volts, axis=0)),
    )
```

`emg/emgpoc/display.py`:

```python
"""Terminal bars for the live scope and trainer (no GUI dependencies)."""


def bar(value: float, full_scale: float, width: int = 40) -> str:
    filled = max(0, min(width, round(width * value / full_scale))) if full_scale > 0 else 0
    return "#" * filled + "." * (width - filled)
```

`emg/emgpoc/serialport.py` (hardware I/O, exercised in Step 13):

```python
"""Find and open the logger board's USB serial port (Uno R3 first, ESP32-C6 later)."""

from __future__ import annotations

import serial
from serial.tools import list_ports

KNOWN_USB_VIDS = {
    0x2341: "Arduino Uno R3",
    0x2A03: "Arduino Uno R3 (arduino.org)",
    0x1A86: "Uno clone (CH340)",
    0x303A: "ESP32-C6 native USB",
}
BAUD = 1_000_000  # the Uno's USB-serial bridge; the ESP32-C6's native USB ignores it


def open_stream(port: str | None = None) -> serial.Serial:
    if port is None:
        found = [p.device for p in list_ports.comports() if p.vid in KNOWN_USB_VIDS]
        if not found:
            raise SystemExit("No logger board found. Plug it in, or pass --port /dev/cu.usbmodemXXXX")
        port = found[0]
    return serial.Serial(port, baudrate=BAUD, timeout=0.01)
```

- [ ] **Step 7: Run them to verify they pass**

Run: `uv run pytest tests/test_streamstats.py tests/test_display.py -q`
Expected: `5 passed`

- [ ] **Step 8: Install the AVR toolchain (macOS, once)**

```bash
brew tap osx-cross/avr && brew install avr-gcc avrdude
cargo install ravedude --locked
```

The firmware folder pins its own nightly (`rust-toolchain.toml`); rustup installs it on the first build and leaves your stable alone.

- [ ] **Step 9: Write the Uno firmware**

`emg/firmware/uno-logger/rust-toolchain.toml`:

```toml
[toolchain]
channel = "nightly-2025-04-27"
components = ["rust-src"]
profile = "minimal"
```

`emg/firmware/uno-logger/.cargo/config.toml`:

```toml
[build]
target = "avr-none"
rustflags = ["-C", "target-cpu=atmega328p"]

[target.'cfg(target_arch = "avr")']
runner = "ravedude"

[unstable]
build-std = ["core"]
```

`emg/firmware/uno-logger/Ravedude.toml`:

```toml
[general]
board = "uno"
# No console: the serial port carries binary frames at 1 Mbaud, read by the Python tools.
open-console = false
```

`emg/firmware/uno-logger/Cargo.toml`:

```toml
[package]
name = "uno-logger"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "uno-logger"
test = false
bench = false

[dependencies]
panic-halt = "1.0.0"
heapless = "0.9"
avr-device = { version = "0.8", features = ["atmega328p", "rt"] }
frame = { path = "../frame" }

[dependencies.arduino-hal]
git = "https://github.com/rahix/avr-hal"
rev = "e5c8f37fe48419956e722490a82b9ca9b9fc61a2"
features = ["arduino-uno"]

# AVRs have very little program memory: build for size
[profile.dev]
panic = "abort"
lto = true
opt-level = "s"

[profile.release]
panic = "abort"
codegen-units = 1
debug = true
lto = true
opt-level = "s"
```

`emg/firmware/uno-logger/src/main.rs`:

```rust
//! First-test logger for an Arduino Uno R3: 4 EMG channels at 2 kHz each over the USB serial
//! bridge at 1 Mbaud, in the same frames as the ESP32-C6 logger (no IMU on this rig).
//!
//! Timer1 fires every 125 us and starts one ADC conversion; the ADC interrupt stores it and
//! selects the next channel. Four slots make one sample: 500 us, 2 kHz per channel.
//! AREF must be wired to the Uno's 3.3 V pin, and the MyoWare boards powered from 3.3 V.
#![no_std]
#![no_main]
#![feature(abi_avr_interrupt)]

use core::cell::RefCell;

use arduino_hal::pac;
use avr_device::interrupt::Mutex;
use heapless::Deque;
use panic_halt as _;

const SLOT_US: u32 = 125;
const RING_LEN: usize = 32; // samples; the main loop needs 180 us per frame at 1 Mbaud

#[derive(Clone, Copy)]
struct Sample {
    t_us: u32,
    seq: u16,
    ch: [u16; frame::EMG_CHANNELS],
}

struct Acquisition {
    clock_us: u32, // wraps like the ESP32's micros(): every 71.6 min
    seq: u16,
    channel: usize,
    current: Sample,
}

static RING: Mutex<RefCell<Deque<Sample, RING_LEN>>> = Mutex::new(RefCell::new(Deque::new()));
static ACQ: Mutex<RefCell<Acquisition>> = Mutex::new(RefCell::new(Acquisition {
    clock_us: 0,
    seq: 0,
    channel: 0,
    current: Sample { t_us: 0, seq: 0, ch: [0; frame::EMG_CHANNELS] },
}));

/// External AREF (3.3 V) on every write: the internal AVcc reference would short against it.
fn select_channel(adc: &pac::adc::RegisterBlock, channel: usize) {
    adc.admux().write(|w| {
        let w = w.refs().aref();
        match channel {
            0 => w.mux().adc0(),
            1 => w.mux().adc1(),
            2 => w.mux().adc2(),
            _ => w.mux().adc3(),
        }
    });
}

#[avr_device::interrupt(atmega328p)]
fn TIMER1_COMPA() {
    avr_device::interrupt::free(|cs| {
        let mut a = ACQ.borrow(cs).borrow_mut();
        a.clock_us = a.clock_us.wrapping_add(SLOT_US);
        if a.channel == 0 {
            a.current.t_us = a.clock_us;
        }
    });
    // SAFETY: register access from an ISR; ADC and ADCSRA are not touched concurrently by main.
    let adc = unsafe { &*pac::ADC::ptr() };
    adc.adcsra().modify(|_, w| w.adsc().set_bit());
}

#[avr_device::interrupt(atmega328p)]
fn ADC() {
    // SAFETY: as above.
    let adc = unsafe { &*pac::ADC::ptr() };
    let value = adc.adc().read().bits() << 2; // 10-bit -> the 12-bit scale the host expects
    avr_device::interrupt::free(|cs| {
        let mut a = ACQ.borrow(cs).borrow_mut();
        let channel = a.channel;
        a.current.ch[channel] = value;
        a.channel = (channel + 1) % frame::EMG_CHANNELS;
        select_channel(adc, a.channel);
        if a.channel == 0 {
            let sample = Sample { seq: a.seq, ..a.current };
            a.seq = a.seq.wrapping_add(1); // a dropped sample still uses up a seq: the host sees the gap
            let _ = RING.borrow(cs).borrow_mut().push_back(sample);
        }
    });
}

#[arduino_hal::entry]
fn main() -> ! {
    let dp = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(dp);
    let mut serial = arduino_hal::default_serial!(dp, pins, 1_000_000);

    // Reference first, before the ADC is ever enabled.
    select_channel(&dp.ADC, 0);
    dp.ADC.didr0().write(|w| w.adc0d().set_bit().adc1d().set_bit().adc2d().set_bit().adc3d().set_bit());
    // 16 MHz / 128 = 125 kHz ADC clock: full 10-bit accuracy, 104 us per conversion.
    dp.ADC.adcsra().write(|w| w.aden().set_bit().adie().set_bit().adps().prescaler_128());

    // Timer1 in CTC mode: 16 MHz / 8 = 2 MHz, 250 ticks = 125 us.
    dp.TC1.tccr1a().write(|w| w.wgm1().set(0));
    dp.TC1.tccr1b().write(|w| w.wgm1().set(0b01).cs1().prescale_8());
    dp.TC1.ocr1a().write(|w| w.set(249));
    dp.TC1.timsk1().write(|w| w.ocie1a().set_bit());

    // SAFETY: all shared state is initialised; from here on the ISRs run.
    unsafe { avr_device::interrupt::enable() };

    loop {
        if let Some(s) = avr_device::interrupt::free(|cs| RING.borrow(cs).borrow_mut().pop_front()) {
            for byte in frame::encode_emg(s.seq, s.t_us, &s.ch) {
                serial.write_byte(byte);
            }
        }
    }
}
```

- [ ] **Step 10: Pin one build dependency**

`encoding_rs` ≥ 0.8.36, pulled in by avr-device's build script, needs a newer Rust than the pinned nightly:

```bash
cd firmware/uno-logger && cargo update -p encoding_rs --precise 0.8.35
```

- [ ] **Step 11: Build it**

Run: `cargo build --release`
Expected: `Finished \`release\` profile`. This plan's firmware was type-checked with `cargo check` for the ATmega328P; linking needs the avr-gcc from Step 8.

- [ ] **Step 12: Flash it**

Plug in the Uno and run `cargo run --release`. If ravedude can't find the port: `RAVEDUDE_PORT=/dev/cu.usbmodemXXXX cargo run --release`.
Expected: avrdude reports the flash verified. There's no console; the port carries binary frames.

- [ ] **Step 13: Check the stream with the band off the arm**

Write `emg/scripts/check_stream.py`:

```python
"""S1: is the logger streaming cleanly? Rate, lost frames, level and noise per channel."""

import argparse
import time

from emgpoc.protocol import FrameParser
from emgpoc.serialport import open_stream
from emgpoc.streamstats import stream_stats

ap = argparse.ArgumentParser()
ap.add_argument("--port")
ap.add_argument("--seconds", type=float, default=5.0)
args = ap.parse_args()

ser, parser, frames = open_stream(args.port), FrameParser(), []
# Opening the port resets the Uno (bootloader ~1 s); on the ESP32-C6 the queue overflowed
# while nobody listened. Either way the first moments are not representative: skip them.
warmup_end = time.monotonic() + 2.0
while time.monotonic() < warmup_end:
    parser.feed(ser.read(4096))
end = time.monotonic() + args.seconds
while time.monotonic() < end:
    frames += parser.feed(ser.read(4096))

st = stream_stats(frames)
print(f"EMG {st.emg_frames} frames at {st.emg_rate_hz:.1f} Hz, lost {st.emg_lost}")
print(f"IMU {st.imu_frames} frames, lost {st.imu_lost}")
print(f"parser: bad CRC {parser.bad_crc}, skipped bytes {parser.skipped_bytes}")
for i, (mean, p2p) in enumerate(zip(st.ch_mean_v, st.ch_p2p_v)):
    print(f"ch{i}: mean {mean:.3f} V   peak-to-peak {p2p * 1000:.1f} mV")
```

Unplug the laptop charger. Run from `prosthesis/emg`: `uv run python scripts/check_stream.py --seconds 60`
Expected: `EMG 120000 frames at 2000.0 Hz, lost 0` (±0.1 %), `IMU 0 frames` (none on this rig), `bad CRC 0`, every channel mean near 1.65 V. If frames are lost at 1 Mbaud, change both the firmware's `default_serial!(dp, pins, 1_000_000)` and `BAUD` in `serialport.py` to `500_000`.

- [ ] **Step 14: Watch muscles on the live scope**

Write `emg/scripts/scope.py`:

```python
"""S1: live RMS bar per channel after filtering. Contract a muscle, watch its bar jump."""

import argparse
import time

import numpy as np

from emgpoc.display import bar
from emgpoc.filters import FilterChain, default_stages
from emgpoc.protocol import ADC_VOLTS_PER_COUNT, EmgFrame, FrameParser
from emgpoc.serialport import open_stream

ap = argparse.ArgumentParser()
ap.add_argument("--port")
ap.add_argument("--full-scale-mv", type=float, default=200.0)
args = ap.parse_args()

ser, parser = open_stream(args.port), FrameParser()
chain, pending, last = FilterChain(default_stages(), 4), [], time.monotonic()
while True:
    pending += [f.ch for f in parser.feed(ser.read(4096)) if isinstance(f, EmgFrame)]
    if time.monotonic() - last >= 0.1 and pending:
        y = chain.process(np.array(pending) * ADC_VOLTS_PER_COUNT)
        rms_mv = np.sqrt(np.mean(y**2, axis=0)) * 1000
        print("\r" + "  ".join(f"ch{i} {bar(v, args.full_scale_mv, 12)} {v:6.1f}mV" for i, v in enumerate(rms_mv)),
              end="", flush=True)
        pending, last = [], time.monotonic()
```

Put the band on the **left** forearm (known-good signal source). Run: `uv run python scripts/scope.py`
Wait 5–10 min for the dry electrodes to settle (sweat lowers the contact resistance). Make a fist: several bars must jump well above their resting level. Relax: they must fall back.

- [ ] **Step 15: Commit**

```bash
git add emg/firmware/uno-logger emg/emgpoc/streamstats.py emg/emgpoc/display.py emg/emgpoc/serialport.py \
        emg/scripts/check_stream.py emg/scripts/scope.py emg/tests/test_streamstats.py emg/tests/test_display.py
git commit -m "Add Uno R3 Rust logger and stream health tools"
```

**Electrical check (part of S1):** 60 s with no lost frames at 2 kHz, and clear bursts on the left forearm.

---

### Task 7: Features

**Files:**
- Create: `emg/emgpoc/features.py`
- Test: `emg/tests/test_features.py`

**Interfaces:**
- Produces: `mav`, `wl`, `zc(w, thr)`, `ssc(w, thr)`; `window_features(w: (win, C), thr: (C,)) -> (4C,)` laid out `[mav…, wl…, zc…, ssc…]`; `feature_matrix(x, win, hop, thr) -> (features (n_win, 4C), window_end_index (n_win,))`; `noise_threshold(rest, k=3.0) -> (C,)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from emgpoc.features import feature_matrix, mav, noise_threshold, ssc, wl, window_features, zc

ZERO = np.array([0.0])


def col(values) -> np.ndarray:
    return np.array(values, dtype=float)[:, None]


def test_mav():
    np.testing.assert_allclose(mav(col([1, -1, 1, -1])), [1.0])


def test_wl():
    np.testing.assert_allclose(wl(col([0, 1, 0, 1])), [3.0])


def test_zc_counts_sign_changes_above_threshold():
    np.testing.assert_allclose(zc(col([1, -1, 1, -1]), ZERO), [3.0])
    np.testing.assert_allclose(zc(col([1, -1, 1, -1]), np.array([5.0])), [0.0])


def test_ssc_counts_slope_sign_changes():
    np.testing.assert_allclose(ssc(col([0, 1, 0, 1, 0]), ZERO), [3.0])
    np.testing.assert_allclose(ssc(col([0, 1, 2, 3, 4]), ZERO), [0.0])


def test_window_features_layout_is_feature_major():
    w = np.column_stack([[1.0, -1.0, 1.0, -1.0], [2.0, 2.0, 2.0, 2.0]])
    f = window_features(w, np.zeros(2))
    # [mav0, mav1, wl0, wl1, zc0, zc1, ssc0, ssc1]
    np.testing.assert_allclose(f, [1, 2, 6, 0, 3, 0, 2, 0])


def test_feature_matrix_windows_and_end_indices():
    x = np.random.default_rng(0).normal(size=(1000, 4))
    feats, ends = feature_matrix(x, win=400, hop=100, thr=np.zeros(4))
    assert feats.shape == (7, 16)
    np.testing.assert_array_equal(ends, [399, 499, 599, 699, 799, 899, 999])
    np.testing.assert_allclose(feats[1], window_features(x[100:500], np.zeros(4)))


def test_signal_shorter_than_window_gives_no_windows():
    feats, ends = feature_matrix(np.zeros((10, 4)), win=400, hop=100, thr=np.zeros(4))
    assert feats.shape == (0, 16) and len(ends) == 0


def test_noise_threshold_is_k_std():
    rest = np.column_stack([np.array([1.0, -1.0] * 50), np.zeros(100)])
    np.testing.assert_allclose(noise_threshold(rest, k=3.0), [3.0, 0.0])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.features'`

- [ ] **Step 3: Implement**

```python
"""Hudgins time-domain features on sliding windows: MAV, WL, ZC, SSC.

A window is labelled by the time of its LAST sample, as it would be in real time.
Feature vector layout for C channels: [mav_0..mav_C-1, wl_*, zc_*, ssc_*].
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("mav", "wl", "zc", "ssc")


def mav(w: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(w), axis=0)


def wl(w: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0)


def zc(w: np.ndarray, thr: np.ndarray) -> np.ndarray:
    a, b = w[:-1], w[1:]
    return np.sum((a * b < 0) & (np.abs(a - b) >= thr), axis=0).astype(np.float64)


def ssc(w: np.ndarray, thr: np.ndarray) -> np.ndarray:
    d1 = w[1:-1] - w[:-2]
    d2 = w[1:-1] - w[2:]
    hit = (d1 * d2 > 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))
    return np.sum(hit, axis=0).astype(np.float64)


def window_features(w: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """w: (win, C) filtered signal; thr: (C,) noise threshold. Returns (4*C,)."""
    return np.concatenate([mav(w), wl(w), zc(w, thr), ssc(w, thr)])


def feature_matrix(x: np.ndarray, win: int, hop: int, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (n, C). Returns (features (n_win, 4*C), index of each window's last sample (n_win,))."""
    starts = np.arange(0, len(x) - win + 1, hop)
    feats = np.array([window_features(x[s : s + win], thr) for s in starts])
    return feats.reshape(len(starts), 4 * x.shape[1]), starts + win - 1


def noise_threshold(rest: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Per-channel ZC/SSC threshold from a filtered resting signal: k standard deviations."""
    return k * np.std(rest, axis=0)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_features.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/features.py emg/tests/test_features.py
git commit -m "Add Hudgins time-domain features on sliding windows"
```

---

### Task 8: Recording protocol

**Files:**
- Create: `emg/emgpoc/session.py`
- Test: `emg/tests/test_session.py`

**Interfaces:**
- Produces: `STILL_CLASSES` (8), `COMMAND_CLASSES` (5), `MOVING_CLASSES` (7), `POSITIONS`, `CONDITIONS`, `GAP`, `INSTRUCTIONS`; `Cue(label, rep, duration_s)`; `StampedCue(label, rep, start_us, end_us)`; `classes_for(condition)`; `build_schedule(condition, reps=5, hold_s=4.0, gap_s=3.0, seed=0) -> list[Cue]`; `label_windows(window_t_us, cues, trim_us=500_000) -> (labels "<U24", reps int64)`.

- [ ] **Step 1: Write the failing test**

```python
from collections import Counter

import numpy as np
import pytest

from emgpoc.session import (
    GAP,
    INSTRUCTIONS,
    MOVING_CLASSES,
    STILL_CLASSES,
    StampedCue,
    build_schedule,
    label_windows,
)


def test_schedule_covers_every_class_once_per_rep():
    schedule = build_schedule("still", reps=5)
    holds = [c for c in schedule if c.label != GAP]
    assert Counter(c.label for c in holds) == {label: 5 for label in STILL_CLASSES}
    for rep in range(5):
        assert sorted(c.label for c in holds if c.rep == rep) == sorted(STILL_CLASSES)


def test_every_hold_is_followed_by_a_gap():
    schedule = build_schedule("moving", reps=2)
    assert [c.label == GAP for c in schedule] == [False, True] * (2 * len(MOVING_CLASSES))


def test_schedule_is_shuffled_but_reproducible():
    a = build_schedule("still", seed=1)
    assert a == build_schedule("still", seed=1)
    firsts = [[c.label for c in build_schedule("still", seed=s) if c.label != GAP][:8] for s in range(5)]
    assert len({tuple(f) for f in firsts}) > 1


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        build_schedule("upside_down")


def test_every_class_has_an_instruction():
    assert set(STILL_CLASSES) | set(MOVING_CLASSES) | {GAP} <= set(INSTRUCTIONS)


def test_label_windows_trims_reaction_time_and_skips_gaps():
    cues = [
        StampedCue("wrist_flex", 0, 0, 4_000_000),
        StampedCue(GAP, 0, 4_000_000, 7_000_000),
        StampedCue("wrist_ext", 0, 7_000_000, 11_000_000),
    ]
    t = np.array([200_000, 1_000_000, 5_000_000, 8_000_000, 12_000_000])
    labels, reps = label_windows(t, cues, trim_us=500_000)
    assert labels.tolist() == ["", "wrist_flex", "", "wrist_ext", ""]
    assert reps.tolist() == [-1, 0, -1, 0, -1]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.session'`

- [ ] **Step 3: Implement**

```python
"""Recording protocol (spec §6, S2): which gestures, in what order, and how windows get labels."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

STILL_CLASSES = (
    "rest",
    "wrist_flex",
    "wrist_ext",
    "pronation_attempt",
    "supination_attempt",
    "cocontraction",
    "imagined_fist",
    "imagined_pinch",
)
# Candidate grip commands: contractions that do not have to move the (free) wrist.
COMMAND_CLASSES = (
    "pronation_attempt",
    "supination_attempt",
    "cocontraction",
    "imagined_fist",
    "imagined_pinch",
)
# "positioning" = slow free wrist flex/extend with no command intended.
MOVING_CLASSES = ("rest", "positioning", *COMMAND_CLASSES)
POSITIONS = ("elbow90_side", "arm_down", "arm_forward")
CONDITIONS = ("still", "moving")
GAP = "gap"  # relax between holds; never used for training

INSTRUCTIONS = {
    "rest": "Relax completely, stay still",
    "wrist_flex": "Bend wrist DOWN against your left hand, hold",
    "wrist_ext": "Bend wrist UP against your left hand, hold",
    "pronation_attempt": "Try to turn palm DOWN (further than it goes), hold",
    "supination_attempt": "Try to turn palm UP (further than it goes), hold",
    "cocontraction": "Tense the whole forearm, no movement, hold",
    "imagined_fist": "Imagine making a fist, hold",
    "imagined_pinch": "Imagine pinching thumb and index, hold",
    "positioning": "Move wrist slowly up and down, NO command",
    GAP: "Relax",
}


@dataclass(frozen=True)
class Cue:
    label: str
    rep: int
    duration_s: float


@dataclass(frozen=True)
class StampedCue:
    """A cue with its start/end on the device clock (unwrapped microseconds)."""

    label: str
    rep: int
    start_us: int
    end_us: int


def classes_for(condition: str) -> tuple[str, ...]:
    if condition == "still":
        return STILL_CLASSES
    if condition == "moving":
        return MOVING_CLASSES
    raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")


def build_schedule(condition: str, reps: int = 5, hold_s: float = 4.0, gap_s: float = 3.0,
                   seed: int = 0) -> list[Cue]:
    """Each rep shows every class once, in a shuffled order, each hold followed by a gap."""
    rng = random.Random(seed)
    schedule: list[Cue] = []
    for rep in range(reps):
        order = list(classes_for(condition))
        rng.shuffle(order)
        for label in order:
            schedule.append(Cue(label, rep, hold_s))
            schedule.append(Cue(GAP, rep, gap_s))
    return schedule


def label_windows(window_t_us: np.ndarray, cues: list[StampedCue], trim_us: int = 500_000
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Label each window end time with its cue, skipping the first trim_us (reaction time).

    Returns (labels, reps). Unlabelled windows (gaps, transitions, before/after cues) get "" and -1.
    """
    labels = np.full(len(window_t_us), "", dtype="<U24")
    reps = np.full(len(window_t_us), -1, dtype=np.int64)
    for cue in cues:
        if cue.label == GAP:
            continue
        inside = (window_t_us >= cue.start_us + trim_us) & (window_t_us < cue.end_us)
        labels[inside] = cue.label
        reps[inside] = cue.rep
    return labels, reps
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_session.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/session.py emg/tests/test_session.py
git commit -m "Add recording protocol: gesture schedule and window labels"
```

---

### Task 9: IMU gating maths

**Files:**
- Create: `emg/emgpoc/gating.py`
- Test: `emg/tests/test_gating.py`

**Interfaces:**
- Produces: `quat_conj`, `quat_mul` (wxyz, `(..., 4)` arrays); `relative_angle_deg(q_band (m,4), q_stub (m,4)) -> (m,)`; `angular_speed_deg_s(angle_deg, t_us, smooth_n=5) -> (m,)`; `speed_at(window_t_us, imu_t_us, speed) -> (n,)`; `gate(pred, speed, thr_deg_s, rest_label="rest") -> pred copy`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from emgpoc.gating import angular_speed_deg_s, gate, quat_conj, quat_mul, relative_angle_deg, speed_at

IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0]])


def about_x(deg: float) -> np.ndarray:
    h = np.radians(deg) / 2
    return np.array([[np.cos(h), np.sin(h), 0.0, 0.0]])


def test_quaternion_times_its_conjugate_is_identity():
    q = about_x(37)
    np.testing.assert_allclose(quat_mul(q, quat_conj(q)), IDENTITY, atol=1e-12)


def test_relative_angle():
    assert relative_angle_deg(IDENTITY, IDENTITY)[0] == pytest.approx(0.0, abs=1e-6)
    assert relative_angle_deg(IDENTITY, about_x(90))[0] == pytest.approx(90.0)
    assert relative_angle_deg(about_x(30), about_x(30))[0] == pytest.approx(0.0, abs=1e-5)
    assert relative_angle_deg(about_x(30), about_x(-20))[0] == pytest.approx(50.0)


def test_speed_of_a_ramp():
    t_us = np.arange(0, 1_000_001, 10_000)  # 100 Hz for 1 s
    angle = np.linspace(0, 90, len(t_us))
    speed = angular_speed_deg_s(angle, t_us)
    assert speed[50] == pytest.approx(90.0)


def test_speed_at_interpolates_to_window_times():
    np.testing.assert_allclose(speed_at(np.array([5_000]), np.array([0, 10_000]), np.array([0.0, 20.0])), [10.0])


def test_gate_replaces_predictions_while_moving():
    pred = np.array(["imagined_fist", "cocontraction", "rest"])
    out = gate(pred, speed=np.array([5.0, 80.0, 90.0]), thr_deg_s=30.0)
    assert out.tolist() == ["imagined_fist", "rest", "rest"]
    assert pred.tolist()[1] == "cocontraction"  # input untouched
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gating.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.gating'`

- [ ] **Step 3: Implement**

```python
"""IMU gating: while the wrist is moving, a contraction is positioning, not a grip command.

Wrist angle proxy = rotation angle between the band IMU (forearm) and the stub IMU.
"""

from __future__ import annotations

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def relative_angle_deg(q_band: np.ndarray, q_stub: np.ndarray) -> np.ndarray:
    """(m, 4) wxyz quaternions -> (m,) angle of the stub relative to the forearm, 0..180 deg."""
    rel = quat_mul(quat_conj(q_band), q_stub)
    rel /= np.linalg.norm(rel, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(rel[:, 0]), 0.0, 1.0)))


def angular_speed_deg_s(angle_deg: np.ndarray, t_us: np.ndarray, smooth_n: int = 5) -> np.ndarray:
    speed = np.abs(np.gradient(angle_deg, t_us / 1e6))
    kernel = np.ones(smooth_n) / smooth_n
    return np.convolve(speed, kernel, mode="same")


def speed_at(window_t_us: np.ndarray, imu_t_us: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Wrist speed at each EMG window end (IMU runs at 100 Hz, windows every 50 ms)."""
    return np.interp(window_t_us, imu_t_us, speed)


def gate(pred: np.ndarray, speed: np.ndarray, thr_deg_s: float, rest_label: str = "rest") -> np.ndarray:
    out = pred.copy()
    out[speed > thr_deg_s] = rest_label
    return out
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_gating.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/gating.py emg/tests/test_gating.py
git commit -m "Add quaternion wrist-speed estimate and IMU gate"
```

---

### Task 10: Session files, dataset, recorder, re-donning check (gate S1)

**Files:**
- Create: `emg/emgpoc/sessionio.py`, `emg/emgpoc/dataset.py`, `emg/scripts/record.py`, `emg/scripts/compare_donning.py`
- Test: `emg/tests/test_sessionio.py`, `emg/tests/test_dataset.py`, `emg/tests/synth.py`, `emg/tests/conftest.py`

**Interfaces:**
- Consumes: `protocol`, `filters`, `features`, `session`, `gating`.
- Produces: `Session(meta, emg, emg_t_us, emg_seq, imu_t_us, imu_q, cues)`; `SessionBuilder(meta)` with `add_frames`, `begin_cue(label, rep)`, `end_cue()`, `last_t_us`, `build() -> Session`; `save_session(path, s)`, `load_session(path) -> Session`; `WIN = 400`, `HOP = 100`; `Dataset(X, y, group, position, condition, speed, thr)` with `.subset(mask)`; `build_dataset(sessions, win, hop) -> Dataset`; `to_product_labels(y)`; `donning_similarity(a_X, a_y, b_X, b_y, n_channels=4) -> dict[str, float]`; test helper `synth.synth_session(position, condition, reps=2, seed=0) -> Session`.

- [ ] **Step 1: Write the test helpers**

`emg/tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # lets tests import the synth helper
```

`emg/tests/synth.py`:

```python
"""Synthetic sessions with known, separable gestures, for end-to-end tests."""

from __future__ import annotations

import numpy as np

from emgpoc.protocol import ADC_VOLTS_PER_COUNT
from emgpoc.session import GAP, StampedCue, build_schedule
from emgpoc.sessionio import Session

FS = 2000
# Muscle activity (volts, std) per channel for each gesture.
PATTERN = {
    "rest": (0.002, 0.002, 0.002, 0.002),
    "wrist_flex": (0.20, 0.02, 0.02, 0.02),
    "wrist_ext": (0.02, 0.20, 0.02, 0.02),
    "pronation_attempt": (0.02, 0.02, 0.20, 0.02),
    "supination_attempt": (0.02, 0.02, 0.02, 0.20),
    "cocontraction": (0.15, 0.15, 0.15, 0.15),
    "imagined_fist": (0.20, 0.02, 0.20, 0.02),
    "imagined_pinch": (0.02, 0.20, 0.02, 0.20),
    "positioning": (0.08, 0.08, 0.002, 0.002),
    GAP: (0.002, 0.002, 0.002, 0.002),
}


def synth_session(position: str, condition: str, reps: int = 2, seed: int = 0) -> Session:
    rng = np.random.default_rng(seed)
    schedule = build_schedule(condition, reps=reps, hold_s=2.0, gap_s=1.0, seed=seed)
    chunks, cues, q_stub, t0 = [], [], [], 1_000_000
    t = t0
    for cue in schedule:
        n = int(cue.duration_s * FS)
        chunks.append(rng.normal(0, 1, size=(n, 4)) * np.array(PATTERN[cue.label]))
        cues.append(StampedCue(cue.label, cue.rep, t, t + n * 500))
        n_imu = n // 20  # 100 Hz
        phase = np.arange(n_imu) / 100 * 2 * np.pi * 0.5
        deg = 30 * np.sin(phase) if cue.label == "positioning" else np.zeros(n_imu)
        h = np.radians(deg) / 2
        q_stub.append(np.column_stack([np.cos(h), np.sin(h), np.zeros(n_imu), np.zeros(n_imu)]))
        t += n * 500
    volts = np.vstack(chunks) + 1.65
    raw = np.clip(np.round(volts / ADC_VOLTS_PER_COUNT), 0, 4095).astype(np.uint16)
    stub = np.vstack(q_stub)
    band = np.tile([1.0, 0.0, 0.0, 0.0], (len(stub), 1))
    return Session(
        meta={"arm": "right", "position": position, "condition": condition, "fs": FS},
        emg=raw,
        emg_t_us=t0 + 500 * np.arange(len(raw), dtype=np.int64),
        emg_seq=(np.arange(len(raw)) & 0xFFFF).astype(np.uint16),
        imu_t_us=t0 + 10_000 * np.arange(len(stub), dtype=np.int64),
        imu_q=np.hstack([band, stub]),
        cues=cues,
    )
```

- [ ] **Step 2: Write the failing tests**

`emg/tests/test_sessionio.py`:

```python
import numpy as np
import pytest

from emgpoc.protocol import ClassFrame, EmgFrame, ImuFrame
from emgpoc.session import StampedCue
from emgpoc.sessionio import SessionBuilder, load_session, save_session

META = {"arm": "right", "position": "arm_down", "condition": "still", "fs": 2000}


def test_builder_collects_frames_and_stamps_cues():
    b = SessionBuilder(META)
    b.add_frames([EmgFrame(0, 1_000, (1, 2, 3, 4)), ImuFrame(0, 1_500, (1, 0, 0, 0), (0, 1, 0, 0))])
    b.begin_cue("wrist_flex", 0)
    b.add_frames([EmgFrame(1, 1_500, (5, 6, 7, 8)), ClassFrame(0, 1_600, 1), EmgFrame(2, 2_000, (9, 9, 9, 9))])
    b.end_cue()
    s = b.build()
    assert s.emg.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]]
    assert s.emg_t_us.tolist() == [1_000, 1_500, 2_000]
    assert s.imu_q.tolist() == [[1, 0, 0, 0, 0, 1, 0, 0]]
    assert s.cues == [StampedCue("wrist_flex", 0, 1_500, 2_000)]


def test_cue_before_any_frame_is_an_error():
    with pytest.raises(RuntimeError):
        SessionBuilder(META).begin_cue("rest", 0)


def test_save_load_roundtrip(tmp_path):
    b = SessionBuilder(META)
    b.add_frames([EmgFrame(i, (4_294_966_000 + 500 * i) & 0xFFFFFFFF, (i, i, i, i)) for i in range(4)])  # wraps
    b.begin_cue("rest", 3)
    b.add_frames([EmgFrame(4, 1_704, (4, 4, 4, 4))])
    b.end_cue()
    s = b.build()
    path = tmp_path / "s.npz"
    save_session(path, s)
    loaded = load_session(path)
    assert loaded.meta == META
    np.testing.assert_array_equal(loaded.emg, s.emg)
    np.testing.assert_array_equal(loaded.emg_t_us, s.emg_t_us)
    assert np.all(np.diff(loaded.emg_t_us) > 0)
    assert loaded.cues == s.cues
    assert loaded.imu_q.shape == (0, 8)
```

`emg/tests/test_dataset.py`:

```python
import numpy as np
import pytest

from emgpoc.dataset import build_dataset, donning_similarity, to_product_labels
from emgpoc.session import StampedCue
from synth import synth_session


def test_dataset_holds_only_labelled_windows_with_groups_and_speed():
    ds = build_dataset([synth_session("arm_down", "still", seed=0), synth_session("arm_down", "moving", seed=1)])
    assert ds.X.shape == (len(ds.y), 16)
    assert "" not in set(ds.y.tolist())
    assert set(ds.condition.tolist()) == {"still", "moving"}
    assert all(g.startswith("arm_down:") for g in ds.group)
    assert np.all(np.isfinite(ds.speed)) and ds.thr.shape == (4,)
    moving_speed = ds.speed[ds.y == "positioning"]
    assert moving_speed.mean() > 10 * ds.speed[ds.y == "rest"].mean()


def test_session_without_rest_cannot_estimate_noise():
    s = synth_session("arm_down", "still", seed=0)
    s.cues = [c for c in s.cues if c.label != "rest"]
    with pytest.raises(ValueError):
        build_dataset([s])


def test_session_without_imu_gets_nan_speed():
    s = synth_session("arm_down", "still", seed=0)
    s.imu_t_us, s.imu_q = s.imu_t_us[:0], s.imu_q[:0]
    assert np.all(np.isnan(build_dataset([s]).speed))


def test_product_labels_fold_non_commands_into_rest():
    y = np.array(["wrist_flex", "positioning", "rest", "imagined_fist"])
    assert to_product_labels(y).tolist() == ["rest", "rest", "rest", "imagined_fist"]


def test_donning_similarity():
    a_X = np.array([[1.0, 0.0, 0.0, 0.0, 9, 9], [0.0, 1.0, 0.0, 0.0, 9, 9]])
    b_X = np.array([[2.0, 0.0, 0.0, 0.0, 1, 1], [1.0, 1.0, 0.0, 0.0, 1, 1]])
    y = np.array(["flex", "ext"])
    sim = donning_similarity(a_X, y, b_X, y)
    assert sim["flex"] == pytest.approx(1.0)  # same proportions, twice as strong
    assert sim["ext"] == pytest.approx(np.sqrt(0.5))
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_sessionio.py tests/test_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.sessionio'`

- [ ] **Step 4: Implement**

`emg/emgpoc/sessionio.py`:

```python
"""Collect frames into a recording session and persist it as .npz."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .protocol import EmgFrame, Frame, ImuFrame, U32Unwrapper
from .session import StampedCue


@dataclass
class Session:
    meta: dict
    emg: np.ndarray  # (n, 4) uint16 raw ADC counts
    emg_t_us: np.ndarray  # (n,) int64, unwrapped device time
    emg_seq: np.ndarray  # (n,) uint16
    imu_t_us: np.ndarray  # (m,) int64
    imu_q: np.ndarray  # (m, 8) float64: band wxyz, stub wxyz
    cues: list[StampedCue] = field(default_factory=list)


class SessionBuilder:
    """Accumulates frames as they arrive and stamps cues on the device clock."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta
        self._unwrap = U32Unwrapper()
        self._emg: list[tuple[int, ...]] = []
        self._emg_t: list[int] = []
        self._emg_seq: list[int] = []
        self._imu_t: list[int] = []
        self._imu_q: list[tuple[float, ...]] = []
        self._cues: list[StampedCue] = []
        self._open: tuple[str, int, int] | None = None
        self.last_t_us: int | None = None

    def add_frames(self, frames: list[Frame]) -> None:
        for f in frames:
            t = self._unwrap(f.t_us)
            self.last_t_us = t
            if isinstance(f, EmgFrame):
                self._emg.append(f.ch)
                self._emg_t.append(t)
                self._emg_seq.append(f.seq)
            elif isinstance(f, ImuFrame):
                self._imu_t.append(t)
                self._imu_q.append(f.q_band + f.q_stub)

    def begin_cue(self, label: str, rep: int) -> None:
        if self.last_t_us is None:
            raise RuntimeError("no frames received yet; is the logger streaming?")
        self._open = (label, rep, self.last_t_us)

    def end_cue(self) -> None:
        if self._open is None:
            raise RuntimeError("end_cue() without begin_cue()")
        label, rep, start = self._open
        self._cues.append(StampedCue(label, rep, start, self.last_t_us))
        self._open = None

    def build(self) -> Session:
        return Session(
            meta=dict(self.meta),
            emg=np.array(self._emg, dtype=np.uint16).reshape(-1, 4),
            emg_t_us=np.array(self._emg_t, dtype=np.int64),
            emg_seq=np.array(self._emg_seq, dtype=np.uint16),
            imu_t_us=np.array(self._imu_t, dtype=np.int64),
            imu_q=np.array(self._imu_q, dtype=np.float64).reshape(-1, 8),
            cues=list(self._cues),
        )


def save_session(path: Path, s: Session) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=json.dumps(s.meta),
        emg=s.emg,
        emg_t_us=s.emg_t_us,
        emg_seq=s.emg_seq,
        imu_t_us=s.imu_t_us,
        imu_q=s.imu_q,
        cue_label=np.array([c.label for c in s.cues], dtype="<U24"),
        cue_rep=np.array([c.rep for c in s.cues], dtype=np.int64),
        cue_start_us=np.array([c.start_us for c in s.cues], dtype=np.int64),
        cue_end_us=np.array([c.end_us for c in s.cues], dtype=np.int64),
    )


def load_session(path: Path) -> Session:
    with np.load(path) as z:
        cues = [
            StampedCue(str(label), int(rep), int(start), int(end))
            for label, rep, start, end in zip(z["cue_label"], z["cue_rep"], z["cue_start_us"], z["cue_end_us"])
        ]
        return Session(
            meta=json.loads(str(z["meta"])),
            emg=z["emg"],
            emg_t_us=z["emg_t_us"],
            emg_seq=z["emg_seq"],
            imu_t_us=z["imu_t_us"],
            imu_q=z["imu_q"],
            cues=cues,
        )
```

`emg/emgpoc/dataset.py`:

```python
"""Turn recorded sessions into a labelled feature table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import feature_matrix, noise_threshold
from .filters import FilterChain, default_stages
from .gating import angular_speed_deg_s, relative_angle_deg, speed_at
from .protocol import ADC_VOLTS_PER_COUNT
from .session import COMMAND_CLASSES, label_windows
from .sessionio import Session

WIN = 400  # 200 ms at 2 kHz
HOP = 100  # a decision every 50 ms


@dataclass
class Dataset:
    X: np.ndarray  # (n, 16) features
    y: np.ndarray  # class names as recorded
    group: np.ndarray  # "position:condition:rep", the unit held out in cross-validation
    position: np.ndarray
    condition: np.ndarray
    speed: np.ndarray  # wrist speed (deg/s) at each window end; nan without IMU data
    thr: np.ndarray  # ZC/SSC noise threshold per channel

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.X[mask], self.y[mask], self.group[mask], self.position[mask],
                       self.condition[mask], self.speed[mask], self.thr)


def filtered(s: Session) -> np.ndarray:
    fs = float(s.meta.get("fs", 2000))
    volts = s.emg.astype(np.float64) * ADC_VOLTS_PER_COUNT
    return FilterChain(default_stages(fs), n_channels=volts.shape[1]).process(volts)


def rest_threshold(s: Session, x: np.ndarray) -> np.ndarray:
    """Noise threshold from the samples recorded during 'rest' cues."""
    rest = np.zeros(len(x), dtype=bool)
    for cue in s.cues:
        if cue.label == "rest":
            rest |= (s.emg_t_us >= cue.start_us + 500_000) & (s.emg_t_us < cue.end_us)
    if not rest.any():
        raise ValueError(f"session {s.meta} has no rest cue to estimate noise from")
    return noise_threshold(x[rest])


def build_dataset(sessions: list[Session], win: int = WIN, hop: int = HOP) -> Dataset:
    signals = [filtered(s) for s in sessions]
    thr = np.median([rest_threshold(s, x) for s, x in zip(sessions, signals)], axis=0)
    parts = []
    for s, x in zip(sessions, signals):
        F, ends = feature_matrix(x, win, hop, thr)
        t = s.emg_t_us[ends]
        labels, reps = label_windows(t, s.cues)
        if len(s.imu_t_us) >= 2:
            angle = relative_angle_deg(s.imu_q[:, :4], s.imu_q[:, 4:])
            speed = speed_at(t, s.imu_t_us, angular_speed_deg_s(angle, s.imu_t_us))
        else:
            speed = np.full(len(t), np.nan)
        keep = labels != ""
        pos, cond = s.meta["position"], s.meta["condition"]
        groups = np.array([f"{pos}:{cond}:{r}" for r in reps[keep]], dtype="<U48")
        n = int(keep.sum())
        parts.append((F[keep], labels[keep], groups, np.full(n, pos, dtype="<U24"),
                      np.full(n, cond, dtype="<U24"), speed[keep]))
    cols = [np.concatenate(c) for c in zip(*parts)]
    return Dataset(*cols, thr=thr)


def to_product_labels(y: np.ndarray) -> np.ndarray:
    """In the free-wrist design only command contractions are grips; everything else means 'no command'."""
    return np.where(np.isin(y, COMMAND_CLASSES), y, "rest").astype("<U24")


def donning_similarity(a_X: np.ndarray, a_y: np.ndarray, b_X: np.ndarray, b_y: np.ndarray,
                       n_channels: int = 4) -> dict[str, float]:
    """Per class: cosine similarity of the mean MAV profile across channels, band taken off and put back on.

    1.0 = the gesture lights up the channels in exactly the same proportions after re-donning.
    """
    out = {}
    for label in sorted(set(a_y.tolist()) & set(b_y.tolist())):
        u = a_X[a_y == label, :n_channels].mean(axis=0)
        v = b_X[b_y == label, :n_channels].mean(axis=0)
        out[label] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return out
```

- [ ] **Step 5: Run them to verify they pass**

Run: `uv run pytest tests/test_sessionio.py tests/test_dataset.py -q`
Expected: `8 passed`

- [ ] **Step 6: Write the recorder and the re-donning check**

`emg/scripts/record.py`:

```python
"""S1/S2: guided recording. Shows each gesture, stamps it on the device clock, saves a session file."""

import argparse
import time
from datetime import datetime
from pathlib import Path

from emgpoc.protocol import FrameParser
from emgpoc.serialport import open_stream
from emgpoc.session import CONDITIONS, GAP, INSTRUCTIONS, POSITIONS, build_schedule
from emgpoc.sessionio import SessionBuilder, save_session
from emgpoc.streamstats import stream_stats

ap = argparse.ArgumentParser()
ap.add_argument("--port")
ap.add_argument("--arm", choices=["right", "left"], required=True)
ap.add_argument("--position", choices=POSITIONS, required=True)
ap.add_argument("--condition", choices=CONDITIONS, required=True)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--donning", type=int, default=1, help="1 = first time the band went on today, 2 = after re-donning")
ap.add_argument("--out", type=Path, default=Path("data/sessions"))
args = ap.parse_args()

stamp = datetime.now()
meta = {"arm": args.arm, "position": args.position, "condition": args.condition, "fs": 2000,
        "donning": args.donning, "recorded": stamp.isoformat(timespec="seconds")}
ser, parser, builder = open_stream(args.port), FrameParser(), SessionBuilder(meta)
frames_all = []


def pump(seconds: float, label: str = "", count: bool = True) -> None:
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        frames = parser.feed(ser.read(4096))
        if count:
            frames_all.extend(frames)
        builder.add_frames(frames)
        if label:
            print(f"\r  {label:<60} {left:4.1f}s", end="", flush=True)


print("Warming up the stream (3 s)...")
pump(3.0, count=False)  # the first burst is the chip's stale queue, not a real gap
schedule = build_schedule(args.condition, reps=args.reps, seed=int(stamp.timestamp()))
for i, cue in enumerate(schedule):
    if cue.label != GAP:
        print(f"\a\n[{i // 2 + 1}/{len(schedule) // 2}] {cue.label.upper()}")
    builder.begin_cue(cue.label, cue.rep)
    pump(cue.duration_s, INSTRUCTIONS[cue.label])
    builder.end_cue()

name = f"{stamp:%Y%m%d_%H%M%S}_{args.arm}_{args.position}_{args.condition}_d{args.donning}.npz"
save_session(args.out / name, builder.build())
st = stream_stats(frames_all)
print(f"\n\nSaved {args.out / name}")
print(f"EMG {st.emg_rate_hz:.1f} Hz, lost {st.emg_lost}; IMU frames {st.imu_frames}; bad CRC {parser.bad_crc}")
if st.emg_lost:
    print("WARNING: frames were lost. Close other USB-heavy apps and record again.")
```

`emg/scripts/compare_donning.py`:

```python
"""S1 gate: do gestures look the same after taking the band off and putting it back on?"""

import argparse
from pathlib import Path

from emgpoc.dataset import build_dataset, donning_similarity
from emgpoc.sessionio import load_session

GATE = 0.90

ap = argparse.ArgumentParser()
ap.add_argument("first", type=Path)
ap.add_argument("second", type=Path)
args = ap.parse_args()

a, b = build_dataset([load_session(args.first)]), build_dataset([load_session(args.second)])
sim = donning_similarity(a.X, a.y, b.X, b.y)
for label, value in sim.items():
    print(f"{label:<20} {value:.3f} {'ok' if value >= GATE else 'LOW'}")
worst = min(sim.values())
print(f"\nGate S1 (every class >= {GATE}): {'PASS' if worst >= GATE else 'FAIL'} (worst {worst:.3f})")
```

- [ ] **Step 7: Record the left arm twice, taking the band off in between**

Charger unplugged. Band on the left forearm, aligned with the ulna mark, 10 min to settle. Then:

```bash
uv run python scripts/record.py --arm left --position elbow90_side --condition still --reps 2 --donning 1
# take the band off, put it back on at the marks, wait 10 min
uv run python scripts/record.py --arm left --position elbow90_side --condition still --reps 2 --donning 2
```

Expected each time: `lost 0`, `bad CRC 0`.

- [ ] **Step 8: Compare**

Run: `uv run python scripts/compare_donning.py data/sessions/<first>_d1.npz data/sessions/<second>_d2.npz`
Expected: one line per class, ending `Gate S1 (every class >= 0.9): PASS`.

If a class is LOW: check that the band went back on at the marks, trim hair under the electrodes, tighten the band, and repeat.

- [ ] **Step 9: Commit**

```bash
git add emg/emgpoc/sessionio.py emg/emgpoc/dataset.py emg/scripts/record.py emg/scripts/compare_donning.py emg/tests
git commit -m "Add guided recorder, session files, dataset builder and re-donning check"
```

**Gate S1:** electrical check (Task 6) passed, and re-donning similarity ≥ 0.90 for every class.

---

### Task 11: Record the right arm (S2, manual)

**Files:** none (data only, in the gitignored `emg/data/sessions/` and `emg/data/sessions_unsettled/`).

- [ ] **Step 1: Record the unsettled minutes first**

Charger unplugged. Put the band on the right forearm at the marks. **Immediately**, before the electrodes settle, record one short session into a separate folder. The real prosthesis meets these minutes every morning; they are kept for later comparison, not for the S3 gate:

```bash
uv run python scripts/record.py --arm right --position elbow90_side --condition still --reps 2 --out data/sessions_unsettled
```

Then wait 10 min. Run `uv run python scripts/check_stream.py`: `lost 0`.

- [ ] **Step 2: Six sessions, 2 min break between them, band stays on**

```bash
for pos in elbow90_side arm_down arm_forward; do
  uv run python scripts/record.py --arm right --position $pos --condition still
  uv run python scripts/record.py --arm right --position $pos --condition moving
done
```

- **Still:** the left hand blocks the stub, so wrist gestures are isometric.
- **Moving:** during `positioning`, move the wrist slowly with no command. For the command gestures, keep the wrist moving slowly while contracting. The Uno rig has no IMUs, so the report measures false triggers without gating; that still shows how often plain wrist movement fires a grip.

- [ ] **Step 3: Left-arm reference**

`uv run python scripts/record.py --arm left --position elbow90_side --condition still`

- [ ] **Step 4: Back up and log**

Copy `emg/data/sessions/` and `emg/data/sessions_unsettled/` to an external disk. In `data/phase0.md`, add a line with the date, session count, and any session that showed lost frames (re-record those).

**Deliverable:** 6 right-arm sessions plus 1 left-arm session, all with `lost 0`.

---

### Task 12: Standardiser, LDA, model files

**Files:**
- Create: `emg/emgpoc/standardize.py`, `emg/emgpoc/lda.py`, `emg/emgpoc/models.py`
- Test: `emg/tests/test_lda.py`

**Interfaces:**
- Produces: `Standardizer.fit/transform/fit_transform` (`mean_`, `scale_`, constant feature → scale 1); `LDA(shrinkage=0.1).fit(X, y)`, `.decision_function(X)`, `.predict(X)`, attributes `classes_`, `W_ (d, k)`, `b_ (k,)`; `StandardizedModel(model)` with `.fit`, `.predict`, `.classes_`, `.scaler`, `.model`; `make_lda(shrinkage=0.1) -> Callable[[], StandardizedModel]`; `save_lda(path, m, thr, win, hop)`; `load_lda(path) -> (StandardizedModel, thr, win, hop)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from emgpoc.lda import LDA
from emgpoc.models import StandardizedModel, load_lda, make_lda, save_lda
from emgpoc.standardize import Standardizer


def blobs(n_per: int = 100, centres=((0, 0), (4, 0), (0, 4)), seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.7, size=(n_per, 2)) for c in centres])
    y = np.repeat(np.array([f"c{i}" for i in range(len(centres))]), n_per)
    return X, y


def test_standardizer_zero_mean_unit_std():
    X = np.random.default_rng(0).normal(5, 3, size=(200, 3))
    Z = Standardizer().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)


def test_standardizer_survives_constant_feature():
    X = np.column_stack([np.arange(10.0), np.full(10, 7.0)])
    Z = Standardizer().fit_transform(X)
    assert np.all(np.isfinite(Z)) and np.all(Z[:, 1] == 0)


def test_lda_separates_blobs():
    X, y = blobs()
    Xt, yt = blobs(seed=1)
    assert np.mean(LDA().fit(X, y).predict(Xt) == yt) > 0.97


def test_lda_predicts_only_known_labels():
    X, y = blobs()
    assert set(LDA().fit(X, y).predict(X)) <= set(y)


def test_lda_survives_dead_channel():
    X, y = blobs()
    X = np.column_stack([X, np.zeros(len(X))])  # a disconnected electrode
    assert np.mean(LDA().fit(X, y).predict(X) == y) > 0.97


def test_saved_model_predicts_identically(tmp_path):
    X, y = blobs()
    m = make_lda()().fit(X, y)
    path = tmp_path / "model.npz"
    save_lda(path, m, thr=np.array([0.1, 0.2]), win=400, hop=100)
    loaded, thr, win, hop = load_lda(path)
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    assert (win, hop) == (400, 100) and thr.tolist() == [0.1, 0.2]
    assert isinstance(loaded, StandardizedModel)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_lda.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.lda'`

- [ ] **Step 3: Implement**

`emg/emgpoc/standardize.py`:

```python
"""Per-feature z-scoring. A constant feature (dead channel) gets scale 1 instead of dividing by 0."""

from __future__ import annotations

import numpy as np


class Standardizer:
    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
```

`emg/emgpoc/lda.py`:

```python
"""Linear discriminant analysis from scratch, with covariance shrinkage.

Scores: g_k(x) = x^T S^-1 m_k - 1/2 m_k^T S^-1 m_k + log p_k, with S the pooled
within-class covariance shrunk toward (trace(S)/d) I, so a dead channel or too
few windows never makes S singular.
"""

from __future__ import annotations

import numpy as np


class LDA:
    def __init__(self, shrinkage: float = 0.1) -> None:
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDA":
        self.classes_, idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        means = np.array([X[idx == c].mean(axis=0) for c in range(k)])
        centred = X - means[idx]
        cov = centred.T @ centred / max(n - k, 1)
        target = np.trace(cov) / d if np.trace(cov) > 0 else 1.0
        cov = (1 - self.shrinkage) * cov + self.shrinkage * target * np.eye(d)
        self.W_ = np.linalg.solve(cov, means.T)  # (d, k)
        priors = np.bincount(idx, minlength=k) / n
        self.b_ = -0.5 * np.sum(means * self.W_.T, axis=1) + np.log(priors)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
```

`emg/emgpoc/models.py`:

```python
"""Standardise-then-classify pipeline, plus saving an LDA model for the trainer and firmware."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .lda import LDA
from .standardize import Standardizer


class StandardizedModel:
    def __init__(self, model) -> None:
        self.model = model
        self.scaler = Standardizer()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardizedModel":
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_


def make_lda(shrinkage: float = 0.1) -> Callable[[], StandardizedModel]:
    return lambda: StandardizedModel(LDA(shrinkage))


def save_lda(path: Path, m: StandardizedModel, thr: np.ndarray, win: int, hop: int) -> None:
    """Everything the live trainer and the firmware export need to reproduce predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=m.scaler.mean_,
        scale=m.scaler.scale_,
        W=m.model.W_,
        b=m.model.b_,
        classes=json.dumps([str(c) for c in m.classes_]),
        thr=thr,
        win=win,
        hop=hop,
    )


def load_lda(path: Path) -> tuple[StandardizedModel, np.ndarray, int, int]:
    with np.load(path) as z:
        lda = LDA()
        lda.W_, lda.b_ = z["W"], z["b"]
        lda.classes_ = np.array(json.loads(str(z["classes"])), dtype="<U24")
        m = StandardizedModel(lda)
        m.scaler.mean_, m.scaler.scale_ = z["mean"], z["scale"]
        return m, z["thr"], int(z["win"]), int(z["hop"])
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_lda.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/standardize.py emg/emgpoc/lda.py emg/emgpoc/models.py emg/tests/test_lda.py
git commit -m "Add from-scratch shrinkage LDA, standardiser and model files"
```

---

### Task 13: MLP from scratch

**Files:**
- Create: `emg/emgpoc/mlp.py`
- Test: `emg/tests/test_mlp.py`

**Interfaces:**
- Produces: `MLP(n_hidden=32, epochs=300, lr=1e-2, batch=64, weight_decay=1e-4, seed=0)` with `.init_params(n_in, n_out)`, `.loss_and_grads(X, y_idx) -> (float, dict)`, `.fit(X, y)`, `.predict_proba(X)`, `.predict(X)`, `.classes_`. Wrap it as `StandardizedModel(MLP())`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from emgpoc.mlp import MLP


def numeric_grad(net: MLP, X, y_idx, name: str, eps: float = 1e-6) -> np.ndarray:
    param = net.params[name]
    grad = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + eps
        up, _ = net.loss_and_grads(X, y_idx)
        param[i] = old - eps
        down, _ = net.loss_and_grads(X, y_idx)
        param[i] = old
        grad[i] = (up - down) / (2 * eps)
    return grad


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X, y_idx = rng.normal(size=(5, 3)), np.array([0, 1, 2, 1, 0])
    net = MLP(n_hidden=4, weight_decay=1e-3)
    net.init_params(3, 3)
    _, grads = net.loss_and_grads(X, y_idx)
    for name, analytic in grads.items():
        numeric = numeric_grad(net, X, y_idx, name)
        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))
        assert rel < 1e-6, name


def test_learns_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array(["off", "on", "on", "off"])
    net = MLP(n_hidden=8, epochs=1500, lr=0.05, batch=4, weight_decay=0.0).fit(X, y)
    assert net.predict(X).tolist() == y.tolist()


def test_probabilities_sum_to_one():
    X = np.random.default_rng(1).normal(size=(20, 3))
    y = np.array(["a", "b"] * 10)
    probs = MLP(epochs=5).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_mlp.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.mlp'`

- [ ] **Step 3: Implement**

```python
"""One-hidden-layer MLP from scratch: tanh hidden layer, softmax output, Adam."""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_hidden: int = 32, epochs: int = 300, lr: float = 1e-2, batch: int = 64,
                 weight_decay: float = 1e-4, seed: int = 0) -> None:
        self.n_hidden, self.epochs, self.lr, self.batch = n_hidden, epochs, lr, batch
        self.weight_decay, self.seed = weight_decay, seed

    def init_params(self, n_in: int, n_out: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, self.n_hidden)),
            "b1": np.zeros(self.n_hidden),
            "W2": rng.normal(0, 1 / np.sqrt(self.n_hidden), size=(self.n_hidden, n_out)),
            "b2": np.zeros(n_out),
        }

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        h = np.tanh(X @ p["W1"] + p["b1"])
        logits = h @ p["W2"] + p["b2"]
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True), h

    def loss_and_grads(self, X: np.ndarray, y_idx: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        p, n, wd = self.params, len(X), self.weight_decay
        probs, h = self._forward(X)
        loss = -np.mean(np.log(probs[np.arange(n), y_idx] + 1e-12))
        loss += 0.5 * wd * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))
        dlogits = probs.copy()
        dlogits[np.arange(n), y_idx] -= 1
        dlogits /= n
        dz = (dlogits @ p["W2"].T) * (1 - h**2)  # tanh' = 1 - tanh^2
        grads = {
            "W2": h.T @ dlogits + wd * p["W2"],
            "b2": dlogits.sum(axis=0),
            "W1": X.T @ dz + wd * p["W1"],
            "b1": dz.sum(axis=0),
        }
        return float(loss), grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.init_params(X.shape[1], len(self.classes_))
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(val) for k, val in self.params.items()}
        rng = np.random.default_rng(self.seed)
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), self.batch):
                batch = order[s : s + self.batch]
                _, grads = self.loss_and_grads(X[batch], y_idx[batch])
                step += 1
                for k, g in grads.items():  # Adam
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g**2
                    m_hat = m[k] / (1 - 0.9**step)
                    v_hat = v[k] / (1 - 0.999**step)
                    self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_mlp.py -q`
Expected: `3 passed` (the gradient check proves the backprop maths)

- [ ] **Step 5: Commit**

```bash
git add emg/emgpoc/mlp.py emg/tests/test_mlp.py
git commit -m "Add from-scratch MLP with gradient-checked backprop and Adam"
```

---

### Task 14: Evaluation and the S3 report (gate S3a)

**Files:**
- Create: `emg/emgpoc/evaluate.py`, `emg/emgpoc/report.py`, `emg/scripts/analyze.py`
- Test: `emg/tests/test_evaluate.py`, `emg/tests/test_report.py`

**Interfaces:**
- Consumes: `dataset`, `lda`/`models`, `mlp`, `gating`.
- Produces: `accuracy`, `confusion_matrix(y_true, y_pred, classes)`, `leave_one_group_out(groups)`, `cv_predict`, `cv_accuracy(X, y, groups, make_model)`, `backward_elimination(X, y, groups, make_model, keep=("rest",), min_classes=2) -> list[(classes, acc)]`, `cross_position_matrix(X, y, position, groups, make_model) -> (positions, acc)`, `false_trigger_rate(y_true, y_pred, commands)`, `command_recall(...)`; `run_analysis(sessions, n_classes=4) -> S3Result(report, chosen, model, thr, accuracy_passed, gating_passed: bool | None, passed)`; `data/models/lda_s3.npz` via `save_lda`.

- [ ] **Step 1: Write the failing tests**

`emg/tests/test_evaluate.py`:

```python
import numpy as np
import pytest

from emgpoc.evaluate import (
    accuracy,
    backward_elimination,
    command_recall,
    confusion_matrix,
    cross_position_matrix,
    cv_accuracy,
    false_trigger_rate,
    leave_one_group_out,
)
from emgpoc.models import make_lda


def test_accuracy_and_confusion():
    t = np.array(["a", "a", "b"])
    p = np.array(["a", "b", "b"])
    assert accuracy(t, p) == pytest.approx(2 / 3)
    assert confusion_matrix(t, p, ["a", "b"]).tolist() == [[1, 1], [0, 1]]


def test_leave_one_group_out_tests_every_window_once_without_overlap():
    groups = np.array(["g0", "g0", "g1", "g2", "g2", "g2"])
    folds = leave_one_group_out(groups)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(6))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def test_cross_validation_does_not_leak_windows_of_the_same_hold():
    # Each hold has its own random centre and a random label: nothing generalises across holds.
    # A random window split scores 1.0 here (verified); holding out whole holds must stay near chance.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(40)], 20)
    centres = rng.normal(size=(40, 64)) * 5
    X = centres.repeat(20, axis=0) + rng.normal(scale=0.01, size=(800, 64))
    y = np.repeat(rng.choice(["a", "b"], size=40), 20)
    assert cv_accuracy(X, y, groups, make_lda()) < 0.8


def separable_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = {"rest": (0, 0), "fist": (6, 0), "pinch": (6.3, 0.2), "cocontraction": (0, 6)}
    X, y, g = [], [], []
    for rep in range(5):
        for label, c in centres.items():
            X.append(rng.normal(c, 1.0, size=(30, 2)))
            y += [label] * 30
            g += [f"p:still:{rep}"] * 30
    return np.vstack(X), np.array(y), np.array(g)


def test_backward_elimination_drops_the_confusable_class_and_keeps_rest():
    X, y, g = separable_dataset()
    path = backward_elimination(X, y, g, make_lda())
    assert len(path[0][0]) == 4
    assert "rest" in path[-1][0]
    dropped = set(path[0][0]) - set(path[1][0])
    assert dropped <= {"fist", "pinch"}
    assert path[1][1] > path[0][1]


def test_cross_position_shift_shows_up_off_diagonal():
    X, y, g = separable_dataset()
    X2 = X + np.array([3.0, 3.0])  # same gestures, signals moved (arm in another position)
    pos = np.array(["down"] * len(X) + ["up"] * len(X))
    groups = np.concatenate([np.char.add("down:", g), np.char.add("up:", g)])
    positions, acc = cross_position_matrix(np.vstack([X, X2]), np.concatenate([y, y]), pos, groups, make_lda())
    assert positions == ["down", "up"]
    assert min(acc[0, 0], acc[1, 1]) > max(acc[0, 1], acc[1, 0])


def test_false_triggers_and_command_recall():
    commands = ("imagined_fist", "cocontraction")
    y_true = np.array(["positioning", "positioning", "rest", "imagined_fist", "cocontraction"])
    y_pred = np.array(["imagined_fist", "rest", "rest", "imagined_fist", "rest"])
    assert false_trigger_rate(y_true, y_pred, commands) == pytest.approx(1 / 3)
    assert command_recall(y_true, y_pred, commands) == pytest.approx(1 / 2)
```

`emg/tests/test_report.py`:

```python
import numpy as np

from emgpoc.report import run_analysis
from emgpoc.session import CONDITIONS
from synth import synth_session


def test_report_end_to_end_on_separable_synthetic_data():
    sessions = [
        synth_session(pos, cond, seed=i)
        for i, (pos, cond) in enumerate((p, c) for p in ("arm_down", "elbow90_side") for c in CONDITIONS)
    ]
    result = run_analysis(sessions)
    assert len(result.chosen) == 4 and "rest" in result.chosen
    assert "## Gate S3" in result.report
    assert result.accuracy_passed and result.gating_passed is True
    assert result.passed, result.report
    assert result.thr.shape == (4,)
    assert set(result.model.classes_) == set(result.chosen)


def test_report_fails_the_gate_when_gestures_are_indistinguishable(monkeypatch):
    import synth

    flat = {k: (0.05, 0.05, 0.05, 0.05) for k in synth.PATTERN}
    monkeypatch.setattr(synth, "PATTERN", flat)
    sessions = [synth.synth_session(pos, "still", seed=i) for i, pos in enumerate(("arm_down", "elbow90_side"))]
    result = run_analysis(sessions)
    assert not result.accuracy_passed and not result.passed
    assert "FAIL" in result.report


def test_without_imu_only_the_accuracy_half_is_decided():
    # The Uno rig has no IMUs: S3a can pass, S3b must say it was not measured.
    sessions = []
    for i, (pos, cond) in enumerate((p, c) for p in ("arm_down", "elbow90_side") for c in CONDITIONS):
        s = synth_session(pos, cond, seed=i)
        s.imu_t_us, s.imu_q = s.imu_t_us[:0], s.imu_q[:0]
        sessions.append(s)
    result = run_analysis(sessions)
    assert result.accuracy_passed
    assert result.gating_passed is None and not result.passed
    assert "NOT MEASURED" in result.report and "build the ESP32-C6" in result.report
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_evaluate.py tests/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.evaluate'`

- [ ] **Step 3: Implement**

`emg/emgpoc/evaluate.py`:

```python
"""Cross-validation and the S3 metrics.

Windows overlap (200 ms windows every 50 ms), so windows of one hold are near-copies of
each other. Splitting windows at random leaks the test hold into training and inflates
accuracy. Every split here holds out whole groups (one rep in one position/condition).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ModelFactory = Callable[[], object]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[index[t], index[p]] += 1
    return m


def leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make_model: ModelFactory) -> np.ndarray:
    pred = np.empty(len(y), dtype=y.dtype)
    for train, test in leave_one_group_out(groups):
        pred[test] = make_model().fit(X[train], y[train]).predict(X[test])
    return pred


def cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make_model: ModelFactory) -> float:
    return accuracy(y, cv_predict(X, y, groups, make_model))


def backward_elimination(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make_model: ModelFactory,
                         keep: tuple[str, ...] = ("rest",), min_classes: int = 2
                         ) -> list[tuple[tuple[str, ...], float]]:
    """Repeatedly drop the class whose removal raises CV accuracy most. Returns the whole path."""
    classes = sorted(set(y.tolist()))
    path = [(tuple(classes), cv_accuracy(X, y, groups, make_model))]
    while len(classes) > min_classes:
        best: tuple[float, str] | None = None
        for c in classes:
            if c in keep:
                continue
            mask = np.isin(y, [k for k in classes if k != c])
            acc = cv_accuracy(X[mask], y[mask], groups[mask], make_model)
            if best is None or acc > best[0]:
                best = (acc, c)
        if best is None:
            break
        classes.remove(best[1])
        path.append((tuple(classes), best[0]))
    return path


def cross_position_matrix(X: np.ndarray, y: np.ndarray, position: np.ndarray, groups: np.ndarray,
                          make_model: ModelFactory) -> tuple[list[str], np.ndarray]:
    """acc[i, j] = train on position i, test on position j. Diagonal = held-out reps, same position."""
    positions = sorted(set(position.tolist()))
    acc = np.zeros((len(positions), len(positions)))
    for i, a in enumerate(positions):
        in_a = position == a
        acc[i, i] = cv_accuracy(X[in_a], y[in_a], groups[in_a], make_model)
        model = make_model().fit(X[in_a], y[in_a])
        for j, b in enumerate(positions):
            if i != j:
                in_b = position == b
                acc[i, j] = accuracy(y[in_b], model.predict(X[in_b]))
    return positions, acc


def false_trigger_rate(y_true: np.ndarray, y_pred: np.ndarray, commands: tuple[str, ...]) -> float:
    """Share of no-command windows (rest, positioning, wrist moves) that came out as a grip command."""
    idle = ~np.isin(y_true, commands)
    return float(np.mean(np.isin(y_pred[idle], commands))) if idle.any() else float("nan")


def command_recall(y_true: np.ndarray, y_pred: np.ndarray, commands: tuple[str, ...]) -> float:
    cmd = np.isin(y_true, commands)
    return float(np.mean(y_pred[cmd] == y_true[cmd])) if cmd.any() else float("nan")
```

`emg/emgpoc/report.py`:

```python
"""S3: from recorded sessions to a markdown report, a gate verdict and a deployable LDA model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import build_dataset, to_product_labels
from .evaluate import (
    accuracy,
    backward_elimination,
    command_recall,
    confusion_matrix,
    cross_position_matrix,
    cv_accuracy,
    cv_predict,
    false_trigger_rate,
)
from .gating import gate
from .mlp import MLP
from .models import StandardizedModel, make_lda
from .sessionio import Session

GATE_CLASSES = 4  # rest + 3 grip commands (spec §6, S3)
GATE_ACCURACY = 0.90
GATE_FALSE_TRIGGER = 0.02  # "near zero"
SPEED_THRESHOLDS = (10.0, 20.0, 30.0, 60.0)  # deg/s


@dataclass
class S3Result:
    report: str
    chosen: tuple[str, ...]
    model: StandardizedModel
    thr: np.ndarray
    accuracy_passed: bool  # S3a: measurable on the Uno rig
    gating_passed: bool | None  # S3b: None = no IMU data yet (ESP32-C6 rig)
    passed: bool  # both


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def run_analysis(sessions: list[Session], n_classes: int = GATE_CLASSES) -> S3Result:
    ds = build_dataset(sessions)
    lines = ["# S3 report", "", f"Sessions: {len(sessions)}, labelled windows: {len(ds.y)}", ""]

    still = ds.subset(ds.condition == "still")
    classes = sorted(set(still.y.tolist()))
    pred = cv_predict(still.X, still.y, still.group, make_lda())
    lines += [f"## All {len(classes)} still classes (LDA, one rep held out)", "",
              f"Accuracy: **{accuracy(still.y, pred):.3f}**", ""]
    cm = confusion_matrix(still.y, pred, classes)
    lines += _table(["true \\ predicted", *classes], [[c, *map(str, row)] for c, row in zip(classes, cm)]) + [""]

    y_prod = to_product_labels(still.y)
    path = backward_elimination(still.X, y_prod, still.group, make_lda())
    lines += ["## Grip commands: backward elimination (rest always kept)", ""]
    lines += _table(["classes", "accuracy", "set"], [[str(len(c)), f"{a:.3f}", ", ".join(c)] for c, a in path]) + [""]
    chosen = next((c for c, _ in path if len(c) == n_classes), path[-1][0])

    m = np.isin(y_prod, chosen)
    positions, acc = cross_position_matrix(still.X[m], y_prod[m], still.position[m], still.group[m], make_lda())
    within = float(np.min(np.diag(acc)))
    lines += [f"## Chosen set: {', '.join(chosen)}", "", "Train on row, test on column (diagonal: held-out reps):", ""]
    lines += _table(["train \\ test", *positions], [[p, *(f"{v:.3f}" for v in row)] for p, row in zip(positions, acc)])
    mlp_acc = cv_accuracy(still.X[m], y_prod[m], still.group[m], lambda: StandardizedModel(MLP()))
    lines += ["", f"Worst within-position accuracy: **{within:.3f}**", f"MLP, all positions pooled: {mlp_acc:.3f}", ""]

    y_all = to_product_labels(ds.y)
    keep = np.isin(y_all, chosen)
    d2, y2 = ds.subset(keep), y_all[keep]
    moving = d2.condition == "moving"
    commands = tuple(c for c in chosen if c != "rest")
    best_ftr = float("nan")
    if moving.any():
        pred2 = cv_predict(d2.X, y2, d2.group, make_lda())[moving]
        truth, speed = d2.y[moving], np.nan_to_num(d2.speed[moving], nan=0.0)
        rows, ftrs = [], []
        for thr in (None, *SPEED_THRESHOLDS):
            p = pred2 if thr is None else gate(pred2, speed, thr)
            ftr, rec = false_trigger_rate(truth, p, commands), command_recall(truth, p, commands)
            if thr is not None:
                ftrs.append(ftr)
            rows.append(["off" if thr is None else f"{thr:.0f}", f"{ftr:.3f}", f"{rec:.3f}"])
        best_ftr = min(ftrs)
        lines += ["## Moving condition: false grip triggers", ""]
        lines += _table(["gate (deg/s)", "false-trigger rate", "command recall"], rows) + [""]
        if np.all(np.isnan(d2.speed[moving])):
            lines += ["No IMU data in moving sessions: gating had nothing to act on.", ""]
    else:
        lines += ["## Moving condition", "", "No moving sessions recorded.", ""]

    acc_ok = bool(within >= GATE_ACCURACY and len(chosen) >= n_classes)
    has_imu = bool(moving.any() and not np.all(np.isnan(d2.speed[moving])))
    gating_ok = bool(best_ftr <= GATE_FALSE_TRIGGER) if has_imu else None
    passed = acc_ok and gating_ok is True
    if passed:
        verdict = "PASS: active hand is viable"
    elif acc_ok and gating_ok is None:
        verdict = "S3a PASS: build the ESP32-C6 + IMU rig to measure S3b"
    else:
        verdict = "FAIL: see spec §8 responses"
    s3b = "NOT MEASURED (no IMU data)" if gating_ok is None else ("PASS" if gating_ok else "FAIL")
    lines += ["## Gate S3", "",
              f"- S3a: {n_classes} classes at >= {GATE_ACCURACY:.0%} in every position: {'PASS' if acc_ok else 'FAIL'}",
              f"- S3b: false triggers <= {GATE_FALSE_TRIGGER:.0%} with IMU gating: {s3b}",
              f"- **{verdict}**", ""]

    model = make_lda()().fit(d2.X, y2)
    return S3Result("\n".join(lines), chosen, model, ds.thr, acc_ok, gating_ok, passed)
```

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_evaluate.py tests/test_report.py -q`
Expected: `9 passed`

- [ ] **Step 5: Prove the leakage test can fail**

Temporarily change the last line of `test_cross_validation_does_not_leak_windows_of_the_same_hold` to use a random window split:

```python
    assert cv_accuracy(X, y, np.random.default_rng(1).integers(0, 10, 800).astype(str), make_lda()) < 0.8
```

Run: `uv run pytest tests/test_evaluate.py -q`
Expected: `1 failed, 5 passed` (a random split scores 1.0). Restore the original line (`groups`) and re-run: `6 passed`.

- [ ] **Step 6: Write the analysis script**

`emg/scripts/analyze.py`:

```python
"""S3: analyse all right-arm sessions, write the report, save the chosen model."""

import argparse
import json
from pathlib import Path

from emgpoc.dataset import HOP, WIN
from emgpoc.models import save_lda
from emgpoc.report import run_analysis
from emgpoc.sessionio import load_session

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", type=Path, default=Path("data/sessions"))
ap.add_argument("--arm", default="right")
ap.add_argument("--report", type=Path, default=Path("data/reports/s3_report.md"))
ap.add_argument("--model", type=Path, default=Path("data/models/lda_s3.npz"))
args = ap.parse_args()

sessions = [s for s in map(load_session, sorted(args.sessions.glob("*.npz"))) if s.meta["arm"] == args.arm]
if not sessions:
    raise SystemExit(f"no {args.arm}-arm sessions in {args.sessions}")
result = run_analysis(sessions)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(result.report)
save_lda(args.model, result.model, result.thr, WIN, HOP)
print(result.report)
print(f"\nReport: {args.report}\nModel:  {args.model} (classes {json.dumps(list(result.chosen))})")
```

- [ ] **Step 7: Run the analysis on the recorded data**

Run: `uv run python scripts/analyze.py`
Expected: the report prints and is saved to `data/reports/s3_report.md`; the model goes to `data/models/lda_s3.npz`. The last section reads `## Gate S3`: S3a PASS or FAIL, and S3b `NOT MEASURED (no IMU data)` on this rig.

- [ ] **Step 8: Decide**

- **S3a PASS** (the report says `build the ESP32-C6 + IMU rig`): continue with Task 15, then buy the Task 16 parts.
- **S3a FAIL:** don't buy anything yet. Apply spec §8: fewer classes, daily practice with the trainer (Task 15) and re-record, or 8 channels. Write the decision at the end of the report.

- [ ] **Step 9: Commit**

```bash
git add emg/emgpoc/evaluate.py emg/emgpoc/report.py emg/scripts/analyze.py emg/tests/test_evaluate.py \
        emg/tests/test_report.py emg/data/reports emg/data/models
git commit -m "Add leakage-safe evaluation and S3 report; record S3 result"
```

---

### Task 15: Live features and the biofeedback trainer

**Files:**
- Create: `emg/emgpoc/streaming.py`, `emg/emgpoc/practice.py`, `emg/scripts/trainer.py`
- Test: `emg/tests/test_streaming.py`, `emg/tests/test_practice.py`

**Interfaces:**
- Consumes: `dataset.WIN/HOP`, `features.window_features`, `filters`, `models.load_lda`.
- Produces: `StreamingFeatures(thr, win=WIN, hop=HOP, n_channels=4, fs=2000.0).push(raw (n, 4)) -> (k, 16)`; `Trial(day, target, hit, latency_s)`; `majority(recent) -> str`; `HoldDetector(hold_n=10).update(decision, target) -> bool`; `PracticeLog(path)` with `.append`, `.trials`, `.daily_hit_rate() -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

`emg/tests/test_streaming.py`:

```python
import numpy as np

from emgpoc.features import feature_matrix
from emgpoc.filters import FilterChain, default_stages
from emgpoc.protocol import ADC_VOLTS_PER_COUNT
from emgpoc.streaming import StreamingFeatures


def test_streaming_matches_offline_for_any_chunking():
    rng = np.random.default_rng(0)
    raw = rng.integers(1800, 2300, size=(3000, 4))
    thr = np.full(4, 0.01)
    filtered = FilterChain(default_stages(2000.0), 4).process(raw * ADC_VOLTS_PER_COUNT)
    offline, _ = feature_matrix(filtered, win=400, hop=100, thr=thr)

    live = StreamingFeatures(thr, win=400, hop=100)
    cuts = np.sort(rng.choice(np.arange(1, 3000), size=40, replace=False))
    rows = [live.push(chunk) for chunk in np.split(raw, cuts)]
    online = np.vstack(rows)

    assert online.shape == offline.shape == (27, 16)
    np.testing.assert_allclose(online, offline, rtol=1e-9, atol=1e-12)


def test_no_rows_until_first_window_is_full():
    live = StreamingFeatures(np.zeros(4), win=400, hop=100)
    assert live.push(np.full((399, 4), 2048)).shape == (0, 16)
    assert live.push(np.full((1, 4), 2048)).shape == (1, 16)
```

`emg/tests/test_practice.py`:

```python
from emgpoc.practice import HoldDetector, PracticeLog, Trial, majority


def test_majority_vote_and_tie_break():
    assert majority(["rest", "fist", "fist"]) == "fist"
    assert majority(["fist", "rest"]) == "rest"


def test_hold_detector_needs_consecutive_matches():
    d = HoldDetector(hold_n=3)
    assert [d.update(x, "fist") for x in ["fist", "fist", "rest", "fist", "fist", "fist"]] == [
        False, False, False, False, False, True,
    ]


def test_log_roundtrip_and_daily_learning_curve(tmp_path):
    log = PracticeLog(tmp_path / "practice.csv")
    log.append(Trial("2026-10-01", "fist", True, 1.2))
    log.append(Trial("2026-10-01", "pinch", False, 5.0))
    log.append(Trial("2026-10-02", "fist", True, 0.9))
    assert log.trials()[1] == Trial("2026-10-01", "pinch", False, 5.0)
    assert log.daily_hit_rate() == {"2026-10-01": 0.5, "2026-10-02": 1.0}


def test_empty_log_has_no_curve(tmp_path):
    assert PracticeLog(tmp_path / "none.csv").daily_hit_rate() == {}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_streaming.py tests/test_practice.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.streaming'`

- [ ] **Step 3: Implement**

`emg/emgpoc/streaming.py`:

```python
"""Real-time feature extraction that matches the offline pipeline sample for sample."""

from __future__ import annotations

import numpy as np

from .dataset import HOP, WIN
from .features import window_features
from .filters import FilterChain, default_stages
from .protocol import ADC_VOLTS_PER_COUNT


class StreamingFeatures:
    """Push raw ADC chunks of any size; get one feature row per completed window.

    Windows end at samples win-1, win-1+hop, ... exactly like features.feature_matrix,
    so a model trained offline sees the same numbers live (no train/serve skew).
    """

    def __init__(self, thr: np.ndarray, win: int = WIN, hop: int = HOP, n_channels: int = 4,
                 fs: float = 2000.0) -> None:
        self.thr, self.win, self.hop = thr, win, hop
        self.chain = FilterChain(default_stages(fs), n_channels)
        self._buf = np.empty((0, n_channels))
        self._buf_start = 0  # absolute index of self._buf[0]
        self._n = 0  # samples received so far
        self._next_end = win  # a window completes when _n reaches this

    def push(self, raw: np.ndarray) -> np.ndarray:
        y = self.chain.process(np.asarray(raw, dtype=np.float64) * ADC_VOLTS_PER_COUNT)
        self._buf = np.vstack([self._buf, y])
        self._n += len(y)
        rows = []
        while self._n >= self._next_end:
            s = self._next_end - self.win - self._buf_start
            rows.append(window_features(self._buf[s : s + self.win], self.thr))
            self._next_end += self.hop
        drop = (self._next_end - self.win) - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop
        return np.array(rows).reshape(len(rows), 4 * self._buf.shape[1])
```

`emg/emgpoc/practice.py`:

```python
"""Biofeedback practice: decision smoothing, target detection, and a daily learning curve."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Trial:
    day: str  # ISO date
    target: str
    hit: bool
    latency_s: float  # time to reach the target, or the timeout on a miss


def majority(recent: list[str]) -> str:
    """Majority vote over the last few decisions; ties go to the most recent label."""
    counts = Counter(recent)
    best = max(counts.values())
    return next(label for label in reversed(recent) if counts[label] == best)


class HoldDetector:
    """A target counts as reached after hold_n consecutive matching decisions (10 x 50 ms = 0.5 s)."""

    def __init__(self, hold_n: int = 10) -> None:
        self.hold_n = hold_n
        self._run = 0

    def update(self, decision: str, target: str) -> bool:
        self._run = self._run + 1 if decision == target else 0
        return self._run >= self.hold_n


class PracticeLog:
    FIELDS = ("day", "target", "hit", "latency_s")

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trial: Trial) -> None:
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.FIELDS)
            w.writerow([trial.day, trial.target, int(trial.hit), f"{trial.latency_s:.3f}"])

    def trials(self) -> list[Trial]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as f:
            return [Trial(r["day"], r["target"], r["hit"] == "1", float(r["latency_s"])) for r in csv.DictReader(f)]

    def daily_hit_rate(self) -> dict[str, float]:
        per_day: dict[str, list[bool]] = {}
        for t in self.trials():
            per_day.setdefault(t.day, []).append(t.hit)
        return {day: sum(h) / len(h) for day, h in sorted(per_day.items())}
```

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_streaming.py tests/test_practice.py -q`
Expected: `6 passed`

- [ ] **Step 5: Write the trainer**

`emg/scripts/trainer.py`:

```python
"""S4: biofeedback practice. Hold the shown gesture for 0.5 s; hits are logged per day."""

import argparse
import random
import time
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np

from emgpoc.display import bar
from emgpoc.models import load_lda
from emgpoc.practice import HoldDetector, PracticeLog, Trial, majority
from emgpoc.protocol import ClassFrame, EmgFrame, FrameParser
from emgpoc.serialport import open_stream
from emgpoc.streaming import StreamingFeatures

ap = argparse.ArgumentParser()
ap.add_argument("--port")
ap.add_argument("--model", type=Path, default=Path("data/models/lda_s3.npz"))
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--timeout", type=float, default=5.0)
ap.add_argument("--log", type=Path, default=Path("data/practice.csv"))
ap.add_argument("--onchip", action="store_true", help="use the firmware's class frames instead of the laptop")
args = ap.parse_args()

model, thr, win, hop = load_lda(args.model)
classes = [str(c) for c in model.classes_]
targets = [c for c in classes if c != "rest"]
ser, parser, live = open_stream(args.port), FrameParser(), StreamingFeatures(thr, win, hop)
recent: deque[str] = deque(maxlen=3)


def next_decisions() -> list[str]:
    frames = parser.feed(ser.read(4096))
    if args.onchip:
        return [classes[f.class_id] for f in frames if isinstance(f, ClassFrame)]
    raw = [f.ch for f in frames if isinstance(f, EmgFrame)]
    rows = live.push(np.array(raw).reshape(-1, 4)) if raw else np.empty((0, 4 * len(thr)))
    return list(model.predict(rows)) if len(rows) else []


def run_for(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        recent.extend(next_decisions())


log, today = PracticeLog(args.log), date.today().isoformat()
for n in range(1, args.trials + 1):
    target = random.choice(targets)
    print(f"\nTrial {n}/{args.trials}: relax...")
    run_for(2.0)
    print(f"  TARGET: {target.upper()}")
    detector, start, hit = HoldDetector(), time.monotonic(), False
    while (elapsed := time.monotonic() - start) < args.timeout and not hit:
        for d in next_decisions():
            recent.append(d)
            hit = detector.update(majority(list(recent)), target) or hit
        now = majority(list(recent)) if recent else "-"
        print(f"\r  now: {now:<20} {bar(elapsed, args.timeout, 20)}", end="", flush=True)
    log.append(Trial(today, target, hit, elapsed))
    print(f"\r  {'HIT ' if hit else 'miss'} in {elapsed:.1f}s{' ' * 40}")

print("\nDaily hit rate (learning curve):")
for day, rate in log.daily_hit_rate().items():
    print(f"  {day}  {bar(rate, 1.0, 20)} {rate:.0%}")
```

- [ ] **Step 6: First practice session**

Charger unplugged, band on the right arm, 10 min settle. Run: `uv run python scripts/trainer.py`
Expected: 20 trials, each ending `HIT` or `miss`, then a line for today in the learning curve.

- [ ] **Step 7: Commit**

```bash
git add emg/emgpoc/streaming.py emg/emgpoc/practice.py emg/scripts/trainer.py emg/tests/test_streaming.py \
        emg/tests/test_practice.py emg/data/practice.csv
git commit -m "Add live streaming features and biofeedback trainer"
```

---

### Task 16: ESP32-C6 rig with IMUs (gate S3b) — only if S3a passed

**Files:**
- Modify: `emg/firmware/Cargo.toml` (add `sensors`)
- Create: `emg/firmware/sensors/Cargo.toml`, `emg/firmware/sensors/src/lib.rs`, `emg/firmware/logger/Cargo.toml`, `emg/firmware/logger/rust-toolchain.toml`, `emg/firmware/logger/.cargo/config.toml`, `emg/firmware/logger/src/main.rs`

**Interfaces:**
- Consumes: `frame` (Task 4); every Python tool unchanged (same frames, same scale; `open_stream` already knows the ESP32-C6's USB ID).
- Produces: `sensors::mcp3208::{command(channel: u8) -> [u8; 3], parse(rx: [u8; 3]) -> u16}`; `sensors::bno055::{ADDR_BAND, ADDR_STUB, REG_CHIP_ID, CHIP_ID, REG_OPR_MODE, MODE_NDOF, REG_QUA_DATA_W_LSB, quaternion(raw: [u8; 8]) -> [i16; 4]}`; the ESP32-C6 logger streaming EMG (2 kHz, 12-bit) and IMU frames (100 Hz).

- [ ] **Step 1: Buy the parts**

| Part | Qty | Note |
|---|---|---|
| Seeed XIAO ESP32C6 | 1 | native USB, RISC-V |
| MCP3208-CI/P (DIP-16) | 1 | 12-bit, 8-ch SPI ADC |
| 100 nF + 1 µF ceramic capacitors | 2 each | decoupling at the MCP3208 |
| BNO055 breakout | 2 | one with ADR tied high (address 0x29) |
| Breadboard | 1 | |

- [ ] **Step 2: Wire it (the band and MyoWare boards move over from the Uno)**

| From | To |
|---|---|
| XIAO 3V3 | MCP3208 VDD (16) and VREF (15); every MyoWare `+`; both BNO055 VIN |
| XIAO GND | MCP3208 AGND (14) and DGND (9); every MyoWare `−`; both BNO055 GND |
| XIAO D8 / GPIO19 | MCP3208 CLK (13) |
| XIAO D9 / GPIO20 | MCP3208 DOUT (12) |
| XIAO D10 / GPIO18 | MCP3208 DIN (11) |
| XIAO D3 / GPIO21 | MCP3208 CS/SHDN (10) |
| MyoWare 1–4 `RAW` | MCP3208 CH0–CH3 (1–4) |
| XIAO D4 / GPIO22, D5 / GPIO23 | both BNO055 SDA, SCL |
| BNO055 #2 ADR | 3V3 (address 0x29; this one goes on the stub) |

Put the 100 nF and 1 µF across VDD–AGND right at the MCP3208 pins.

- [ ] **Step 3: Add `sensors` to the workspace**

Replace `emg/firmware/Cargo.toml`:

```toml
# Host-testable no_std building blocks. The ESP32-C6 binary lives in logger/ (own target config).
[workspace]
resolver = "2"
members = ["frame", "sensors"]
exclude = ["logger", "uno-logger"]

[workspace.package]
edition = "2021"
version = "0.1.0"
```

`emg/firmware/sensors/Cargo.toml`:

```toml
[package]
name = "sensors"
edition.workspace = true
version.workspace = true

[dependencies]
```

- [ ] **Step 4: Write the failing test: `emg/firmware/sensors/src/lib.rs` containing only its test module**

```rust
#![no_std]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp3208_commands() {
        assert_eq!(mcp3208::command(0), [0x06, 0x00, 0x00]);
        assert_eq!(mcp3208::command(3), [0x06, 0xC0, 0x00]);
        assert_eq!(mcp3208::command(4), [0x07, 0x00, 0x00]);
        assert_eq!(mcp3208::command(7), [0x07, 0xC0, 0x00]);
    }

    #[test]
    fn mcp3208_parse_ignores_the_leading_bits() {
        assert_eq!(mcp3208::parse([0xFF, 0xFA, 0xBC]), 0xABC);
        assert_eq!(mcp3208::parse([0x00, 0x00, 0x00]), 0);
        assert_eq!(mcp3208::parse([0x00, 0x0F, 0xFF]), 4095);
    }

    #[test]
    #[should_panic]
    fn mcp3208_rejects_channel_8() {
        mcp3208::command(8);
    }

    #[test]
    fn bno055_quaternion_is_four_le_i16() {
        let raw = [0x00, 0x40, 0x00, 0x00, 0x00, 0xC0, 0x01, 0x00];
        assert_eq!(bno055::quaternion(raw), [16384, 0, -16384, 1]);
    }
}
```

- [ ] **Step 5: Run to verify it fails**

Run: `cd firmware && cargo test -q -p sensors`
Expected: compile errors — `failed to resolve: use of undeclared crate or module \`mcp3208\``

- [ ] **Step 6: Implement: replace `emg/firmware/sensors/src/lib.rs` completely**

```rust
//! Pure byte-level logic for the two chips on the band, so it can be tested on the laptop.
#![no_std]

/// MCP3208: 8-channel 12-bit SAR ADC on SPI (mode 0, <= 1 MHz at 3.3 V).
pub mod mcp3208 {
    /// 3-byte command for a single-ended read: start bit, SGL=1, channel D2 D1 D0.
    pub fn command(channel: u8) -> [u8; 3] {
        assert!(channel < 8, "MCP3208 has channels 0..=7");
        [0x06 | (channel >> 2), (channel & 0x03) << 6, 0x00]
    }

    /// The 12-bit result sits in the low nibble of byte 1 and all of byte 2.
    pub fn parse(rx: [u8; 3]) -> u16 {
        (u16::from(rx[1] & 0x0F) << 8) | u16::from(rx[2])
    }
}

/// BNO055: 9-axis IMU with on-chip fusion, I2C at 0x28 (ADR low) or 0x29 (ADR high).
pub mod bno055 {
    pub const ADDR_BAND: u8 = 0x28;
    pub const ADDR_STUB: u8 = 0x29;
    pub const REG_CHIP_ID: u8 = 0x00;
    pub const CHIP_ID: u8 = 0xA0;
    pub const REG_OPR_MODE: u8 = 0x3D;
    pub const MODE_NDOF: u8 = 0x0C; // full fusion: accelerometer + gyro + magnetometer
    pub const REG_QUA_DATA_W_LSB: u8 = 0x20; // 8 bytes: w, x, y, z as i16 LE, 1 LSB = 2^-14

    pub fn quaternion(raw: [u8; 8]) -> [i16; 4] {
        core::array::from_fn(|i| i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp3208_commands() {
        assert_eq!(mcp3208::command(0), [0x06, 0x00, 0x00]);
        assert_eq!(mcp3208::command(3), [0x06, 0xC0, 0x00]);
        assert_eq!(mcp3208::command(4), [0x07, 0x00, 0x00]);
        assert_eq!(mcp3208::command(7), [0x07, 0xC0, 0x00]);
    }

    #[test]
    fn mcp3208_parse_ignores_the_leading_bits() {
        assert_eq!(mcp3208::parse([0xFF, 0xFA, 0xBC]), 0xABC);
        assert_eq!(mcp3208::parse([0x00, 0x00, 0x00]), 0);
        assert_eq!(mcp3208::parse([0x00, 0x0F, 0xFF]), 4095);
    }

    #[test]
    #[should_panic]
    fn mcp3208_rejects_channel_8() {
        mcp3208::command(8);
    }

    #[test]
    fn bno055_quaternion_is_four_le_i16() {
        let raw = [0x00, 0x40, 0x00, 0x00, 0x00, 0xC0, 0x01, 0x00];
        assert_eq!(bno055::quaternion(raw), [16384, 0, -16384, 1]);
    }
}
```

- [ ] **Step 7: Run to verify it passes**

Run: `cargo test -q -p sensors`
Expected: `test result: ok. 4 passed`

- [ ] **Step 8: Write the ESP32-C6 logger**

`emg/firmware/logger/rust-toolchain.toml` (esp-hal 1.2 needs Rust ≥ 1.95; pinned for this folder only):

```toml
[toolchain]
channel = "1.95"
components = ["rust-src"]
targets = ["riscv32imac-unknown-none-elf"]
```

`emg/firmware/logger/.cargo/config.toml`:

```toml
[build]
target = "riscv32imac-unknown-none-elf"

[target.riscv32imac-unknown-none-elf]
runner = "espflash flash"
rustflags = ["-C", "link-arg=-Tlinkall.x"]
```

`emg/firmware/logger/Cargo.toml`:

```toml
[package]
name = "emg-logger"
version = "0.1.0"
edition = "2021"

[dependencies]
esp-hal = { version = "1.2", features = ["esp32c6", "unstable"] }
esp-bootloader-esp-idf = { version = "0.6", features = ["esp32c6"] }
esp-backtrace = { version = "0.20", features = ["esp32c6", "panic-handler", "println"] }
esp-println = { version = "0.18", default-features = false, features = ["esp32c6", "uart"] }
critical-section = "1.2"
heapless = "0.9"
frame = { path = "../frame" }
sensors = { path = "../sensors" }

[profile.dev]
opt-level = "s"

[profile.release]
opt-level = 3
debug = true
```

`emg/firmware/logger/src/main.rs`:

```rust
//! Track S logger for ESP32-C6: 4 EMG channels at 2 kHz (MCP3208 on SPI) + 2 BNO055
//! quaternions at 100 Hz (I2C), streamed as binary frames over the native USB serial.
//! Debug text goes to UART0 (GPIO16), never to USB: USB carries only frames.
#![no_std]
#![no_main]

use core::cell::RefCell;

use critical_section::Mutex;
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    handler,
    i2c::master::{Config as I2cConfig, I2c},
    main,
    spi::{
        master::{Config as SpiConfig, Spi},
        Mode,
    },
    time::{Duration, Instant, Rate},
    timer::{timg::TimerGroup, PeriodicTimer},
    usb::usb_serial_jtag::UsbSerialJtag,
    Blocking,
};
use heapless::Deque;
use sensors::{bno055, mcp3208};

esp_bootloader_esp_idf::esp_app_desc!();

const SAMPLE_PERIOD: Duration = Duration::from_micros(500); // 2 kHz per channel
const IMU_PERIOD_US: u64 = 10_000; // 100 Hz
const QUEUE_LEN: usize = 1024; // ~0.5 s of slack for the main loop

struct Sample {
    t_us: u32,
    seq: u16,
    ch: [u16; frame::EMG_CHANNELS],
}

struct Sampler {
    spi: Spi<'static, Blocking>,
    timer: PeriodicTimer<'static, Blocking>,
    seq: u16,
}

static SAMPLER: Mutex<RefCell<Option<Sampler>>> = Mutex::new(RefCell::new(None));
static QUEUE: Mutex<RefCell<Deque<Sample, QUEUE_LEN>>> = Mutex::new(RefCell::new(Deque::new()));

fn now_us() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

#[handler]
fn on_sample_tick() {
    critical_section::with(|cs| {
        let mut sampler = SAMPLER.borrow_ref_mut(cs);
        let Some(s) = sampler.as_mut() else { return };
        s.timer.clear_interrupt();
        let seq = s.seq;
        s.seq = seq.wrapping_add(1); // a dropped sample still uses up a seq: the host sees the gap
        let t_us = now_us() as u32;
        let mut ch = [0u16; frame::EMG_CHANNELS];
        for (i, v) in ch.iter_mut().enumerate() {
            let mut buf = mcp3208::command(i as u8);
            if s.spi.transfer(&mut buf).is_ok() {
                *v = mcp3208::parse(buf);
            }
        }
        // Queue full means the main loop fell behind: drop the sample.
        let _ = QUEUE.borrow_ref_mut(cs).push_back(Sample { t_us, seq, ch });
    });
}

fn bno055_start(i2c: &mut I2c<'static, Blocking>, addr: u8) -> bool {
    let mut id = [0u8];
    if i2c.write_read(addr, &[bno055::REG_CHIP_ID], &mut id).is_err() || id[0] != bno055::CHIP_ID {
        return false;
    }
    i2c.write(addr, &[bno055::REG_OPR_MODE, bno055::MODE_NDOF]).is_ok()
}

fn bno055_quat(i2c: &mut I2c<'static, Blocking>, addr: u8) -> [i16; 4] {
    let mut raw = [0u8; 8];
    match i2c.write_read(addr, &[bno055::REG_QUA_DATA_W_LSB], &mut raw) {
        Ok(()) => bno055::quaternion(raw),
        Err(_) => [0; 4], // all-zero quaternion = "no reading"; the host treats it as invalid
    }
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    let delay = Delay::new();
    let mut usb = UsbSerialJtag::new(p.USB_DEVICE);

    // MCP3208 at 1 MHz (its limit at 3.3 V is ~1.2 MHz). CS toggles per transfer = per conversion.
    let spi = Spi::new(p.SPI2, SpiConfig::default().with_frequency(Rate::from_khz(1000)).with_mode(Mode::_0))
        .unwrap()
        .with_sck(p.GPIO19)
        .with_miso(p.GPIO20)
        .with_mosi(p.GPIO18)
        .with_cs(p.GPIO21);

    // BNO055 stretches the I2C clock; 100 kHz keeps that well inside the default timeout.
    let mut i2c = I2c::new(p.I2C0, I2cConfig::default().with_frequency(Rate::from_khz(100)))
        .unwrap()
        .with_sda(p.GPIO22)
        .with_scl(p.GPIO23);
    delay.delay_millis(700); // BNO055 needs ~650 ms after power-up
    let imu_ok = bno055_start(&mut i2c, bno055::ADDR_BAND) && bno055_start(&mut i2c, bno055::ADDR_STUB);
    esp_println::println!("emg-logger: imu_ok={}", imu_ok);

    let timg0 = TimerGroup::new(p.TIMG0);
    let mut timer = PeriodicTimer::new(timg0.timer0);
    timer.set_interrupt_handler(on_sample_tick);
    timer.listen();
    // Start only once the ISR can find its state, or the first ticks would never be cleared.
    critical_section::with(|cs| {
        let mut slot = SAMPLER.borrow_ref_mut(cs);
        let sampler = slot.insert(Sampler { spi, timer, seq: 0 });
        sampler.timer.start(SAMPLE_PERIOD).unwrap();
    });


    let mut imu_seq: u16 = 0;
    let mut next_imu = now_us();
    loop {
        while let Some(s) = critical_section::with(|cs| QUEUE.borrow_ref_mut(cs).pop_front()) {
            let _ = usb.write(&frame::encode_emg(s.seq, s.t_us, &s.ch));
        }
        if imu_ok && now_us() >= next_imu {
            next_imu += IMU_PERIOD_US;
            let band = bno055_quat(&mut i2c, bno055::ADDR_BAND);
            let stub = bno055_quat(&mut i2c, bno055::ADDR_STUB);
            let _ = usb.write(&frame::encode_imu(imu_seq, now_us() as u32, &band, &stub));
            imu_seq = imu_seq.wrapping_add(1);
        }
    }
}
```

- [ ] **Step 9: Build and flash**

```bash
cd firmware/logger
cargo build --release          # Expected: Finished `release` profile
cargo install espflash --locked
cargo run --release            # Expected: espflash detects esp32c6, "Flashing has completed!"
```

No `--monitor`: the USB port carries only binary frames. Debug text goes to UART0 on D6/GPIO16 (any USB-UART adapter, 115200 baud).

- [ ] **Step 10: Check the stream**

Charger unplugged. Run: `uv run python scripts/check_stream.py --seconds 60`
Expected: `EMG 120000 frames at 2000.0 Hz, lost 0`, `IMU ~6000 frames, lost 0`, `bad CRC 0`, channel means near 1.65 V.

- [ ] **Step 11: Re-record the full set on this rig**

Uno and ESP32 sessions differ in resolution, so analyse them separately. Band at the marks, stub BNO055 taped to the stub, 10 min settle, then:

```bash
for pos in elbow90_side arm_down arm_forward; do
  uv run python scripts/record.py --arm right --position $pos --condition still --out data/sessions_esp32
  uv run python scripts/record.py --arm right --position $pos --condition moving --out data/sessions_esp32
done
uv run python scripts/analyze.py --sessions data/sessions_esp32 --report data/reports/s3_esp32.md
```

Expected: the report's gate section now shows a measured S3b line (PASS or FAIL) instead of `NOT MEASURED`. The model saved to `data/models/lda_s3.npz` is the one Task 18 puts on the chip.

- [ ] **Step 12: Commit**

```bash
git add emg/firmware/Cargo.toml emg/firmware/Cargo.lock emg/firmware/sensors emg/firmware/logger \
        emg/data/reports emg/data/models
git commit -m "Add ESP32-C6 logger with MCP3208 and IMUs; record S3b"
```

**Gate S3b:** false triggers ≤ 2 % with IMU gating. If it fails, see spec §8 (lockable wrist in the cap).

---

### Task 17: Rust `dsp` crate, checked against Python

**Files:**
- Modify: `emg/firmware/Cargo.toml` (add `dsp` to the members)
- Create: `emg/firmware/dsp/Cargo.toml`, `emg/firmware/dsp/src/lib.rs`, `emg/firmware/dsp/tests/golden_test.rs`, `emg/emgpoc/export_rs.py`, `emg/scripts/make_golden.py`
- Generated: `emg/firmware/dsp/tests/golden/mod.rs`
- Test: `emg/tests/test_export_rs.py`

**Interfaces:**
- Consumes: `filters`, `features`, `models`, `streaming` (Python reference).
- Produces: `f32_literal`, `f32_array`, `render_model_rs(m, thr, win, hop) -> str`, `render_golden_rs(seed=0) -> str`; Rust `dsp::{Biquad::new([f32; 5]), Biquad::step, window_features(w, n_ch, thr, out), lda_argmax(feat, mean, scale, w, b) -> usize, Model { stages, thr, mean, scale, w, b, volts_per_count, hop }, Classifier::<N_CH, WIN, N_STAGES>::new(&Model), Classifier::push(&Model, &[u16; N_CH]) -> Option<usize>, Classifier::features() -> &[f32]}`.

- [ ] **Step 1: Write the failing Python test**

```python
import numpy as np
import pytest

from emgpoc.export_rs import f32_literal, render_golden_rs, render_model_rs
from emgpoc.models import make_lda


def test_float_literals_are_valid_rust():
    assert f32_literal(171.0) == "171.0"
    assert f32_literal(-0.5) == "-0.5"
    assert f32_literal(1e-20) == "9.99999968e-21"
    assert np.float32(float(f32_literal(0.1))) == np.float32(0.1)


def test_non_finite_values_are_refused():
    with pytest.raises(ValueError):
        f32_literal(float("nan"))


def test_golden_is_deterministic():
    assert render_golden_rs(seed=0) == render_golden_rs(seed=0)


def test_model_source_declares_consistent_sizes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 16)) + np.repeat(rng.normal(0, 3, size=(3, 16)), 20, axis=0)
    y = np.repeat(np.array(["rest", "imagined_fist", "cocontraction"]), 20)
    text = render_model_rs(make_lda()().fit(X, y), thr=np.full(4, 0.01), win=400, hop=100)
    assert "pub const D: usize = 16;" in text and "pub const K: usize = 3;" in text
    w_line = next(line for line in text.splitlines() if line.startswith("pub const W:"))
    assert w_line.count(",") + 1 == 48
    assert '["cocontraction", "imagined_fist", "rest"]' in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_export_rs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.export_rs'`

- [ ] **Step 3: Implement the exporter and the golden script**

`emg/emgpoc/export_rs.py`:

```python
"""Generate Rust sources: the deployable model (firmware/logger/src/model.rs) and DSP golden test data."""

from __future__ import annotations

import numpy as np

from .features import window_features
from .filters import FilterChain, default_stages
from .models import StandardizedModel, make_lda
from .protocol import ADC_VOLTS_PER_COUNT
from .streaming import StreamingFeatures


def f32_literal(v: float) -> str:
    """float32-exact Rust literal: 9 significant digits round-trip a float32 exactly."""
    if not np.isfinite(v):
        raise ValueError(f"cannot export non-finite value {v}")
    text = f"{float(np.float32(v)):.9g}"
    if not any(ch in text for ch in ".e"):  # "171" would be an integer literal
        text += ".0"
    return text


def f32_array(values) -> str:
    return "[" + ", ".join(f32_literal(v) for v in np.ravel(values)) + "]"


def render_model_rs(m: StandardizedModel, thr: np.ndarray, win: int, hop: int, fs: float = 2000.0) -> str:
    stages = default_stages(fs)
    W, b = m.model.W_, m.model.b_
    d, k = W.shape
    stage_rows = ", ".join(f32_array([s.b0, s.b1, s.b2, s.a1, s.a2]) for s in stages)
    classes = ", ".join(f'"{c}"' for c in m.classes_)
    return f"""// Generated by emgpoc.export_rs. Do not edit; re-run scripts/export_model.py.
#![allow(dead_code)]

pub const N_CH: usize = {len(thr)};
pub const WIN: usize = {win};
pub const HOP: usize = {hop};
pub const D: usize = {d};
pub const K: usize = {k};
pub const VOLTS_PER_COUNT: f32 = {f32_literal(ADC_VOLTS_PER_COUNT)};
pub const STAGES: [[f32; 5]; {len(stages)}] = [{stage_rows}];
pub const THR: [f32; N_CH] = {f32_array(thr)};
pub const MEAN: [f32; D] = {f32_array(m.scaler.mean_)};
pub const SCALE: [f32; D] = {f32_array(m.scaler.scale_)};
pub const W: [f32; D * K] = {f32_array(W)};
pub const B: [f32; K] = {f32_array(b)};
pub const CLASSES: [&str; K] = [{classes}];
"""


def render_golden_rs(seed: int = 0) -> str:
    """Inputs plus Python's outputs, for firmware/dsp/tests. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)

    stage = default_stages()[0]
    sig = np.float32(np.sin(np.arange(64) * 0.3) + 0.1 * rng.normal(size=64)).astype(np.float64)
    filt = FilterChain([stage], 1).process(sig[:, None])[:, 0]

    win, n_ch = 400, 4
    window = np.float32(rng.normal(0, 0.1, size=(win, n_ch))).astype(np.float64)
    thr = np.float32(np.full(n_ch, 0.05)).astype(np.float64)
    feats = window_features(window, thr)

    X = rng.normal(size=(300, 16)) + np.repeat(rng.normal(0, 3, size=(3, 16)), 100, axis=0)
    y = np.repeat(np.array(["a", "b", "c"]), 100)
    m = make_lda()().fit(X, y)
    probe = X[[0, 150, 299, 42, 250]]
    expected = [int(np.argmax(m.model.decision_function(m.scaler.transform(p[None]))[0])) for p in probe]
    probe_rows = ", ".join(f32_array(p) for p in probe)

    # A raw stream with 6 distinct gestures, decided by the streaming pipeline + an LDA fit on it.
    amp = np.repeat(rng.choice([4.0, 120.0], size=(6, 4)), 500, axis=0)  # ADC counts
    raw = np.clip(np.round(2048 + rng.normal(size=(3000, 4)) * amp), 0, 4095).astype(np.uint16)
    s_thr = np.float32(np.full(4, 0.01)).astype(np.float64)
    rows = StreamingFeatures(s_thr, win=400, hop=100).push(raw)
    labels = np.array([f"g{i // 5}" for i in range(len(rows))])
    sm = make_lda()().fit(rows, labels)
    decisions = [list(sm.classes_).index(c) for c in sm.predict(rows)]
    raw_rows = ", ".join("[" + ", ".join(str(int(v)) for v in r) + "]" for r in raw)
    stage_rows = ", ".join(f32_array([s.b0, s.b1, s.b2, s.a1, s.a2]) for s in default_stages())

    return f"""// Generated by emgpoc.export_rs.render_golden_rs(seed={seed}). Do not edit.
#![allow(dead_code)]

pub const STAGE: [f32; 5] = {f32_array([stage.b0, stage.b1, stage.b2, stage.a1, stage.a2])};
pub const SIG_IN: [f32; 64] = {f32_array(sig)};
pub const SIG_OUT: [f32; 64] = {f32_array(filt)};

pub const NCH: usize = {n_ch};
pub const WINDOW: [f32; {win * n_ch}] = {f32_array(window)};
pub const THR: [f32; {n_ch}] = {f32_array(thr)};
pub const FEATS: [f32; {4 * n_ch}] = {f32_array(feats)};

pub const MEAN: [f32; 16] = {f32_array(m.scaler.mean_)};
pub const SCALE: [f32; 16] = {f32_array(m.scaler.scale_)};
pub const W: [f32; 48] = {f32_array(m.model.W_)};
pub const B: [f32; 3] = {f32_array(m.model.b_)};
pub const PROBE: [[f32; 16]; 5] = [{probe_rows}];
pub const PROBE_CLASS: [usize; 5] = [{", ".join(map(str, expected))}];

pub const VOLTS_PER_COUNT: f32 = {f32_literal(ADC_VOLTS_PER_COUNT)};
pub const STREAM_WIN: usize = 400;
pub const STREAM_HOP: usize = 100;
pub const STREAM_STAGES: [[f32; 5]; 3] = [{stage_rows}];
pub const STREAM_THR: [f32; 4] = {f32_array(s_thr)};
pub const STREAM_MEAN: [f32; 16] = {f32_array(sm.scaler.mean_)};
pub const STREAM_SCALE: [f32; 16] = {f32_array(sm.scaler.scale_)};
pub const STREAM_W: [f32; {sm.model.W_.size}] = {f32_array(sm.model.W_)};
pub const STREAM_B: [f32; {len(sm.model.b_)}] = {f32_array(sm.model.b_)};
pub const STREAM_RAW: [[u16; 4]; {len(raw)}] = [{raw_rows}];
pub const STREAM_DECISIONS: [usize; {len(decisions)}] = [{", ".join(map(str, decisions))}];
pub const STREAM_FEATURES: [[f32; 16]; {len(rows)}] = [{", ".join(f32_array(r) for r in rows)}];
"""
```

`emg/scripts/make_golden.py`:

```python
"""Regenerate firmware/dsp/tests/golden/mod.rs from the Python reference implementation."""

from pathlib import Path

from emgpoc.export_rs import render_golden_rs

out = Path(__file__).resolve().parents[1] / "firmware" / "dsp" / "tests" / "golden" / "mod.rs"
out.write_text(render_golden_rs())
print(f"wrote {out}")
```

- [ ] **Step 4: Run it to verify it passes, then generate the golden data**

Run: `uv run pytest tests/test_export_rs.py -q` → Expected: `4 passed`
Run: `uv run python scripts/make_golden.py` → Expected: `wrote .../firmware/dsp/tests/golden/mod.rs`

- [ ] **Step 5: Add `dsp` to the workspace and write the failing Rust test**

In `emg/firmware/Cargo.toml` change the members line to:

```toml
members = ["frame", "dsp", "sensors"]
```

(If you skipped Task 16 because S3a failed, you stop before this task anyway: on-chip work only makes sense on the ESP32 rig.)

`emg/firmware/dsp/Cargo.toml`:

```toml
[package]
name = "dsp"
edition.workspace = true
version.workspace = true

[dependencies]
libm = "0.2"
```

`emg/firmware/dsp/tests/golden_test.rs`:

```rust
//! The Rust signal chain must reproduce the Python reference (emgpoc) on the same inputs.
//! golden/mod.rs is generated: `uv run python scripts/make_golden.py`.
mod golden;

use dsp::{lda_argmax, window_features, Biquad};
use golden::*;

#[test]
fn biquad_matches_python() {
    let mut f = Biquad::new(STAGE);
    for (x, want) in SIG_IN.iter().zip(SIG_OUT) {
        let got = f.step(*x);
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
}

#[test]
fn window_features_match_python() {
    let mut out = [0.0f32; 16];
    window_features(&WINDOW, NCH, &THR, &mut out);
    for i in 0..8 {
        let tol = 1e-4 * FEATS[i].abs() + 1e-6;
        assert!((out[i] - FEATS[i]).abs() < tol, "feature {i}: {} vs {}", out[i], FEATS[i]);
    }
    assert_eq!(&out[8..], &FEATS[8..], "ZC/SSC counts must be exact");
}

#[test]
fn lda_matches_python() {
    for (probe, want) in PROBE.iter().zip(PROBE_CLASS) {
        assert_eq!(lda_argmax(probe, &MEAN, &SCALE, &W, &B), want);
    }
}

#[test]
fn streaming_classifier_matches_python_decisions() {
    let model = dsp::Model {
        stages: &STREAM_STAGES,
        thr: &STREAM_THR,
        mean: &STREAM_MEAN,
        scale: &STREAM_SCALE,
        w: &STREAM_W,
        b: &STREAM_B,
        volts_per_count: VOLTS_PER_COUNT,
        hop: STREAM_HOP,
    };
    let mut clf = dsp::Classifier::<4, STREAM_WIN, 3>::new(&model);
    let mut decisions = Vec::new();
    for raw in STREAM_RAW.iter() {
        if let Some(d) = clf.push(&model, raw) {
            let want = &STREAM_FEATURES[decisions.len()];
            for (i, (got, want)) in clf.features().iter().zip(want).enumerate() {
                let tol = if i < 8 { 1e-4 * want.abs() + 1e-6 } else { 0.0 }; // MAV/WL float, ZC/SSC counts
                assert!((got - want).abs() <= tol, "window {} feature {i}: {got} vs {want}", decisions.len());
            }
            decisions.push(d);
        }
    }
    assert_eq!(decisions, STREAM_DECISIONS.to_vec());
}
```

`emg/firmware/dsp/src/lib.rs` (empty for now):

```rust
#![no_std]
```

- [ ] **Step 6: Run to verify it fails**

Run: `cd firmware && cargo test -q -p dsp`
Expected: compile errors — `unresolved imports \`dsp::lda_argmax\`, \`dsp::window_features\`, \`dsp::Biquad\``

- [ ] **Step 7: Implement `emg/firmware/dsp/src/lib.rs`**

```rust
//! On-chip signal chain. Must match emgpoc/filters.py, features.py and lda.py.
#![no_std]

use libm::fabsf;

/// Biquad in transposed direct form II, coefficients normalised so a0 = 1.
#[derive(Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// `c` = [b0, b1, b2, a1, a2], as exported by emgpoc.export_rs.
    pub const fn new(c: [f32; 5]) -> Self {
        Self { b0: c[0], b1: c[1], b2: c[2], a1: c[3], a2: c[4], z1: 0.0, z2: 0.0 }
    }

    pub fn step(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// `w`: row-major window [win][n_ch], oldest sample first. `thr`: [n_ch].
/// `out`: [4 * n_ch] laid out as mav[], wl[], zc[], ssc[].
pub fn window_features(w: &[f32], n_ch: usize, thr: &[f32], out: &mut [f32]) {
    let win = w.len() / n_ch;
    let at = |i: usize, c: usize| w[i * n_ch + c];
    for c in 0..n_ch {
        let (mut mav, mut wl, mut zc, mut ssc) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..win {
            let x = at(i, c);
            mav += fabsf(x);
            if i + 1 < win {
                let nx = at(i + 1, c);
                wl += fabsf(nx - x);
                if x * nx < 0.0 && fabsf(x - nx) >= thr[c] {
                    zc += 1.0;
                }
            }
            if i >= 1 && i + 1 < win {
                let d1 = x - at(i - 1, c);
                let d2 = x - at(i + 1, c);
                if d1 * d2 > 0.0 && (fabsf(d1) >= thr[c] || fabsf(d2) >= thr[c]) {
                    ssc += 1.0;
                }
            }
        }
        out[c] = mav / win as f32;
        out[n_ch + c] = wl;
        out[2 * n_ch + c] = zc;
        out[3 * n_ch + c] = ssc;
    }
}

/// Standardise `feat` with mean/scale, score k = b.len() classes with W (row-major [d][k]), return argmax.
pub fn lda_argmax(feat: &[f32], mean: &[f32], scale: &[f32], w: &[f32], b: &[f32]) -> usize {
    let k = b.len();
    let mut best = (0usize, f32::NEG_INFINITY);
    for j in 0..k {
        let mut s = b[j];
        for i in 0..feat.len() {
            s += (feat[i] - mean[i]) / scale[i] * w[i * k + j];
        }
        if s > best.1 {
            best = (j, s);
        }
    }
    best.0
}

/// Everything the classifier needs, as exported by emgpoc.export_rs (model.rs).
pub struct Model<'a> {
    pub stages: &'a [[f32; 5]],
    pub thr: &'a [f32],
    pub mean: &'a [f32],
    pub scale: &'a [f32],
    pub w: &'a [f32],
    pub b: &'a [f32],
    pub volts_per_count: f32,
    pub hop: usize,
}

const MAX_FEATURES: usize = 64;

/// Streaming filter + window + LDA, one raw sample at a time.
/// Windows end at samples WIN-1, WIN-1+hop, ... exactly like emgpoc.streaming.StreamingFeatures.
pub struct Classifier<const N_CH: usize, const WIN: usize, const N_STAGES: usize> {
    filters: [[Biquad; N_CH]; N_STAGES],
    ring: [[f32; N_CH]; WIN],
    window: [[f32; N_CH]; WIN],
    feats: [f32; MAX_FEATURES],
    n: usize,
}

impl<const N_CH: usize, const WIN: usize, const N_STAGES: usize> Classifier<N_CH, WIN, N_STAGES> {
    pub fn new(model: &Model) -> Self {
        assert_eq!(model.stages.len(), N_STAGES);
        assert!(4 * N_CH <= MAX_FEATURES);
        Self {
            filters: core::array::from_fn(|k| [Biquad::new(model.stages[k]); N_CH]),
            ring: [[0.0; N_CH]; WIN],
            window: [[0.0; N_CH]; WIN],
            feats: [0.0; MAX_FEATURES],
            n: 0,
        }
    }

    /// Feed one raw ADC sample; returns a class index when a window completes.
    pub fn push(&mut self, model: &Model, raw: &[u16; N_CH]) -> Option<usize> {
        let slot = self.n % WIN;
        for c in 0..N_CH {
            let mut y = raw[c] as f32 * model.volts_per_count;
            for stage in self.filters.iter_mut() {
                y = stage[c].step(y);
            }
            self.ring[slot][c] = y;
        }
        self.n += 1;
        if self.n < WIN || (self.n - WIN) % model.hop != 0 {
            return None;
        }
        for i in 0..WIN {
            self.window[i] = self.ring[(self.n + i) % WIN]; // unroll the ring, oldest first
        }
        let feats = &mut self.feats[..4 * N_CH];
        window_features(self.window.as_flattened(), N_CH, model.thr, feats);
        Some(lda_argmax(feats, model.mean, model.scale, model.w, model.b))
    }

    /// Features of the most recently completed window (for checking against Python).
    pub fn features(&self) -> &[f32] {
        &self.feats[..4 * N_CH]
    }
}
```

- [ ] **Step 8: Run to verify it passes**

Run: `cargo test -q -p dsp`
Expected: `test result: ok. 4 passed`

- [ ] **Step 9: Prove the streaming test catches an off-by-one window**

In `Classifier::push`, temporarily change `self.ring[(self.n + i) % WIN]` to `self.ring[(self.n + i + 1) % WIN]`.
Run: `cargo test -q -p dsp` → Expected: `streaming_classifier_matches_python_decisions` FAILS on a feature mismatch.
Restore the line and re-run → `4 passed`.

- [ ] **Step 10: Commit**

```bash
git add emg/firmware/Cargo.toml emg/firmware/Cargo.lock emg/firmware/dsp emg/emgpoc/export_rs.py \
        emg/scripts/make_golden.py emg/tests/test_export_rs.py
git commit -m "Add no_std Rust DSP chain verified against Python golden data"
```

---

### Task 18: On-chip classification (gate S4)

**Files:**
- Modify: `emg/firmware/logger/Cargo.toml` (optional `dsp`, `onchip` feature), `emg/firmware/logger/src/main.rs` (on-chip path)
- Create: `emg/emgpoc/onchip.py`, `emg/scripts/export_model.py`, `emg/scripts/compare_onchip.py`
- Generated: `emg/firmware/logger/src/model.rs`
- Test: `emg/tests/test_onchip.py`

**Interfaces:**
- Consumes: `dsp::{Model, Classifier}`, `frame::encode_class`, `render_model_rs`, `load_lda`.
- Produces: the `onchip` firmware (EMG frames plus one class frame per window, carrying the `t_us` of the window's last sample); `onchip_agreement(emg_t_us, filtered, class_t_us, class_ids, model, thr, win, settle_samples=4000) -> (fraction, n)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from emgpoc.features import window_features
from emgpoc.models import make_lda
from emgpoc.onchip import onchip_agreement


def setup_stream(seed: int = 0):
    rng = np.random.default_rng(seed)
    amp = np.repeat(rng.choice([0.01, 0.2], size=(20, 4)), 500, axis=0)  # 20 gestures x 0.25 s
    filtered = rng.normal(size=(10_000, 4)) * amp
    t = 1_000 + 500 * np.arange(10_000, dtype=np.int64)
    thr = np.full(4, 0.02)
    ends = np.arange(399, 10_000, 100)
    X = np.array([window_features(filtered[e - 399 : e + 1], thr) for e in ends])
    y = np.where(X[:, 0] > X[:, 1], "a", "b")
    model = make_lda()().fit(X, y)
    ids = np.array([list(model.classes_).index(p) for p in model.predict(X)])
    return t, filtered, t[ends], ids, model, thr


def test_identical_decisions_agree_fully():
    t, filtered, ct, ids, model, thr = setup_stream()
    agreement, n = onchip_agreement(t, filtered, ct, ids, model, thr, win=400, settle_samples=0)
    assert agreement == 1.0 and n == len(ct)


def test_disagreements_are_counted_and_settling_is_skipped():
    t, filtered, ct, ids, model, thr = setup_stream()
    flipped = ids.copy()
    flipped[-10:] = 1 - flipped[-10:]
    agreement, n = onchip_agreement(t, filtered, ct, flipped, model, thr, win=400, settle_samples=4000)
    assert n == len([e for e in range(399, 10_000, 100) if e >= 4000])
    assert agreement == (n - 10) / n


def test_frames_without_matching_samples_are_ignored():
    t, filtered, ct, ids, model, thr = setup_stream()
    agreement, n = onchip_agreement(t, filtered, ct + 1, ids, model, thr, win=400, settle_samples=0)
    assert n == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_onchip.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'emgpoc.onchip'`

- [ ] **Step 3: Implement**

```python
"""Check that the firmware's decisions equal the laptop's on the same raw samples."""

from __future__ import annotations

import numpy as np

from .features import window_features
from .models import StandardizedModel


def onchip_agreement(emg_t_us: np.ndarray, filtered: np.ndarray, class_t_us: np.ndarray,
                     class_ids: np.ndarray, model: StandardizedModel, thr: np.ndarray, win: int,
                     settle_samples: int = 4000) -> tuple[float, int]:
    """Recompute each on-chip decision from the streamed raw EMG and compare.

    A class frame carries the time of its window's last sample, so the laptop can rebuild
    exactly that window. The first settle_samples are skipped: the laptop's filters start
    from zero while the firmware's have been running since boot.
    Returns (agreement fraction, number of decisions compared).
    """
    index = {int(t): i for i, t in enumerate(emg_t_us)}
    classes = list(model.classes_)
    same = total = 0
    for t, cid in zip(class_t_us, class_ids):
        i = index.get(int(t))
        if i is None or i < max(win - 1, settle_samples):
            continue
        feats = window_features(filtered[i - win + 1 : i + 1], thr)[None, :]
        same += int(model.predict(feats)[0] == classes[int(cid)])
        total += 1
    return (same / total if total else float("nan")), total
```

- [ ] **Step 4: Run it to verify it passes, then the whole suite**

Run: `uv run pytest tests/test_onchip.py -q` → Expected: `3 passed`
Run: `uv run pytest -q` → Expected: `79 passed`

- [ ] **Step 5: Export the S3 model to Rust**

Write `emg/scripts/export_model.py`:

```python
"""S4: write the trained model into firmware/logger/src/model.rs for the on-chip build."""

import argparse
from pathlib import Path

from emgpoc.export_rs import render_model_rs
from emgpoc.models import load_lda

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=Path("data/models/lda_s3.npz"))
args = ap.parse_args()

model, thr, win, hop = load_lda(args.model)
out = Path(__file__).resolve().parents[1] / "firmware" / "logger" / "src" / "model.rs"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(render_model_rs(model, thr, win, hop))
print(f"wrote {out} with classes {[str(c) for c in model.classes_]}")
```

Run: `uv run python scripts/export_model.py`
Expected: `wrote .../firmware/logger/src/model.rs with classes [...]`, listing the 4 chosen classes.

- [ ] **Step 6: Add the on-chip path to the firmware**

Replace `emg/firmware/logger/Cargo.toml`:

```toml
[package]
name = "emg-logger"
version = "0.1.0"
edition = "2021"

[dependencies]
esp-hal = { version = "1.2", features = ["esp32c6", "unstable"] }
esp-bootloader-esp-idf = { version = "0.6", features = ["esp32c6"] }
esp-backtrace = { version = "0.20", features = ["esp32c6", "panic-handler", "println"] }
esp-println = { version = "0.18", default-features = false, features = ["esp32c6", "uart"] }
critical-section = "1.2"
heapless = "0.9"
frame = { path = "../frame" }
sensors = { path = "../sensors" }
dsp = { path = "../dsp", optional = true }

[features]
onchip = ["dep:dsp"]

[profile.dev]
opt-level = "s"

[profile.release]
opt-level = 3
debug = true
```

Replace `emg/firmware/logger/src/main.rs`:

```rust
//! Track S logger for ESP32-C6: 4 EMG channels at 2 kHz (MCP3208 on SPI) + 2 BNO055
//! quaternions at 100 Hz (I2C), streamed as binary frames over the native USB serial.
//! Debug text goes to UART0 (GPIO16), never to USB: USB carries only frames.
//! `--features onchip` also classifies on the chip and sends a class frame every 50 ms.
#![no_std]
#![no_main]

use core::cell::RefCell;

use critical_section::Mutex;
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    handler,
    i2c::master::{Config as I2cConfig, I2c},
    main,
    spi::{
        master::{Config as SpiConfig, Spi},
        Mode,
    },
    time::{Duration, Instant, Rate},
    timer::{timg::TimerGroup, PeriodicTimer},
    usb::usb_serial_jtag::UsbSerialJtag,
    Blocking,
};
use heapless::Deque;
use sensors::{bno055, mcp3208};

esp_bootloader_esp_idf::esp_app_desc!();

#[cfg(feature = "onchip")]
mod model; // generated: uv run python scripts/export_model.py

const SAMPLE_PERIOD: Duration = Duration::from_micros(500); // 2 kHz per channel
const IMU_PERIOD_US: u64 = 10_000; // 100 Hz
const QUEUE_LEN: usize = 1024; // ~0.5 s of slack for the main loop

struct Sample {
    t_us: u32,
    seq: u16,
    ch: [u16; frame::EMG_CHANNELS],
}

struct Sampler {
    spi: Spi<'static, Blocking>,
    timer: PeriodicTimer<'static, Blocking>,
    seq: u16,
}

static SAMPLER: Mutex<RefCell<Option<Sampler>>> = Mutex::new(RefCell::new(None));
static QUEUE: Mutex<RefCell<Deque<Sample, QUEUE_LEN>>> = Mutex::new(RefCell::new(Deque::new()));

fn now_us() -> u64 {
    Instant::now().duration_since_epoch().as_micros()
}

#[handler]
fn on_sample_tick() {
    critical_section::with(|cs| {
        let mut sampler = SAMPLER.borrow_ref_mut(cs);
        let Some(s) = sampler.as_mut() else { return };
        s.timer.clear_interrupt();
        let seq = s.seq;
        s.seq = seq.wrapping_add(1); // a dropped sample still uses up a seq: the host sees the gap
        let t_us = now_us() as u32;
        let mut ch = [0u16; frame::EMG_CHANNELS];
        for (i, v) in ch.iter_mut().enumerate() {
            let mut buf = mcp3208::command(i as u8);
            if s.spi.transfer(&mut buf).is_ok() {
                *v = mcp3208::parse(buf);
            }
        }
        // Queue full means the main loop fell behind: drop the sample.
        let _ = QUEUE.borrow_ref_mut(cs).push_back(Sample { t_us, seq, ch });
    });
}

fn bno055_start(i2c: &mut I2c<'static, Blocking>, addr: u8) -> bool {
    let mut id = [0u8];
    if i2c.write_read(addr, &[bno055::REG_CHIP_ID], &mut id).is_err() || id[0] != bno055::CHIP_ID {
        return false;
    }
    i2c.write(addr, &[bno055::REG_OPR_MODE, bno055::MODE_NDOF]).is_ok()
}

fn bno055_quat(i2c: &mut I2c<'static, Blocking>, addr: u8) -> [i16; 4] {
    let mut raw = [0u8; 8];
    match i2c.write_read(addr, &[bno055::REG_QUA_DATA_W_LSB], &mut raw) {
        Ok(()) => bno055::quaternion(raw),
        Err(_) => [0; 4], // all-zero quaternion = "no reading"; the host treats it as invalid
    }
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    let delay = Delay::new();
    let mut usb = UsbSerialJtag::new(p.USB_DEVICE);

    // MCP3208 at 1 MHz (its limit at 3.3 V is ~1.2 MHz). CS toggles per transfer = per conversion.
    let spi = Spi::new(p.SPI2, SpiConfig::default().with_frequency(Rate::from_khz(1000)).with_mode(Mode::_0))
        .unwrap()
        .with_sck(p.GPIO19)
        .with_miso(p.GPIO20)
        .with_mosi(p.GPIO18)
        .with_cs(p.GPIO21);

    // BNO055 stretches the I2C clock; 100 kHz keeps that well inside the default timeout.
    let mut i2c = I2c::new(p.I2C0, I2cConfig::default().with_frequency(Rate::from_khz(100)))
        .unwrap()
        .with_sda(p.GPIO22)
        .with_scl(p.GPIO23);
    delay.delay_millis(700); // BNO055 needs ~650 ms after power-up
    let imu_ok = bno055_start(&mut i2c, bno055::ADDR_BAND) && bno055_start(&mut i2c, bno055::ADDR_STUB);
    esp_println::println!("emg-logger: imu_ok={}", imu_ok);

    let timg0 = TimerGroup::new(p.TIMG0);
    let mut timer = PeriodicTimer::new(timg0.timer0);
    timer.set_interrupt_handler(on_sample_tick);
    timer.listen();
    // Start only once the ISR can find its state, or the first ticks would never be cleared.
    critical_section::with(|cs| {
        let mut slot = SAMPLER.borrow_ref_mut(cs);
        let sampler = slot.insert(Sampler { spi, timer, seq: 0 });
        sampler.timer.start(SAMPLE_PERIOD).unwrap();
    });

    #[cfg(feature = "onchip")]
    let model = dsp::Model {
        stages: &model::STAGES,
        thr: &model::THR,
        mean: &model::MEAN,
        scale: &model::SCALE,
        w: &model::W,
        b: &model::B,
        volts_per_count: model::VOLTS_PER_COUNT,
        hop: model::HOP,
    };
    #[cfg(feature = "onchip")]
    let mut classifier = dsp::Classifier::<{ model::N_CH }, { model::WIN }, { model::STAGES.len() }>::new(&model);
    #[cfg(feature = "onchip")]
    let mut class_seq: u16 = 0;

    let mut imu_seq: u16 = 0;
    let mut next_imu = now_us();
    loop {
        while let Some(s) = critical_section::with(|cs| QUEUE.borrow_ref_mut(cs).pop_front()) {
            let _ = usb.write(&frame::encode_emg(s.seq, s.t_us, &s.ch));
            #[cfg(feature = "onchip")]
            if let Some(class_id) = classifier.push(&model, &s.ch) {
                // t_us of the window's last sample, so the laptop can rebuild the same window.
                let _ = usb.write(&frame::encode_class(class_seq, s.t_us, class_id as u8));
                class_seq = class_seq.wrapping_add(1);
            }
        }
        if imu_ok && now_us() >= next_imu {
            next_imu += IMU_PERIOD_US;
            let band = bno055_quat(&mut i2c, bno055::ADDR_BAND);
            let stub = bno055_quat(&mut i2c, bno055::ADDR_STUB);
            let _ = usb.write(&frame::encode_imu(imu_seq, now_us() as u32, &band, &stub));
            imu_seq = imu_seq.wrapping_add(1);
        }
    }
}
```

- [ ] **Step 7: Build and flash both variants**

Run: `cd firmware/logger && cargo build --release && cargo build --release --features onchip`
Expected: both `Finished`.
Flash the on-chip build: `cargo run --release --features onchip`

- [ ] **Step 8: Check on-chip against laptop on real data**

Write `emg/scripts/compare_onchip.py`:

```python
"""S4 gate: do the firmware's decisions match the laptop's on the same samples?"""

import argparse
import time
from pathlib import Path

import numpy as np

from emgpoc.filters import FilterChain, default_stages
from emgpoc.models import load_lda
from emgpoc.onchip import onchip_agreement
from emgpoc.protocol import ADC_VOLTS_PER_COUNT, ClassFrame, EmgFrame, FrameParser, U32Unwrapper
from emgpoc.serialport import open_stream

GATE = 0.95

ap = argparse.ArgumentParser()
ap.add_argument("--port")
ap.add_argument("--model", type=Path, default=Path("data/models/lda_s3.npz"))
ap.add_argument("--seconds", type=float, default=60.0)
args = ap.parse_args()

model, thr, win, _ = load_lda(args.model)
ser, parser, frames = open_stream(args.port), FrameParser(), []
print(f"Recording {args.seconds:.0f} s: rest, then cycle through your gestures...")
end = time.monotonic() + args.seconds
while time.monotonic() < end:
    frames += parser.feed(ser.read(4096))

unwrap = U32Unwrapper()
emg_t, emg, cls_t, cls_id = [], [], [], []
for f in frames:
    t = unwrap(f.t_us)
    if isinstance(f, EmgFrame):
        emg_t.append(t)
        emg.append(f.ch)
    elif isinstance(f, ClassFrame):
        cls_t.append(t)
        cls_id.append(f.class_id)
if not cls_t:
    raise SystemExit("no class frames: flash the firmware built with --features onchip")
filtered = FilterChain(default_stages(), 4).process(np.array(emg) * ADC_VOLTS_PER_COUNT)
agreement, n = onchip_agreement(np.array(emg_t), filtered, np.array(cls_t), np.array(cls_id), model, thr, win)
print(f"Compared {n} decisions: agreement {agreement:.3f}")
print(f"Gate S4 on-chip (>= {GATE}): {'PASS' if agreement >= GATE else 'FAIL'}")
```

Charger unplugged, band on, 10 min settle. Run: `uv run python scripts/compare_onchip.py`, rest, then cycle through the chosen gestures for 60 s.
Expected: `Compared N decisions: agreement 0.9xx` and `Gate S4 on-chip (>= 0.95): PASS`.

- [ ] **Step 9: Daily practice on the chip's decisions**

Every day for 2 weeks: `uv run python scripts/trainer.py --onchip` (10 min). The learning curve prints at the end of each session.

- [ ] **Step 10: Commit**

```bash
git add emg/firmware/logger emg/emgpoc/onchip.py emg/scripts/export_model.py emg/scripts/compare_onchip.py \
        emg/tests/test_onchip.py emg/data/practice.csv
git commit -m "Run the classifier on the ESP32-C6 and verify it against the laptop"
```

**Gate S4:** on-chip agreement ≥ 95 %, and at least 10 practice days in the learning curve.
