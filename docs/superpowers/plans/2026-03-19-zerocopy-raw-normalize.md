# Zerocopy Raw Normalize Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the macOS zerocopy path so raw transport tensors stay internal while the public `ZEROCOPY` path continues to return normalized `RGB + flipped` output.

**Architecture:** Keep the native macOS bridge unchanged for now and move the raw-to-public normalization boundary into `ObservationConverter`. Minimize diff by touching the current macOS zerocopy branches only, adding a small normalization helper and focused tests.

**Tech Stack:** Python, PyTorch, pytest, existing CraftGround environment converter code

---

## Chunk 1: MacOS Zerocopy Boundary

### Task 1: Add failing tests for macOS zerocopy public normalization

**Files:**
- Modify: `tests/python/unit/test_observation_converter.py` or nearest existing converter test file
- Modify: `src/craftground/environment/observation_converter.py`

- [ ] **Step 1: Locate or create the nearest unit-test file for `ObservationConverter`**

Run: `rg -n "ObservationConverter|initialize_zerocopy|ScreenEncodingMode.ZEROCOPY" tests src/craftground`
Expected: identify the closest existing test home; only create a new test file if no suitable one exists.

- [ ] **Step 2: Write a failing test for the macOS path that preserves public output semantics**

Test idea:

```python
def test_macos_zerocopy_returns_rgb_flipped_from_raw_transport(...):
    converter = ObservationConverter(...)
    raw = fake_tensor_with_known_bgra_rows(...)
    converter.internal_type = ObservationTensorType.APPLE_TENSOR
    converter.last_observations[0] = raw

    obs, _ = converter.convert(fake_observation(...))

    assert torch.equal(obs, expected_rgb_flipped)
```

- [ ] **Step 3: Write a failing test that initialization stores raw transport data without normalization**

Test idea:

```python
def test_initialize_zerocopy_stores_raw_apple_tensor(...):
    converter = ObservationConverter(...)
    raw = fake_raw_tensor(...)
    mock_initialize_from_mach_port.return_value = raw

    converter.initialize_zerocopy(fake_mach_port_bytes)

    assert converter.last_observations[0] is raw
```

- [ ] **Step 4: Run the targeted tests and verify they fail for the intended reason**

Run: `pytest <targeted-test-file> -k zerocopy -v`
Expected: FAIL due to missing helper/incorrect state handling, not import or syntax errors.

- [ ] **Step 5: Commit the failing tests**

```bash
git add <targeted-test-file>
git commit -m "test: add macos zerocopy normalization coverage"
```

### Task 2: Implement raw/public separation with minimal diff

**Files:**
- Modify: `src/craftground/environment/observation_converter.py`
- Test: `tests/python/unit/test_observation_converter.py` or nearest existing converter test file

- [ ] **Step 1: Add a small helper for macOS raw-to-public normalization**

Implementation target:

```python
def _normalize_apple_zerocopy_tensor(self, tensor: "TorchArrayType") -> "TorchArrayType":
    return tensor.clone()[:, :, [2, 1, 0]].flip(0)
```

Keep this helper private and adjacent to zerocopy handling.

- [ ] **Step 2: Make `initialize_zerocopy()` store the raw Apple tensor only**

Requirements:

- no normalization in initialization
- keep storing the raw transport tensor
- ensure type tracking updates the field used by `convert()`

- [ ] **Step 3: Update the public `ZEROCOPY` branch to call the helper at return time**

Requirements:

- preserve current `RGB + flipped` behavior for Apple tensors
- do not broaden the change into unrelated output types
- keep non-Apple branches unchanged unless required for correctness

- [ ] **Step 4: Fix the touched zerocopy state variable inconsistency if still present**

The code currently mixes `internal_type` and `observation_tensor_type`. The implementation should ensure the active state variable used by `convert()` is updated consistently in the touched path.

- [ ] **Step 5: Run the targeted tests and verify they pass**

Run: `pytest <targeted-test-file> -k zerocopy -v`
Expected: PASS

- [ ] **Step 6: Run any neighboring converter tests**

Run: `pytest <targeted-test-file> -v`
Expected: PASS with no zerocopy regressions in nearby behavior.

- [ ] **Step 7: Commit the implementation**

```bash
git add src/craftground/environment/observation_converter.py <targeted-test-file>
git commit -m "refactor: separate raw macos zerocopy transport from public normalization"
```

## Chunk 2: Verification And Handoff

### Task 3: Verify branch state and prepare for PR

**Files:**
- Modify: `docs/superpowers/specs/2026-03-19-zerocopy-raw-normalize-design.md`
- Modify: `docs/superpowers/plans/2026-03-19-zerocopy-raw-normalize.md`

- [ ] **Step 1: Run a focused verification sweep**

Run:

```bash
pytest <targeted-test-file> -v
git status --short
git log --oneline -5
```

Expected:

- tests passing
- only intended files modified
- branch contains the new zerocopy commits

- [ ] **Step 2: Update docs if implementation differs from the planned low-diff shape**

Keep spec and plan aligned with actual code decisions.

- [ ] **Step 3: Prepare PR summary**

Include:

- internal raw/public separation
- no public `ZEROCOPY` behavior change
- future raw-mode support enabled

- [ ] **Step 4: Commit any doc adjustments**

```bash
git add docs/superpowers/specs/2026-03-19-zerocopy-raw-normalize-design.md docs/superpowers/plans/2026-03-19-zerocopy-raw-normalize.md
git commit -m "docs: record zerocopy raw normalize design and plan"
```
