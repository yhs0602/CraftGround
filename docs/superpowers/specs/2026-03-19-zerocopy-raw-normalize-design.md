# Zerocopy Raw Normalize Design

## Summary

Refactor the macOS zerocopy path so the internal transport tensor remains in its native raw layout, while the public `ZEROCOPY` API continues to expose the current normalized image contract (`RGB + flipped`). This keeps the current user-facing behavior stable, reduces coupling between the custom MPS bridge and presentation logic, and creates a clean path to expose a future raw mode without redesigning the pipeline again.

## Current Problem

The macOS zerocopy path currently mixes three concerns in one flow:

1. Transporting the framebuffer into Python as an MPS-backed tensor.
2. Representing the native producer layout.
3. Returning the user-facing normalized layout.

This makes the custom bridge responsible for data semantics it should not own. It also forces the current implementation to rely on `clone()` in the public path as both a correctness workaround and a format-conversion boundary.

## Decision

Use the approved option:

- Keep the internal macOS zerocopy representation as native raw data.
- Preserve the existing public `ZEROCOPY` output semantics.
- Move normalization (`RGB + flip`) to the public conversion boundary.
- Do not expose a new raw public mode in this change.

This is the lowest-diff option that improves maintainability without giving up future extensibility.

## Scope

In scope:

- macOS zerocopy internals in `observation_converter.py`
- separation between raw transport tensor and normalized public tensor
- tests for the new internal/public boundary behavior
- cleanup of the internal zerocopy type-tracking bug if encountered in the touched path

Out of scope:

- changing the public API shape of `ScreenEncodingMode.ZEROCOPY`
- introducing a new public raw mode
- redesigning the native C++ bridge
- changing non-macOS behavior unless required for shared helper safety

## Design

### Internal Representation

`initialize_zerocopy()` should store the macOS tensor exactly as produced by the native bridge. No channel reorder, Y-flip, or user-facing normalization should happen at initialization time.

The raw tensor is the transport artifact. It is allowed to be in producer-native layout as long as the public boundary is responsible for converting it into the documented output format.

### Public Representation

The existing `ScreenEncodingMode.ZEROCOPY` path should continue to return the same logical image as before. The normalization step should happen in a helper near the public return site, not during bridge setup.

For the macOS path this means:

- source: raw Apple tensor
- public output: normalized `RGB + flipped`

This keeps behavior stable for users while isolating future raw-mode support to a small API addition later.

### State Ownership

The converter should make the distinction between:

- raw zerocopy tensor state
- normalized public observation state

The implementation may use separate fields or a clearly named helper-based boundary, but the raw/public distinction must be explicit in code. Avoid clever state machines. This should stay easy to follow inside the existing file.

### Future Raw Mode

This refactor should make a later `ZEROCOPY_RAW` or equivalent mode straightforward:

- the raw tensor already exists as a first-class internal concept
- the later feature would only add a new public branch that returns the raw tensor directly
- the normalized branch would remain unchanged

## Testing Strategy

Add targeted tests around the converter boundary:

- macOS zerocopy initialization stores the raw transport tensor without premature normalization
- public `ZEROCOPY` still returns the same normalized semantics as before
- normalization helper is the only place that performs the macOS format conversion

Tests should focus on layout semantics and call boundaries, not the real native bridge.

## Risk Management

Primary risk:

- accidentally changing the user-visible output of `ZEROCOPY`

Mitigation:

- keep the public return contract unchanged
- isolate the new normalization helper
- cover the output semantics with tests

Secondary risk:

- touching shared converter state and unintentionally affecting non-macOS paths

Mitigation:

- keep the diff local to the macOS-specific branches whenever possible
- avoid broad refactors outside the touched boundary

## Success Criteria

The change is successful when:

- current `ZEROCOPY` callers still receive `RGB + flipped` output
- the macOS zerocopy path internally preserves a raw transport tensor
- the code clearly separates raw transport from public normalization
- the refactor makes a future raw public mode a small additive change
