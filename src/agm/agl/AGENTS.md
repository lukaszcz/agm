# AgL area guidance

## Architecture and project layout

Read @docs/arch/agl/index.md to understand AgL implementation architecture.

**IMPORTANT**: Update docs/arch/agl/**/*.md whenever AgL implementation architecture changes – always keep these files up-to-date with the codebase.

The primary purpose of architecture docs in docs/arch/agl/**/*.md is to provide agents with a quick but comprehensive overview of the system's architecture and the codebase. Treat the docs as an onboarding guide. When updating, do not add brittle implementation details, but do include info on where to find relevant codebase references. Be succinct, not verbose. Provide architectural overview, not mechanism details. Match the existing writing style and succinctness level.

## AgL language reference

The AgL reference documentation is in `docs/agl/reference/`. This reference is written from the language user perspective (audience is expert programmers familiar with CS / PL concepts).

**IMPORTANT**: The documentation MUST NOT reference the implementation in any way - ONLY describe the AgL language.

**IMPORTANT**: Each change to the AgL language syntax or semantics MUST be accompanied by a corresponding change in the AgL reference documentation (`docs/agl/reference`).

The documentation MUST NOT include historical statements about abandoned or superseded features / syntax / semantics (avoid expressions like "no longer", "previous", "formerly"), and it MUST NOT mention previous AgL language versions or any version numbers. The AgL language reference describes ONLY the CURRENT language version. Remove ALL superseded information when updating.

## Testing

Whenever you add a new language feature, create end-to-end test program examples under `tests/agl/programs/` exercising this feature thoroughly (in combination with other language features). Follow TDD - add end-to-end test program examples as the FIRST step before any other implementation work.
