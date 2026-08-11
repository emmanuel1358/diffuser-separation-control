---
name: task-author-changelog
description: Maintains the reverse-chronological task-author changelog for shared grader, runtime, validator, harness, schema, and authoring-contract changes. Use whenever shared framework code changes could affect task APIs, scoring, calibration, validation, artifacts, resources, security behavior, migrations, or required image rebuilds.
---

# Task Author Changelog

Keep `docs/CHANGELOG.md` synchronized with task-author-visible shared changes.

## Decide whether an entry is required

Update the changelog when a shared change affects any of:

- public classes, functions, arguments, defaults, or serialized schemas;
- scoring, calibration, evidence, baselines, or fault attribution;
- submission formats, output paths, artifact loaders, or constraints;
- validator, release, ground-truth, or migration requirements;
- runtime resources, tool behavior, sandboxing, or hidden environments; or
- task image rebuild, lock regeneration, or author action.

Do not add an entry for a behavior-preserving internal refactor, test-only
change, typo fix, or documentation clarification.

## Update workflow

1. Inspect the complete diff, including tests and generated schema/example
   updates.
2. Identify concrete effects on existing and new tasks.
3. Read `docs/CHANGELOG.md` and update the latest related entry when the work is
   part of the same change set; otherwise prepend a new entry.
4. Use an ISO 8601 heading with local UTC offset:
   `## YYYY-MM-DDTHH:MM:SS±HH:MM — Summary`.
5. Keep entries in reverse chronological order.
6. Lead with required actions and breaking migrations, then summarize new or
   changed behavior. Name exact APIs, flags, schema versions, and commands.
7. State when an image rebuild, calibration regeneration, evaluation-plan
   refresh, or trusted-CI rerun is required.
8. Do not claim platform fixes that are outside this repository.
9. Verify `README.md` still links to `docs/CHANGELOG.md`.

## Entry shape

```markdown
## 2026-08-10T13:28:00-07:00 — Short summary

### Action required

- Migration or rebuild steps.

### Changed

- Author-visible API or behavior changes.
```

Use only the sections that add value. Keep each bullet actionable and avoid
commit-by-commit implementation detail.
