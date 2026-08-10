# Archived XFOIL Source

- Upstream project: <https://web.mit.edu/drela/Public/web/xfoil/>
- Source mirror: <https://github.com/RobotLocomotion/xfoil>
- Pinned mirror commit: `d11a1544b53623c01bb3b0cceb5862311be0e1f8`
- Program version reported by the vendored source: `6.97`
- Original mirror snapshot SHA-256:
  `08f4fae92150813a7e34cdeead4b20d9e50e25156ca1658615a4dfea8b47187f`
- Deterministic task archive: `data/xfoil-source.tar.gz`
- Deterministic task archive SHA-256:
  `fabf00130639aadf5365eea07446df1a267ff907b5f6f0b9ef6d14aa039cde2c`

The task archive contains the complete 252-file pinned source tree under one
`xfoil-source/` root. Git metadata is excluded. Archive entries are sorted and
normalized to uid/gid 0, fixed modes, and an epoch mtime; the gzip header also
uses an epoch mtime. `data/xfoil-source.tar.gz.sha256` is checked before every
Docker extraction. The rollout image receives the verified extracted tree at
`/data/xfoil-source`, so source inspection and runtime-offline builds do not
depend on the network.

The archived components have distinct upstream terms:

- XFOIL source headers (including `src/xfoil.f`) grant
  **GPL-2.0-or-later**; the complete GNU GPL v2 text is preserved at
  `xfoil-source/src/gpl.txt`.
- Plotlib source headers (including `plotlib/plt_base.f`) grant the
  **GNU Library General Public License v2 or later**, represented by SPDX
  `LGPL-2.0-or-later`; its complete notice is preserved at
  `xfoil-source/plotlib/GPL-library`.
- The separately bundled Rust ceiling reference is **MIT** licensed. Its
  notice is preserved with the maintainer-only ceiling sources (not present
  in the rollout image).
- Task and framework material outside those components is Apache-2.0.

These notices describe separate components; Apache-2.0 task material does not
relicense XFOIL or Plotlib. The repository and rollout image include GPL/LGPL
source and an XFOIL executable built from it. No commercial-delivery approval
or permissive relicensing is claimed. Any downstream distribution needs a
specific legal review of source, notice, and corresponding-source obligations.
