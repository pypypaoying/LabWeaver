# Analysis skills attribution

The four `SKILL.md` files are Chinese abridgements and adaptations by the
AnaliyMaster project of selected skills from Anthropic's
[`knowledge-work-plugins`](https://github.com/anthropics/knowledge-work-plugins)
repository, pinned at commit `2574e29377c7173d0cecffef29a6ecbd7f8ccb5d`.

The upstream work uses Apache License 2.0. Its complete legitimate license text
matches the canonical Apache 2.0 text after whitespace normalization. A canonical
copy from the Apache Software Foundation is included in
`THIRD_PARTY_LICENSE.txt`; no legitimate license term or attribution notice
was removed. The pinned repository root had no separate `NOTICE` file.

The upstream LICENSE at this commit has blob SHA
`f53c679acba79cbb292961fd9fb9e87ca20eab22` (11607 ASCII bytes).
It has unrelated non-license material after the complete license ending.
That tail is omitted from this distribution and never enters model context.
The included canonical copy comes from
https://github.com/apache/.github/blob/main/LICENSE
(blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`), and its license terms were
also checked against https://www.apache.org/licenses/LICENSE-2.0.txt.

Original sources:

| Local skill | Upstream path | Upstream Git blob SHA |
| --- | --- | --- |
| data-exploration | data/skills/explore-data/SKILL.md | `f1e7032d0abee77903fc3abbfdc7646e383d4993` |
| statistical-analysis | data/skills/statistical-analysis/SKILL.md | `59cc87de24a0c1b2b5b1d7ba2d8d10eb314a6557` |
| data-visualization | data/skills/data-visualization/SKILL.md | `409cce60f59b463a8386c83cde03dc7e70be93f3` |
| analysis-validation | data/skills/validate-data/SKILL.md | `30a1bac41bf9679dc7d96b7ec59a116f03ec0575` |

Modifications: translate and shorten task guidance; remove warehouse connectors,
external services, arbitrary filesystem access, optional dependency installation,
and instructions to expand the user's assignment. Add the project's immutable
string snapshot, stable column IDs, `emit_*` artifact protocol, finite repair
budgets and independent-verification boundaries. This is adapted guidance,
not new execution tools or independent correctness proofs. It does not install
or run the upstream Claude plugin, and does not require a Claude model.

Fingerprints use canonical UTF-8 with LF newlines, including frontmatter and
adaptation notices. The catalog is loaded separately from bodies; a reviewed
body is opened only for an explicit valid skill ID. Working-directory files or
paths in CSV/document data cannot add skills. License and attribution files
are distributed with the package but are not loaded into model context.
