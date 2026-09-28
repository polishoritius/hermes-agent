# Retirement Notice

**STATUS:** RETIRED / ARCHIVED REFERENCE

**PURPOSE:** Historical reference for a 2026-09 local experiment extending [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with four Project-Poiesis-flavored local patches.

## Important

- **This repository is NOT Project Poiesis Core.** Project Poiesis Core is `polishoritius/Trivia-in-Invaluable-Lab`.
- **Project Poiesis Core does NOT depend on this fork.** A cross-repository audit found zero references to this repository, its code, or its secrets anywhere in Core.
- **This fork contains an opt-in PP Context Bridge (`agent/pp_context_bridge.py`) that reads from and calls into Core** — the dependency runs *from this fork toward Core*, never the reverse.
- **That bridge was never found to have entered active use.** A dedicated local usage audit (shell history, session logs, config, git reflog, cron/agent logs) found no evidence of it being invoked beyond the single development/test session that built it on 2026-09-24. The whole local Hermes Agent installation has shown no activity of any kind since 2026-09-24, independent of the bridge specifically.
- **The historical default bridge target `C:\GitHub\Project-Poiesis-Work` may be stale and must not be treated as a canonical checkout.** At last check it was materially behind Core's `origin/main`.
- **The fork is extremely behind upstream** (tens of thousands of commits as of retirement). Do not attempt normal upstream synchronization without a deliberate rebuild decision.
- **If the concept is revived later, prefer starting from fresh upstream** and selectively porting the four local patches below, rather than rebasing this fork.

## Naming note — read this if you found this repo by searching "Hermes"

Project Poiesis internal: **PP-Hermes-Role** — a constrained execution/planning role defined inside Core (`Trivia-in-Invaluable-Lab`'s `scripts/hermes_*.py`), never an external agent.

This repository: **Hermes-Agent-Fork** — a fork of NousResearch's general-purpose personal AI agent product.

**Shared naming does NOT imply Core depends on this repository.** But this archived fork *does* contain an old opt-in bridge that depends on Core in the reverse direction (see above) — do not assume total independence either. Verify direction before acting on anything here.

## Local patches preserved (4 commits, all on `main`)

| Commit | Title | Purpose |
|---|---|---|
| `16656cc4ff` | fix(model-switch): normalize current_provider alias | Bugfix in the model picker/custom-endpoint alias normalization. |
| `adb50571e3` | feat: add local light chat path | A lightweight chat entry point (`agent/light_chat.py`) alongside the main agent core. |
| `b523ddd102` | feat: add Project Poiesis context bridge | `agent/pp_context_bridge.py` — reads Project Poiesis's persona ontology/registry files and dispatches to Core's `scripts/persona_chatter.py::run_chatter()`, targeting a local Core clone path (`PP_REPO_PATH`, default `C:\GitHub\Project-Poiesis-Work`). Read-mostly; explicitly never touches Core's Growth/VERIFY/Council/Write/Discord modules. |
| `03930f90ff` | feat: add deterministic fast conversation router | A natural-language router (`agent/fast_router.py`) plus display formatters (`agent/pp_bridge_formatter.py`) that make the PP Context Bridge reachable without explicit `/pp ...` commands. |

## Workflows

All 26 CI/automation workflows inherited from upstream have been left in place for historical reference. The four with independent scheduled/background triggers have been **disabled** (via the GitHub Actions API, no file changes) as part of this retirement:

- `Install & Update E2E` (was: every 12 hours)
- `OSV-Scanner` (was: weekly)
- `Skills Index Freshness Check` (was: every 4 hours)
- `Build Skills Index` (was: twice daily)

No workflow in this repository posts externally, deploys, or calls a real Provider on a schedule going forward.

## Credentials

`gh secret list` on this repository returns **zero configured secrets**. Several workflow files (inherited from upstream) reference secret names (`DOCKERHUB_TOKEN`, `DOCKERHUB_USERNAME`, `APP_PRIVATE_KEY`, `VERCEL_DEPLOY_HOOK`, `GH_IMAGE_SESSION_TOKEN`) — none of these are actually set here, so those workflow steps would only ever have run with empty values. No secret values were inspected, and none were removed as part of this retirement.
