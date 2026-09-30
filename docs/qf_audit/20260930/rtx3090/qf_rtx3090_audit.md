# RTX 3090 audit — Phase B update, 2026-09-30

Phase B completed the permitted remote/report checks and prepared a controlled Git connection. **Scientific synchronization is incomplete:** the exact source CODE SHA is unavailable remotely, and the supplied handoff omits the independent expected REPORT SHA. No scientific baseline was selected, no data were transferred, and no experiments were launched. The original Phase A snapshot is preserved unchanged in `phase_a_archive/20260930T143629Z/` and reproduced below as historical evidence.

## Current Git identities and blockers

| Item | Identity / outcome | Evidence status and method |
| --- | --- | --- |
| Verified supplied project remote | `git@github.com:bigchan22/lob_seq2seq.git` | verified from supplied handoff, remote advertisement, matching published report identity |
| Original local scientific branch / CODE SHA / dirty state | no verified checkout; branch/SHA unknown; dirty state unknown | preserved Phase A finding; new bridge is not a pre-existing scientific checkout |
| Source inspected CODE SHA | `feb10a130bb036617229d6c4c9bc4a96df3b9fe6` | supplied source audit identity |
| Exact CODE availability | **unavailable**: narrow filtered full-SHA fetch returned `upload-pack: not our ref` | verified transport response; source branch also not advertised in targeted ref query |
| Source dirty caveat | 47 reference symlink-to-file changes and 15 executable-mode changes; source reports no core Python/config/test content diff | source-reported; filesystem delta not reconstructed or transferred |
| Source audit branch | `audit/qf-a5000-20260930-142604` | verified currently advertised ref |
| Source REPORT SHA observed and fetched | `0ebdb6782f2322e346dfb0c303762aab8635419c` | verified against current remote advertisement; **not independently confirmed against a supplied expected SHA** |
| Missing source handoff field | full expected REPORT SHA/publication receipt | asked user for it; the supplied JSON refers to a separate receipt/final response |
| Already-published main / report parent | `073cd89b556224eeae22413c3c05188a947e990a` | independently advertised, direct parent of source REPORT; used only as audit publication base |
| Scientific ancestry / comparison worktree | unavailable / not created | no original local CODE and exact source CODE cannot be fetched |
| Own audit publication branch | `audit/qf-rtx3090-20260930-1506` | isolated sparse worktree prepared from published main; commit/push receipt is separate |

Do not equate REPORT SHA with CODE SHA. The published report contains audit evidence about source code it does not include. Published `main` has an older `src/` layout and is not substituted for the inspected `lob_forecasting/` scientific snapshot. The September historical candidate remains unverified. No conclusion of identical/ahead/behind/diverged scientific branches is possible.

## Acquisition and preservation

The remote was contacted using existing authentication with noninteractive strict host verification. Protocol v2 advertised `fetch=shallow wait-for-done filter`. A new repository was created at `GIT_BRIDGE_ROOT` using `--filter=blob:none --no-checkout --single-branch --depth=2 --no-tags --no-recurse-submodules`, with `GIT_LFS_SKIP_SMUDGE=1`. No existing remote was changed. No unrestricted fallback occurred.

Before reading report blobs, the local object store contained exactly two commits, 29 trees, and zero blobs. `git ls-tree -r` inspected paths without asking for missing blob sizes. The fetched REPORT tree has 118 paths, including 61 under `data/`, and no AGENTS.md, .gitattributes, or .gitmodules. Source blob metadata reports published main contains 383,485,889 bytes of ordinary tracked content and its wider scientific history includes a 134,496,215-byte checkpoint. These byte totals are source-reported; no scientific blobs were fetched to remeasure them. LFS skip-smudge alone would not protect these ordinary Git blobs.

Seven explicit audit files were fetched and saved under `RECEIVED_SOURCE_AUDIT_ROOT/0ebdb6782f2322e346dfb0c303762aab8635419c/`: the source Markdown report, handoff JSON, source/run CSV inventories, Git-blob inventory, source summary, and run summary. Their combined uncompressed content is 398,267 bytes. Local object enumeration confirmed every downloaded blob was one of these seven audit records. No workbook, cache, prediction array, or checkpoint was materialized. The filtered clone is shallow and does not establish full-history coverage.

`WORKSPACE_ROOT` in the unchanged received source files refers to the source machine, even though both machines use similar path strings. The supplied handoff and fetched report agree on the source CODE SHA, branch, dirty caveat, remote, publication base, hardware, and six representative data hashes. They are different serialized files because the published copy uses aliases and reflects its publication-stage snapshot; no byte-equality claim is made for those two handoff files.

The own publication worktree is `AUDIT_PUBLICATION_WORKTREE`. It was created with skip-smudge and `--no-checkout`, then an explicit non-cone sparse path limited materialization to `docs/qf_audit/20260930/rtx3090/`. Before audit files were copied in, status was clean and only the `.git` pointer existed. No inherited scientific file was materialized. The own branch parent was checked against current advertised `main`, so no unpublished scientific commits are incidentally included as ancestors. Staging is limited to named sanitized audit files. The full own REPORT SHA and verified push outcome will be recorded in `publication_receipt.json` and the exact local handoff after publication, without amending the report commit to insert its own hash.

## Cross-server data and environment comparison

The source handoff corrects the second machine's expected hardware: it reports **four RTX 4090s**, not RTX A5000s. This host's four RTX 3090s were directly verified in Phase A. Hardware does not establish scientific authority or recency.

| Category | Source audit evidence | RTX3090 finding |
| --- | --- | --- |
| Aggregates | 27 supplied workbooks; 161,779,048 bytes; one Sheet1 and 45 columns in all header scans | no corresponding accessible local files identified; none transferred |
| Derived runtime caches | 27 top-level CSVs; 204,319,112 bytes | none transferred; compatibility and byte identity unresolved |
| Coverage of source audit | all workbook headers, three workbooks fully scanned, all 27 top-level CSVs scanned | source report parsed; no local workbook/cache scan |
| Representative identity | six source SHA-256 hashes, three workbooks plus three caches, consistent across source handoff/report | no local scientific hashes to compare; matching names/sizes would not suffice |
| Existing results | 129 mixed inventory rows; 66 distinct QF fit artifact sets according to source classification | no local executions added; local run inventory remains header-only |
| Historical environment | Python 3.10.19, torch 2.5.1, NumPy 2.0.1, pandas 2.3.3, sklearn 1.7.2, openpyxl 3.1.5 | local default Python 3.9.13 has no torch metadata |
| Alternate environment | source mpnn: Python 3.9.16, torch 1.12.1/CUDA 11.3; limited source CPU imports passed | local mpnn: Python 3.9.13, torch 2.7.1+cu118, NumPy 1.21.5, pandas 1.4.4, sklearn 1.0.2, openpyxl 3.0.10; metadata only |
| Launch readiness | historical lob interpreter permission denied on source; hardcoded old launch/data paths and broken worktree pointers reported | exact source code unavailable; no project import or runtime compatibility established |

One source copy of all 27 workbooks plus the 27 runtime CSVs totals **366,098,160 bytes (349.14 MiB)**, obtained by summing the received source inventory and cross-checking its summary. This is an input-storage estimate, not a complete disk/RAM budget; environments, temporary tensors, other caches, checkpoints, predictions, and new outputs are excluded. Phase A measured approximately 463.3 GiB available locally. The source inventory lists the exact required stems for a later authorized provisioning step. No provisioning was performed.

Source runtime reportedly discovers the top-level CSVs directly; workbook conversion is separate and skips existing CSV destinations without a freshness/hash check. This means source files and caches must be checked separately. Without local counterparts and the exact generating code, cross-server data compatibility is unknown. Provider aggregation, filling, timing, units, and corporate-action conventions remain undocumented.

## Scientific evidence received, not locally recomputed

The source audit describes an August legacy study: 13 no-investor features, daily per-asset forward-fill then zero-fill, a constructed 09:00–15:20 ten-minute grid, and 38 labels from 39 grid points. These are code-imposed constructions, not claims of 39 exchange observations. Labels use float32 midpoint returns to the next grid endpoint with thresholds ±0.0001 and Down/Flat/Up class order 0/1/2; nonpositive current midpoints default Flat. No target crosses days. Fixed log/log1p transforms do not have fitted population statistics; corrected UM alone has a train-only scaler.

The source report assigns complete common dates to train (2015-07-01–2016-11-23, 345 days), validation (2016-11-24–2017-02-03, 49 days), and already examined development holdout (2017-02-06–2017-06-30, 99 days). Validation selected checkpoints and tuning. The holdout is not untouched. Source metrics distinguish normalized sklearn natural-log loss from paired-analysis clipping at 1e-12 without renormalization. Its saved-prediction checks report matching full-support key/label hashes and no seed probability ensemble; peer-intervention support is separately restricted and changes model history. None of these checks were rerun here because prediction arrays and scientific source are unavailable locally.

| Source-reported Stage-2 result | Validation NLL | Development-holdout NLL |
| --- | --- | --- |
| UA seed 42 | 0.887593050 | 0.908952596 |
| UC seed 42 | 0.903287486 | 0.922755800 |
| UA mean of three per-seed scores | 0.887126571 | 0.908244411 |
| UC mean of three per-seed scores | 0.904152693 | 0.922516639 |

The source-reported seed-42 validation UA−UC difference is −0.015694436. These recovered August records do not match the supplied historical 0.886570 / −0.017301 / three-seed 0.886392 anchors. This does not disprove records on an uninspected machine or revision. GRU_U1 and trained lag-aware executions were not recovered by that source audit. Proposed qf_v2, rolling confirmation, shared-query controls, and TLOB are not promoted to completed experiments.

The 129 source inventory rows include aggregate analyses, copied references, older invocations, and individual fits; they do not mean 129 independent trainings. Their completion states and evidence completeness remain separate. Original per-fit dirty/source/cache provenance is incomplete. `run_inventory.csv` in this local audit remains empty of local executions; the received source CSV is kept separately with its original provenance.

## Checks, readiness, and reconciliation

Verified locally: supplied remote identity and filter capability; current advertised REPORT/main refs; fetched REPORT identity against that advertisement; exact CODE fetch rejection; source report/hand-off field consistency; metadata counts and storage sums; seven-blob audit-only acquisition; source REPORT's direct published parent; sparse audit publication setup; JSON/CSV parsing and audit-file checksums. No third-party imports, arbitrary entrypoints, project tests, checkpoint loads, prediction regeneration, or scientific cache writes were justified or performed.

Current readiness is separate by category: hardware inventory verified; code blocked by missing inspected SHA and unresolved original local checkout; source/cache presence and byte compatibility unestablished locally; candidate environment installation metadata present but project runtime untested; source audit metadata available with independent expected REPORT SHA still pending. The connection is an **audit bridge**, not synchronization of complete programs or datasets.

The coordinator should obtain the source publication receipt/full expected REPORT SHA, then arrange safe access to the exact CODE SHA or a verified code-only export mapped to it, with the dirty filesystem delta enumerated. Published old main must not substitute silently. Any accessible original RTX3090 project should then be compared before selecting a baseline. Keep the source August results, unresolved September/GRU anchors, and proposed qf_v2 protocols distinct. No merge/rebase, code reconstruction from prose, data transfer, or experiment launch is part of this phase.

Current outputs: `WORKSPACE_ROOT/qf_rtx3090_audit.md`, `qf_rtx3090_handoff.json`, `source_inventory.csv`, `run_inventory.csv`, `cross_server_comparison.json`, and `audit_files.sha256`. Exact acquisition evidence is in the local `phase_b_*.json` files; the source audit files are in the received-report directory above. Sanitized publication copies are under `AUDIT_PAYLOAD/`. The immutable Phase A archive retains its original checksum manifest.

---

# Preserved Phase A report — historical snapshot before the handoff

All statements below about missing handoffs/remotes/publication refer to Phase A. The current Phase B findings above supersede those operational statuses while preserving the pre-synchronization observations.

# RTX 3090 server — initial independent KRX audit

Audit date: 2026-09-30 UTC. Phase A completed within the stated accessible search scope. Phase B is **pending_source_handoff**. No A5000 SERVER_HANDOFF was supplied. The machine identity is verified by its GPU inventory; its scientific code and data location remain unresolved.

## Findings and evidence convention

The starting directory `WORKSPACE_ROOT` was empty before audit outputs were created and is not a Git checkout. Bounded discovery found no verified KRX project repository, source aggregates, caches, or run artifacts. This is **not** a finding that they are absent from the entire server. A running Jupyter process owned by `PROTECTED_PROCESS_OWNER` provides a location clue, but directory listing of `PROTECTED_HOME` and inspection of that process's working-directory/executable/output links returned PermissionError. Its association with this paper is unknown.

Evidence statuses: **verified** means directly observed within the stated scope; **inferred** means an interpretation of observed metadata; **unknown** means not established; **contradicted** means observed evidence conflicts with a claim. None of the supplied historical scientific claims could be verified or contradicted. Hardware, code, environment, data, results, and readiness are assessed separately.

## Search scope, instructions, and preservation

The initial workspace inspection and `rg --files` returned no project files. `FILESYSTEM_ROOT/AGENTS.md`, `HOME_PARENT/AGENTS.md`, and `USER_ROOT/AGENTS.md` were absent; the initially empty workspace had no AGENTS.md. `MOUNT_ROOT/AGENTS.md` and `EXTERNAL_DISK_MOUNTPOINT/AGENTS.md` were absent. `PROTECTED_HOME/AGENTS.md` could not be read because directory access was denied. An AGENTS.md filename was observed in the unrelated `countmil` repository; no contents, source, or modifications were inspected there, so its subtree instructions did not apply to these workspace writes.

Discovery used one-level visible-entry metadata from `USER_ROOT`, followed by immediate-entry metadata in these nine directories: `hosung`, `CBO_online_opt`, `Ptab`, `countmil`, `Mac`, `mathematics_conjectures`, `fsconv`, `fsconv-dependence`, and `BSDE`. A depth-at-most-two `rg --files --hidden` filename search in those directories looked for qf/KRX/LOB/handoff markers and `KR*_t.xlsx`. Exact `lob_forecasting`, `configs/qf`, `experiments/qf`, `qf_v2`, `docs/qf`, and historical-example workbook paths were also checked. The example filename was a discovery clue, not an assumed asset universe or schema. These searches returned no matches. Deeper unrelated project trees and data contents were not searched.

`DATA_PATH_CANDIDATE`, `DATASETS_PATH_CANDIDATE`, `WORKSPACE_PATH_CANDIDATE`, `WORKSPACES_PATH_CANDIDATE`, `PROJECTS_PATH_CANDIDATE`, and `USER_ROOT/lob_forecasting` were absent. `SERVICE_ROOT` and `EXTERNAL_DISK_MOUNTPOINT` were empty; `MOUNT_ROOT` contained `external_hdd`; `OPTIONAL_SOFTWARE_ROOT` contained `containerd`, which was not entered. Mount metadata did not show an external filesystem mounted at `EXTERNAL_DISK_MOUNTPOINT`. `PROTECTED_HOME` was considered only after the running process owner supplied the clue. Its directory and process-link PermissionErrors were not bypassed. No other users' directories, credentials, shell histories, or unrelated data contents were scanned.

No pull, fetch, clone, checkout, worktree operation, merge, rebase, stash, reset, clean, submodule operation, LFS download, installation, data transfer, training, cache rebuild, checkpoint load, or prediction regeneration was performed. No process was stopped or attached to. Git inspection used local configuration/ref queries with `GIT_OPTIONAL_LOCKS=0`; raw remote URLs were parsed in memory and only credential-free identities were emitted. Process arguments and environments were not read. New audit outputs are the only intentional filesystem writes. No operation requiring `GIT_LFS_SKIP_SMUDGE=1` was performed.

## Hardware and current activity

| Finding | Observation | Evidence / method | Status |
| --- | --- | --- | --- |
| GPUs | 4 × NVIDIA GeForce RTX 3090, 24,576 MiB each | `nvidia-smi --query-gpu`, approximately 14:28 UTC | verified |
| Driver | 515.57 on all GPUs | same query | verified |
| GPU activity | 0% GPU and memory utilization; 0 MiB reported used; 24,268 MiB reported free on each | same point-in-time query; remaining capacity should not be inferred by subtraction | verified |
| GPU temperatures | indices 0–3: 29, 28, 31, 28 °C | same query | verified |
| GPU compute jobs | query returned a header and no process rows | `nvidia-smi --query-compute-apps` | verified at snapshot |
| CPU | 2 × Intel Xeon Gold 5317, 24 physical cores / 48 logical CPUs total | `lscpu` | verified |
| RAM | 270,084,370,432 bytes total; 259,920,805,888 bytes available | `free -b` snapshot | verified |
| Disk | root ext4, total 982,240,026,624 bytes; 497,442,263,040 bytes available to this user (about 463.3 GiB) | `statvfs`, 14:30:34 UTC; workspace uses this filesystem | verified |
| OS | Ubuntu 22.04 Jammy | selected `SYSTEM_OS_RELEASE` fields | verified |
| Availability | GPUs were idle when queried; no reservation was made | snapshot interpretation | inferred, not a scheduling guarantee |

Scientific/interpreter process metadata was filtered from `ps` without arguments. Existing processes identified were:

| PID | Owner | Process / relation | Working directory and outputs | Status |
| --- | --- | --- | --- | --- |
| 452183 | PROTECTED_PROCESS_OWNER | Jupyter notebook, parent PID 1 | cwd/executable/stdout/stderr symlink access denied | existence verified; paper association unknown |
| 1357281 | AUDIT_USER | Jupyter notebook, parent PID 1 | cwd `USER_ROOT`; interpreter in `anaconda3/envs/mpnn`; stdout/stderr `USER_ROOT/nohup.out` | verified metadata; log contents not read |
| 1510530, 1635170 | AUDIT_USER | Python children of PID 1357281 | cwd `USER_ROOT/Mac/lr_beta_predictor`; same mpnn interpreter; outputs are pipes | verified metadata; no KRX attribution |

Each of these showed 0.0% CPU at the process snapshot. Notebook kernels can retain state; these observations do not prove that all CPU work on the server is absent. Audit subprocesses were excluded from the saved job inventory.

## Repository and pre-synchronization state

**Selected repository: none. Verified project remote: none. Original KRX branch, full CODE SHA, dirty state, worktrees, tracked-data/LFS/history-size status, and unpublished scientific commits: unknown.** The historical commit `99e305a904eaebeaa6d9bedec827426c3de9a593` was not located or verified; it is not a selected baseline.

Six adjacent directories had `.git` markers. Their local remote identities and immediate paths did not identify them as this project. Local `for-each-ref` queries found no ref names containing qf, leadlag, lob, or krx. No remote was contacted. The following are rejected discovery candidates, not audited KRX checkouts:

| Directory under `USER_ROOT` | Discovery evidence | Assessment |
| --- | --- | --- |
| CBO_online_opt | remote project name CBO_online_opt; no matching project paths/refs | inferred unrelated; no scientific baseline selected |
| Ptab | remote project name Ptab; no matching paths/refs | inferred unrelated |
| countmil | remote project name countmil; no matching paths/refs | inferred unrelated |
| Mac | Overleaf remote; no matching paths/refs | inferred unrelated |
| mathematics_conjectures | remote project name mathematics_conjectures; no matching paths/refs | inferred unrelated |
| BSDE | no local remote URL entries; no matching paths/refs | inferred unrelated |

Their HEADs, dirty changes, untracked content, worktrees, and scientific history were not inspected because no evidence established project relevance. `fsconv` was empty. `fsconv-dependence` contained other study directories and no checked KRX markers; its study contents were not inspected. No inference of cleanliness, completeness, or relative recency is made for any uninspected repository. No unique local KRX results or code could be identified; this does not rule out unique work in inaccessible or undiscovered locations.

## Environment readiness

The default shell resolved Python to `CONDA_ROOT/bin/python` and `python3` to the same environment. Python 3.9.13 and Git 2.34.1 were observed. `CUDA_TOOLKIT_ROOT/bin/nvcc --version` reported CUDA toolkit 11.7, V11.7.64. Driver version, toolkit version, and the PyTorch build's CUDA version are separate observations.

The following versions were read with isolated Python stdlib metadata queries (`-I -S -B`, using the interpreter's site-packages path). PyTorch build fields were parsed as AST literals from `torch/version.py`, without importing torch. These are bounded selected-package inventories, not environment dumps.

| Interpreter environment | Python | torch package / build CUDA | numpy | pandas | openpyxl | PyYAML |
| --- | --- | --- | --- | --- | --- | --- |
| `CONDA_ROOT` | 3.9.13 | no package metadata | 1.21.5 | 1.4.4 | 3.0.10 | 6.0 |
| `CONDA_ROOT/envs/mpnn` | 3.9.13 | 2.7.1+cu118 / 11.8 | 1.21.5 | 1.4.4 | 3.0.10 | 6.0 |
| `CONDA_ROOT/envs/tf_env` | 3.11.9 | no package metadata | 1.26.4 | 2.2.2 | no package metadata | 6.0.1 |
| `CONDA_ROOT/envs/bsde` | 3.9.18 | no package metadata | 1.22.3 | no package metadata | no package metadata | 6.0.1 |

The base and mpnn environments also had scipy 1.9.1, scikit-learn 1.0.2, and tqdm 4.64.1. No metadata for pyarrow, polars, lightning, pytorch-lightning, or einops was found in these four environments. The handoff contains the full selected-package results. No claim is made that the missing packages are required: project dependency/lock files were not found. Other projects' virtual environments were not audited.

**Installed:** metadata establishes an existing candidate PyTorch installation in mpnn. **Plausibly compatible with this project:** unknown without its source and requirements. **Demonstrably usable:** only isolated stdlib execution is verified; no third-party import, torch CPU operation, CUDA context, or project check was run. mpnn is used by existing notebook processes and was not activated, changed, or selected as the paper environment. Actual KRX launcher commands and the project interpreter are unknown.

## Source data, pipeline, and scientific semantics

`source_inventory.csv` separates supplied aggregate workbooks, derived caches, checkpoints, logs, predictions, metrics, project code/configs, and dependency manifests. Each category is absent in the initially empty workspace; its availability elsewhere is unresolved. The bounded filename searches did not locate recognized KRX aggregates. Unrelated projects' similarly named data/log/result directories are not evidence of KRX data. No workbook, cache, prediction file, tensor, or checkpoint was opened or hashed. No scientific manifest/checksum was found; representative scientific checksums are unavailable. Artifact checksum manifests cover only the new audit files.

Coverage is an exhaustive entry listing of the initial workspace, shallow discovery elsewhere, and no schema sample. Asset count, calendar coverage, session frequency, observations/targets per day, schema, aggregate provider behavior, and storage required for later data provisioning are unknown. Current free disk alone cannot establish data readiness. Git synchronization would not establish the presence of untracked data.

| Pipeline component or scientific claim | Finding | Status |
| --- | --- | --- |
| Launcher → config → source/cache loader | no project entrypoints located | unknown |
| Loader → model aliases → checkpoint selection → scoring | no source, configs, checkpoint metadata, or score code located | unknown |
| Feature sets; investor inclusion and lag | cannot inspect effective settings | unknown |
| Label formula, horizon, thresholds, class order | cannot inspect code or predictions | unknown |
| Preprocessing fit population and leakage boundaries | no fit metadata/source | unknown |
| Missingness handling and fills introduced by code | no code or aggregate inspection | unknown |
| Provider aggregation/filling and original observations | no provider documentation or source inspection | unknown |
| Session/day-boundary behavior and split dates | no loader/split artifacts | unknown |
| Legacy 70/10/20; validation versus development holdout | historical description only; no evaluated-support evidence | unknown |

Supplied KRX workbooks are aggregate inputs; a directory named `raw/` would not exclude them from inspection. Nonmissing aggregate values would not by themselves prove exchange observations; unchanged prices would not prove imputation. Raw exchange-event acquisition is out of scope. Previously examined test periods must be treated as development data, not untouched lockboxes; no new lockbox claim is made here.

## Executions, historical anchors, and metric checks

No project-specific execution was identified. `run_inventory.csv` has the required evidence/provenance columns and zero data rows; it does **not** assert zero runs on this machine. Execution status (config_only, started, completed, failed/interrupted, unknown) and evidence completeness are separate fields. There was insufficient evidence to insert even a config-only run. Local training versus copied directories, resumes, and independent reruns are all unresolved.

S, U0, U1, UM/UM_V2_DEPTH, UX/UX_BASELINE, UC, UA/UA_SYNC, GRU_U1, statistical/MLP/TCN baselines, lag-aware variants, and peer-input interventions are historical search clues only. Their alias mappings, effective implementations, seeds, run/config/code identity, data/split/support identity, checkpoint-selection criteria, artifact paths, and metrics have not been verified locally.

The supplied anchors—seed-42 UA log loss 0.886570, UC approximately 0.9039, UA−UC −0.017301; three-seed mean UA 0.886392 and GRU 0.907096—remain user-supplied claims with no local corroborating artifacts. Neither validation nor development-holdout attribution can be established. No recalculation or scientific comparison was performed.

The prioritized CPU UA/UC or GRU saved-prediction check was skipped because no relevant prediction files or scoring entrypoints were located. Row keys, class mapping, original support, logits/probabilities, normalization, log base, clipping, weighting, and seed aggregation are unknown. No original-support or common-intersection scores were computed, and seed probabilities were not averaged.

No local evidence establishes implemented legacy features or completed qf_v2, rolling confirmation, shared-query attention, or TLOB experiments. The proposed items remain unverified proposals, not completed experiments.

## Checks performed and skipped

Performed: ancestor instruction checks; empty-workspace and non-repository verification; bounded path/filename/local-ref discovery; sanitized local remote-identity inspection for relevance; read-only GPU/CPU/RAM/disk/OS snapshots; no-argument process inventory and accessible process-link metadata; isolated selected-package/version metadata queries; new exact and sanitized audit outputs; audit-file validation and checksums.

Skipped with reason: scientific Git-state/history/worktree/LFS inspection (no verified project checkout); remote reachability/publication (no verified project remote); workbook schema/checksum and source/cache identity checks (no relevant accessible files); pipeline/config/entrypoint inspection (no project source); imports/runtime readiness and prediction rescoring (no project requirements/entrypoints/predictions); data storage estimate (no verified manifest); A5000 commit ancestry, report comparison, and isolated comparison worktree (handoff missing). Forbidden training, installs, transfers, regeneration, and destructive Git actions were not attempted. No credentials or access policy were changed.

## Readiness and controlled next step

| Category | Readiness | Evidence status |
| --- | --- | --- |
| Hardware | four RTX 3090s and available storage verified | verified |
| GPU availability | idle at snapshot; not reserved | verified snapshot / inferred availability |
| Code | unresolved; no verified KRX checkout or CODE SHA | unknown |
| Environment | mpnn torch metadata present; project compatibility and runtime untested | verified installation metadata / unknown suitability |
| Data/cache | no KRX inputs identified; inaccessible location remains a possibility | unknown outside workspace |
| Artifacts/metrics | no attributable runs/predictions identified | unknown |
| Experiment readiness | not established; code, input/support identity, requirements, and runtime checks missing | unknown; experiments not authorized in this phase |
| Synchronization | pending_source_handoff | verified handoff absent |
| Audit publication | unavailable: verified project remote missing | verified blocker |

No audit branch, REPORT commit, fetch, or comparison worktree exists. The sanitized publication payload is prepared locally under `docs/qf_audit/20260930/rtx3090/`; it is not a Git commit. Its paths use root aliases and its process-owner labels use role aliases. The exact local report retains machine paths.

To resume, the coordinator needs either the accessible KRX project location or the A5000 handoff with a verified sanitized project remote, full inspected CODE SHA, dirty-state caveat, audit branch and full REPORT SHA, plus tracked-data/LFS/history-size findings. Code and report revisions must stay distinct. Before any acquisition, assess ordinary tracked blobs as well as LFS; use verified code-only or supported narrowly filtered no-checkout acquisition and set `GIT_LFS_SKIP_SMUDGE=1`. Compare both servers without declaring either authoritative, and preserve unknown/unavailable revisions as blockers. No remote URL was guessed and no authentication workaround was attempted.

## Saved outputs

Exact local outputs are `WORKSPACE_ROOT/qf_rtx3090_audit.md`, `WORKSPACE_ROOT/qf_rtx3090_handoff.json`, `WORKSPACE_ROOT/source_inventory.csv`, `WORKSPACE_ROOT/run_inventory.csv`, and `WORKSPACE_ROOT/audit_files.sha256`. Sanitized counterparts are in `AUDIT_PAYLOAD/`. The handoff records the machine-readable evidence, scope, actions, and unresolved questions. No scientific baseline was selected and no experiments were launched.
