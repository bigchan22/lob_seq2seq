# RTX3090 Git recovery record

Updated 2026-10-01T01:41:45.877799+00:00. Continued the saved audit; no broad rediscovery. Previous outputs are preserved at `RECEIVER_ROOT/git_recovery_archive/20261001T014145Z`.

**Recovered:** published August code export, plus 27 supplied KRX aggregate workbooks and 27 existing CSV caches. All 128 exported source files and all 54 input identities match the received manifests. September code is not recovered. No 4090 access is required for this result.

## Latest resume: additional audit compared

At 2026-10-01T01:37:18.820449+00:00, Git access succeeded and one new report ref was advertised: `refs/heads/audit/qf-receiver-git-20261001-010238` at `2912890319b19a731e6e446b486ff7c8c6f3c3e3`. Its parent is the already verified export ROOT `200e5758de0f5cb545322f5b5ab522ad645f9209`. The diff adds exactly eight audit files and changes no scientific code. Only those metadata files (77,455 bytes) were retrieved; no datasets, saved predictions or checkpoints were transferred. No new portability commit or September branch was advertised.

The new report's code manifest is byte-identical to the export-root manifest; all 54 input identities match our existing source SHA/path/blob/size/SHA-256 records. It describes another RTX4090 session and explicitly leaves actual A5000 identity unresolved. Its claim of local backup predictions/checkpoints is source-reported evidence, not a copy on this receiver; no endpoint or accessibility is inferred.

A bounded identity query confirms **four RTX3090s** on this receiver (24,576 MiB each, driver 515.57). The external report raises a PyTorch 1.12.1 API limitation. One safe local CPU inspection establishes that our existing PyTorch **2.7.1+cu118 explicitly supports `torch.load(weights_only=...)`**; that specific source-environment blocker does not apply here. No checkpoint was loaded and GPU runtime remains untested. Previous syntax/import checks remain valid for the unchanged source.

Evidence: `RECEIVER_ROOT/git_recovery_cross_receiver_comparison.json`, `RECEIVER_ROOT/git_recovery_local_torch_api_check.json`, and `RECEIVER_ROOT/git_recovery_receiver_hardware_confirmation.json`.

## Verified Git and version identity

Verified established remote: `git@github.com:bigchan22/lob_seq2seq.git`. Bridge: `RECEIVER_ROOT/git_bridge/lob_seq2seq`. It remains shallow and partial (`blob:none`). Its configured default fetch names only the old audit branch; it never established the absence of other published branches.

The latest successful advertised heads/tags check was at 2026-10-01T01:37:18.820449+00:00; no tags or September branch were advertised:

| Advertised ref | Full SHA |
| --- | --- |
| `refs/heads/audit/qf-a5000-20260930-142604` | `0ebdb6782f2322e346dfb0c303762aab8635419c` |
| `refs/heads/audit/qf-receiver-git-20261001-010238` | `2912890319b19a731e6e446b486ff7c8c6f3c3e3` |
| `refs/heads/audit/qf-rtx3090-20260930-1506` | `119bd79600d2f53a10fb7be3d9ab39ec62f23678` |
| `refs/heads/main` | `073cd89b556224eeae22413c3c05188a947e990a` |
| `refs/heads/recovery/qf-august-code-20260930-220020` | `200e5758de0f5cb545322f5b5ab522ad645f9209` |
| `refs/heads/recovery/qf-git-handoff-20260930-220020` | `a323db8aea04fbb33b8e32e7644647106150899c` |

The two exact recovery refs were fetched with blob filtering, shallow depth 4, no tags/submodules and `GIT_LFS_SKIP_SMUDGE=1`. Only selected small code/report blobs were materialized. No unrestricted history/checkpoint acquisition occurred.

- **CODE / parentless export ROOT:** `200e5758de0f5cb545322f5b5ab522ad645f9209`. No parent fields in its commit object. Code worktree: `RECEIVER_ROOT/recovery_worktrees/qf_august_export_200e5758de0f`.
- **Source mapping:** August `feb10a130bb036617229d6c4c9bc4a96df3b9fe6`. All 128 declared source blob IDs, modes, sizes and SHA-256 hashes match the export, totaling 198,543 bytes. The unavailable original August commit tree was not independently inspected here; source attribution remains exporter-declared.
- **REPORT:** `a323db8aea04fbb33b8e32e7644647106150899c`. Its sole parent is the export ROOT and its diff adds seven compact audit records only. No later portability commit was observed.
- The complete export is 134 files / 256,150 bytes, including six recovery helper/metadata files. All 134 local hashes/modes were rechecked after readiness checks; worktree is clean.
- Existing legacy reference `RECEIVER_ROOT/recovery_worktrees/legacy_main_073cd89b5562` remains separate. Its old code is not substituted for the August export.
- September `99e305a904eaebeaa6d9bedec827426c3de9a593` remains unadvertised and was rejected by the earlier exact filtered fetch (`not our ref`). No repeated retry was made. This is current remote unavailability, not proof of loss from A5000 or another verified repository.

The source report declares 62 original dirty paths (47 reference symlink materializations and 15 mode differences), with no required dirty delta among the approved exported source contents. Unrelated dirty work was excluded.

## Handoff identity

The new published recovery report was received and verified. It describes its own hardware as **RTX4090**, explicitly contradicting its A5000 naming. Therefore an actual A5000 handoff is **not yet hardware-confirmed**. This does not invalidate verified Git bytes or require access to the old 4090. The optional question to the user is whether the actual A5000 has a separate handoff/ref, especially for September work.

Received records: `RECEIVER_ROOT/received_accessible_server_recovery/a323db8aea04fbb33b8e32e7644647106150899c/docs/qf_recovery/20260930/accessible_server`. Source execution/run inventories and rescoring summaries are metadata from that report; they are not new local RTX3090 executions or locally reproduced metrics.

## Recovered inputs and provenance

Input root: `RECEIVER_ROOT/recovered_inputs/krx_main_073cd89b5562`.

| Category | Count | Bytes | Exact local directory |
| --- | ---: | ---: | --- |
| Supplied aggregate XLSX workbooks | 27 | 161,779,048 | `RECEIVER_ROOT/recovered_inputs/krx_main_073cd89b5562/data/raw/KRXaggData` |
| Existing derived CSV caches | 27 | 204,319,112 | `RECEIVER_ROOT/recovered_inputs/krx_main_073cd89b5562/data/processed` |
| Total | 54 | 366,098,160 | 349.14 MiB |

Published input source: `main` at `073cd89b556224eeae22413c3c05188a947e990a`. These are ordinary Git blobs, not LFS pointers. Each file's original tracked path, blob ID, byte count, SHA-256 and destination is in `RECEIVER_ROOT/qf_rtx3090_input_manifest.json` and `.csv`. Manifest JSON SHA-256: `d0c16994cb4e1b86d69c4c1d1fc524ccfb17972e985bf40054ae9e4667746974`.

Every file was verified while originally copied into this new input directory. This resume reused those files: all 54 identities match the new export input manifest in path, published SHA, blob ID, bytes and SHA-256, and all 54 local sizes were rechecked. No files were overwritten, recopied, converted or rebuilt. The exporter also declares the input blobs identical at August source SHA; that original-tree association remains source-reported.

All 27 CSV headers have 45 columns and contain every column required by the recovered loader; configured asset count 27 matches the actual CSV count. This is exhaustive header coverage and zero data-row inspection in this readiness pass. Previous workbook metadata sampling covered one workbook's sheet name. Calendar coverage, provider timestamp/release semantics and provider-side filling are not established by these checks. Nonmissing values or unchanged prices do not establish exchange observations or imputation.

## Bounded environment and configuration checks

Existing interpreter: `RECEIVER_PYTHON_ROOT/envs/mpnn/bin/python`; Python 3.9.13. Safe CPU imports passed for NumPy 1.21.5, pandas 1.4.4, PyTorch 2.7.1+cu118, scikit-learn 1.0.2, openpyxl 3.0.10, PyYAML 6.0 and tqdm 4.64.1, plus six reviewed project modules. All 92 Python files pass syntax parsing with this interpreter. No dependency installation was required. Requirements and pyproject are unpinned; no lockfile was recovered. Source-reported environment versions differ and are retained in the JSON.

Eleven of twelve YAML files parse, including selected August model/protocol configs. `experiments/registry.yaml:13` fails parsing because of an unquoted question mark in a flow mapping. That file contains unscheduled qf_v2 placeholders; this does not demonstrate qf_v2 implementation/execution and does not invalidate the separately parsed August configs. No repair was made.

Imports were isolated with GPUs hidden and bytecode disabled after reviewing their import closure. No queue launcher, CLI main, dataset/model constructor, tensor builder, converter, checkpoint loader, scoring run, inference or training function was executed. CUDA/GPU usability and complete end-to-end runtime remain untested; successful imports establish a narrower result.

## Implemented August semantics, from code

The QF runner reads `configs/qf/models/matched_temporal_v1.yaml`, then calls `qf_training.load_data`, `cli.parallel.bake_tensors`, data transforms, model factory and training/scoring routines. Selected protocol is development-only, no investor features, one-step state representation. Implemented models include S/U0/U1/UM/UX/UC/UA and UM_V2_LITE/UM_V2_DEPTH. Later GRU_U1, UA_SYNC and September lag-aware experiments are not recovered here.

Labels use **±1 bp**, not sign-only direction: next same-day midpoint return greater than +0.0001 is Up (2), less than −0.0001 Down (0), otherwise Flat (1); nonpositive current midpoint defaults to Flat. The loader imposes 39 grid endpoints from 09:00 through 15:20 and 38 targets; these are implementation facts, not verified provider sampling semantics. Alignment uses forward fill then zero fill within each asset/day, with no backward fill in this recovered August path. Own features are deterministic transforms; UM_V2 scaling is fit on training days. Investor data are excluded in the selected QF path.

The common-date chronological split is floor(70%)/through floor(80%)/remainder. Checkpoints use validation NLL improvement with min_delta; pilot also saves best validation macro-F1. Calendar dates were not recomputed. Previously examined holdout is development data, not an untouched lockbox. Prediction probabilities are softmax outputs; metric wrapper converts to float64 and normalizes before fixed-class sklearn log loss. No local predictions were available for support checks or rescoring.

## Remaining portability and artifacts

The export lists 21 files with historical absolute paths. Confirmed examples: `qf_plan.py:5`, `discrimination_plan.py:4`, both worker launchers, and `discrimination_runner.py` input/pilot paths. They were preserved as exported. Pilot runner supports `QF_DATA_DIR`, but Stage-2's embedded data and pilot paths require a portability change. Relative config paths also require project working directory.

Prepared mapping only; no persistent environment changes or output directories were created:

- project cwd: `RECEIVER_ROOT/recovery_worktrees/qf_august_export_200e5758de0f`
- interpreter: `RECEIVER_PYTHON_ROOT/envs/mpnn/bin/python`
- `QF_DATA_DIR`: `RECEIVER_ROOT/recovered_inputs/krx_main_073cd89b5562/data/processed`
- future `QF_ARTIFACT_ROOT`: `RECEIVER_ROOT/recovered_runs/qf_august_pilot`
- future `QF_DISC_ROOT`: `RECEIVER_ROOT/recovered_runs/qf_august_stage2`

No prediction arrays or checkpoints were recovered. The source report inventories **11 local-only files, 6,256,259 bytes**, including two UA/UC prediction archives and `selected_configurations.yaml` (806 bytes). Their source aliases, relative paths, hashes and proposed exact destinations are in `RECEIVER_ROOT/qf_rtx3090_optional_artifact_copy_manifest.json`. Obtain these only from an accessible holder who confirms possession if exact selected Stage-2 replay/result verification is needed. No SSH endpoint or inaccessible-host dependency is assumed. Generic August code/input availability does not depend on these saved results. Old main checkpoints were deliberately excluded; their absence here does not establish that historical results are lost.

## Preservation, outputs and next action

Bridge HEAD remains `0ebdb6782f2322e346dfb0c303762aab8635419c`. Prior audit HEAD remains `119bd79600d2f53a10fb7be3d9ab39ec62f23678`. Audit, legacy and new code worktrees are clean. Jobs, environments and scientific inputs were not changed. No data was published. A sanitized audit-only snapshot is prepared on `audit/qf-rtx3090-recovery-20261001-014145`, based on the already published code-only export ROOT. Its final full report SHA and push verification are recorded separately in `RECEIVER_ROOT/qf_rtx3090_recovery_publication_receipt.json`. The prior audit SHA is not a commit of this updated recovery report.

Outputs:

- `RECEIVER_ROOT/qf_rtx3090_git_recovery.md`
- `RECEIVER_ROOT/qf_rtx3090_git_recovery.json`
- `RECEIVER_ROOT/qf_rtx3090_input_manifest.json` and `.csv`
- `RECEIVER_ROOT/qf_rtx3090_scientific_code_manifest.json`
- `RECEIVER_ROOT/git_recovery_august_readiness.json`
- `RECEIVER_ROOT/git_recovery_august_preservation.json`
- `RECEIVER_ROOT/qf_rtx3090_optional_artifact_copy_manifest.json`
- `RECEIVER_ROOT/qf_rtx3090_git_recovery.sha256`

**Next concrete setup action:** Prepare a separate portability change from the unchanged August export: parameterize launcher/plan interpreter and Stage-2 data/pilot paths, use the existing mpnn environment and recovered CSVs, and choose fresh output directories. Local torch.load supports weights_only; preserve that option. Obtain actual A5000 identity and any newer September ref or required saved result artifacts separately before choosing the paper baseline. No experiment launch in this recovery pass.

Recovery/setup stops here, without launching experiments. Actual A5000 inventory and the September version remain separately pending; published August code and matching inputs are already available.
