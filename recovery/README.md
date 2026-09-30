# Recovered August KRX scientific code

This is a reviewed, parentless **code-only snapshot** of selected committed paths from
`feb10a130bb036617229d6c4c9bc4a96df3b9fe6`, the August development study.
It is not the missing September study. No training, inference, or cache rebuilding
was performed for recovery. Exact export and report commit hashes are provided in
the recovery handoff, outside these self-contained source-mapping records.

`code_manifest.csv` maps every copied source file to its original commit, path,
Git blob, mode, byte count and exported SHA-256. These files are byte-identical to
the selected source commit. Newly authored recovery metadata and the input-copy
helper are outside that original-source manifest. Datasets, predictions,
checkpoints, environments, bytecode, notebook backup files, papers and manuscripts
are excluded. No inherited history or Git alternates are present.

The original scientific checkout is dirty: reference symlinks have become regular
files and launcher executable modes changed. No content delta was found in the
approved source/config/test/dependency files. Only committed bytes and modes were
exported; unrelated dirty files were not copied. Existing checkouts and permissions
were not changed.

The source contains historical hardcoded paths, especially the Stage-2 launchers
and `discrimination_runner.py`. They are preserved for faithful recovery and listed
in `export_manifest.json`. Do not run the old launch/stop/package scripts merely
to inspect them. The pilot runner supports `QF_DATA_DIR`; Stage-2 path portability
and environment reproducibility still need review before any separately authorized
experiment. Requirements are unpinned. This snapshot does not authorize training.

## Obtain this code on the receiving server

First inventory and preserve that server's existing and dirty work. Use a fresh
destination and the exact verified branch/SHA from ACCESSIBLE_SERVER_HANDOFF.
The established remote is `git@github.com:bigchan22/lob_seq2seq.git`.

```sh
export GIT_LFS_SKIP_SMUDGE=1
git clone --filter=blob:none --no-checkout --single-branch \
  --branch recovery/qf-august-code-20260930-220020 \
  git@github.com:bigchan22/lob_seq2seq.git lob_qf_recovered
git -C lob_qf_recovered rev-parse HEAD
# Compare the full SHA with the independently supplied handoff before materializing.
git -C lob_qf_recovered checkout HEAD -- .
```

The remote's protocol-v2 `filter` capability was verified during recovery. Recheck
filter support and clone output on receipt. Do not fall back to an unfiltered
historical clone. A code-only root has no old data-bearing scientific ancestors.

## Restore the already-published aggregate inputs separately

Published main commit `073cd89b556224eeae22413c3c05188a947e990a` contains the actual
supplied aggregate workbooks and runtime CSV caches as **ordinary Git blobs**.
`input_manifest.csv` lists exactly 27 `data/raw/KRXaggData/KR*_t.xlsx` and 27
top-level `data/processed/KR*_t.csv` files: 366,098,160 bytes combined. All 54 input
blob IDs are equal at that published commit and the selected August source SHA.
Source workbook/provider observation and release semantics remain unknown; this
does not prevent restoring the existing bytes.

After verifying remote filtering, fetch only commit/tree metadata first:

```sh
git -C lob_qf_recovered fetch --filter=blob:none --depth=1 --no-tags \
  --no-recurse-submodules origin 073cd89b556224eeae22413c3c05188a947e990a
python3 lob_qf_recovered/recovery/materialize_inputs.py \
  --repository lob_qf_recovered --destination lob_qf_inputs_new
```

The helper writes only to a new destination, retrieves only the manifest-listed
blobs, and verifies Git blob IDs, SHA-256 and byte counts. It never converts or
rebuilds caches. In a filtered clone, individual `cat-file` requests may cause
Git's lazy retrieval of those specific authorized input blobs. It does not request
checkpoints or other data paths. It refuses an existing destination; failed partial
files remain available for inspection rather than being silently overwritten.

Existing published history also includes checkpoints and other data: **skip-smudge
and no-checkout alone do not prevent ordinary historical blob downloads**. The
helper needs no LFS pull; these 54 paths are ordinary blobs, not LFS pointers.

## Result family and artifact boundary

August code uses same-day 39-point 09:00–15:20 grids, 38 forecast origins, a
relative-return threshold of ±0.0001 (±1 bp), and Down/Flat/Up = 0/1/2. The main
QF study excludes investor features and selects checkpoints by validation NLL with
minimum improvement 0.0001. Full-period splits are train 2015-07-01–2016-11-23,
validation 2016-11-24–2017-02-03 and previously examined development holdout
2017-02-06–2017-06-30. These are not untouched lockbox periods.

The completed local evidence includes 27 S and six shared seed-42 pilot fits,
five seed-7 shared extensions, corrected UM_V2_DEPTH and UA/UC tuning/confirmation
runs. UA_SYNC, GRU_U1, UX_BASELINE, and trained lag-aware UA were not recovered in
this version. Missing September evidence is not proof it never existed elsewhere.

Saved predictions/checkpoints and selected-run artifacts are local recovery
references, not included in this code branch. The original source tree ignores
the QF runtime artifacts. The receiving server can restore code and inputs through
Git now; exact saved-result replay needs an owner-controlled copy of identified
local artifacts if required, without new data publication. No old-server export
or old audit receipt is a prerequisite for the code/input recovery above.
