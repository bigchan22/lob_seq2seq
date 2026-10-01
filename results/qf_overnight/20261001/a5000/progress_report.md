# qf-august-overnight-20261001: a5000 progress

Updated 2026-10-01T18:50:56.477794+00:00; elapsed 3.21 hours. Exact source commit `08fcbdac350a00b7ca587d0c95c803766c01e9bf`; protocol `bd1488a2d411295d7a2118c50ddcc0b158786123613af1676062e2270d1637d1`.
Queue states: `{"SUCCEEDED": 178}`. Completed distinct full training jobs on this owner: 127; aliases and smokes are not counted as training. Local terminal: True.

Currently running: `[]`.

The finite global plan has153 unconditional full fits and up to4 conditional fixed-LR fits (157 maximum), plus explicit selection/reuse/analysis jobs. This owner never claims the other owner's models. A job succeeds only after result/prediction validation. Checkpoints remain task-local; hashes/relative names accompany each run. All selected compressed probabilities, histories, masks and scores are transferred through this result branch. Every LR trial and failure is recorded in selected_configs.json; performance never gates further valid work.

Read metrics and per-asset/subgroup/calibration tables under runs/ and aggregate/. Incomplete shared comparisons are explicitly listed in aggregate/incomplete_comparisons.json, not zero-filled. Means are per-seed losses, not probability ensembles. Historical replay is separate. R1 cites exact main-fit reuse; R2/R3 use their own training-only transforms. P0–P3 and validation/development_holdout/rolling_evaluation remain separate. Paired bootstrap uses10000 whole-date moving-block draws, length5, seed20261001, row weights and fold boundaries.

Limitations: provider aggregation/release timing unknown; historical selection and retrospective holdout; only3 optimization seeds; adapted-model scope where applicable. These classification results do not establish executable trading profitability or causality. P3 uses future endpoint quality ex post and never masks peer input. UC historical description conflict is preserved in the frozen replay namespace.

This report does not wait for the peer's final report to describe local completion. Local publication receipts outside Git and peer_receipt.json identify advertised result SHAs; no file tries to embed its own final commit SHA.
