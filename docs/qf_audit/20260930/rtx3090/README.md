# RTX3090 recovery audit snapshot

Sanitized metadata only. Exact paths remain in the local receiver reports.

Root aliases: RECEIVER_ROOT is the receiver workspace; RECEIVER_PYTHON_ROOT its existing conda installation; RECEIVER_HOME its home directory; HISTORICAL_HOME denotes an original launcher path, not a configured destination. Paths recorded under these aliases are not claims that remote files exist on this receiver.

Scientific CODE/root SHA: 200e5758de0f5cb545322f5b5ab522ad645f9209. Source mapping: feb10a130bb036617229d6c4c9bc4a96df3b9fe6 (exporter-declared original August commit). Published input SHA: 073cd89b556224eeae22413c3c05188a947e990a. All 54 existing workbook/cache identities match; inventory rows contain paths/hashes/sizes only. No source data rows, prediction arrays, checkpoints, environments, or credentials are included.

This audit branch starts from the already-published code-only export root. Its new diff consists exclusively of files in this audit directory. The commit hash of this snapshot is supplied separately in the receiver handoff/publication receipt, not embedded in its own report.

The receiver is confirmed as four RTX3090 GPUs. The separately inspected external audit 2912890319b19a731e6e446b486ff7c8c6f3c3e3 describes RTX4090 hardware and does not confirm an actual A5000 handoff. No new scientific/portability commit was found. CPU readiness checks do not establish GPU or complete experiment readiness. No experiments were launched.
