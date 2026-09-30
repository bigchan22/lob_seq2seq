PROTOCOL_LEGACY = "legacy_v1"
PROTOCOL_QF = "qf_v2"
MASTER_GRID = tuple(f"{m//60:02d}:{m%60:02d}:00" for m in range(9*60, 15*60+21, 10))
