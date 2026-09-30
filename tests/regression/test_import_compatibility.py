def test_legacy_and_canonical_identity():
 from src.data_loader.dataset import LOBDataset as old_d
 from lob_forecasting.data.dataset import LOBDataset as new_d
 from src.models.seq2seq import LOBTransformer as old_m
 from lob_forecasting.models.transformers import LOBTransformer as new_m
 assert old_d is new_d and old_m is new_m
