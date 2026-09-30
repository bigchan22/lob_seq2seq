import gzip,csv
from pathlib import Path
PREDICTION_COLUMNS=('date','timestamp','asset','chronological_fold','split','seed','model_id','protocol_id','horizon','representation','feature_set','true_class','predicted_class','probability_down','probability_flat','probability_up','validation_selected_checkpoint_epoch','original_current_price_flag','original_future_price_flag','stale_observation_flag','session_regime_flag')
def write_predictions(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'wt',newline='') as f:w=csv.DictWriter(f,fieldnames=PREDICTION_COLUMNS);w.writeheader();w.writerows(rows)
