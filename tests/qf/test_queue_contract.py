import pytest
from lob_forecasting.experiments.queue_gates import validate_gpu
from lob_forecasting.experiments.qf_plan import build_plan
def test_confirmatory_count(): assert build_plan()['confirmatory_gpu_jobs']==72
def test_gpu_rejection():
 with pytest.raises(ValueError):validate_gpu(2)
def test_primary_feature_set():assert all(x['feature_set']=='no_investor' for x in build_plan()['jobs'])
