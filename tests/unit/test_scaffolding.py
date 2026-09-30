from pathlib import Path
import pytest
from lob_forecasting.data.baking import investor_raw_columns
from lob_forecasting.data.splits import chronological_split_indices
from lob_forecasting.experiments.config import ExperimentConfig
from lob_forecasting.experiments.manifest import manifest_hash
from lob_forecasting.experiments.run_directory import RunDirectory

def test_split_rounding(): assert chronological_split_indices(10,.7,.1)==(7,7+1)
def test_investor_columns(): assert len(investor_raw_columns())==16
def test_config_and_hash():
 d=ExperimentConfig("x").to_dict(); assert d["protocol_version"]=="legacy_v1"; assert manifest_hash(d)==manifest_hash(d)
def test_overwrite_protection(tmp_path):
 RunDirectory(tmp_path,"e","r")
 with pytest.raises(FileExistsError): RunDirectory(tmp_path,"e","r")
