from dataclasses import asdict, dataclass, field
@dataclass
class ExperimentConfig:
 experiment_id:str; protocol_version:str="legacy_v1"; model_family:str=""; dataset:dict=field(default_factory=dict); feature_set:str=""; horizon:str=""; label_definition:str=""; seed:int=0; split_identifier:str=""; checkpoint_policy:str="best_validation"; prediction_output_policy:str="retain"; artifact_root:str="artifacts"
 def to_dict(self): return asdict(self)
