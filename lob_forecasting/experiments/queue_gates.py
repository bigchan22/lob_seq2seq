from pathlib import Path
ALLOWED_GPUS={0,1}
def validate_gpu(gpu):
 if gpu not in ALLOWED_GPUS:raise ValueError("physical GPU must be 0 or 1")
 return gpu
def lockbox_allowed(root,clean,data_frozen,configs_frozen):return Path(root,"LOCKBOX_AUTHORIZED.txt").exists() and clean and data_frozen and configs_frozen
