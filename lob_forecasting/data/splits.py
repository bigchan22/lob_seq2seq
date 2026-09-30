def chronological_split_indices(num_days, train_ratio, val_ratio):
    if not 0 < train_ratio < 1: raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1: raise ValueError("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1: raise ValueError("--train-ratio + --val-ratio must leave a non-empty test split.")
    train_end=int(num_days*train_ratio); val_end=int(num_days*(train_ratio+val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= num_days: raise ValueError("Invalid chronological split; need non-empty train, validation, and test splits.")
    return train_end,val_end
