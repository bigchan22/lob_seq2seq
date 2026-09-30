import torch
from lob_forecasting.models.transformers import AssetAwareLOBTransformer, LOBTransformer

def test_causal_mask_integrity():
    # 1. Setup Dummy Data (B, 38, N, F)
    B, T, N, F = 2, 38, 5, 10
    d_model = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LOBTransformer(num_assets=N, num_features=F, d_model=d_model).to(device)
    model.eval() # Fix dropout for consistent comparison

    # Create random features
    features = torch.randn(B, T, N, F).to(device)

    # 2. Baseline Pass
    with torch.no_grad():
        baseline_logits = model(features)

    # 3. Sabotaged Pass
    # We change features at index 20, 21, ..., 37
    # If the mask works, baseline_logits at index 0...19 should NOT change.
    sabotaged_features = features.clone()
    sabotaged_features[:, 20:, :, :] += 100.0 # Add massive "future" noise

    with torch.no_grad():
        sabotaged_logits = model(sabotaged_features)

    # 4. Compare Results
    # Check if the output at time step 19 (before the sabotage) is identical
    diff = torch.abs(baseline_logits[:, :20, :, :] - sabotaged_logits[:, :20, :, :]).max().item()

    print("--- Causal Mask Stress Test ---")
    print(f"Sabotage applied from time step 20 onwards.")
    print(f"Max difference in outputs for steps 0-19: {diff:.10f}")

    if diff < 1e-6:
        print("✅ PASS: The model at t_n cannot see features from t_{n+1}. Mask is watertight.")
    else:
        print("❌ FAIL: Future leakage detected! Check the generate_causal_mask implementation.")
    assert diff < 1e-6, "Future leakage detected in LOBTransformer causal mask."

    # 5. Check "Cheating" Visibility
    # Naturally, the logits AT and AFTER step 20 SHOULD be different
    diff_after = torch.abs(baseline_logits[:, 20:, :, :] - sabotaged_logits[:, 20:, :, :]).max().item()
    print(f"Difference in outputs for steps 20-37: {diff_after:.4f} (Should be > 0)")


def test_asset_aware_causal_mask_integrity():
    B, T, N, F = 2, 38, 5, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AssetAwareLOBTransformer(
        num_assets=N,
        num_features=F,
        d_model=64,
        nhead=8,
        temporal_layers=2,
        cross_asset_layers=1,
        dropout=0.0,
    ).to(device)
    model.eval()

    features = torch.randn(B, T, N, F).to(device)
    with torch.no_grad():
        baseline_logits = model(features)

    sabotaged_features = features.clone()
    sabotaged_features[:, 20:, :, :] += 100.0
    with torch.no_grad():
        sabotaged_logits = model(sabotaged_features)

    diff = torch.abs(baseline_logits[:, :20, :, :] - sabotaged_logits[:, :20, :, :]).max().item()
    diff_after = torch.abs(baseline_logits[:, 20:, :, :] - sabotaged_logits[:, 20:, :, :]).max().item()

    print("--- Asset-Aware Causal Mask Stress Test ---")
    print(f"Sabotage applied from time step 20 onwards.")
    print(f"Max difference in outputs for steps 0-19: {diff:.10f}")
    if diff < 1e-6:
        print("✅ PASS: The asset-aware model at t_n cannot see features from t_{n+1}.")
    else:
        print("❌ FAIL: Future leakage detected in the asset-aware model.")
    assert diff < 1e-6, "Future leakage detected in AssetAwareLOBTransformer causal mask."
    print(f"Difference in outputs for steps 20-37: {diff_after:.4f} (Should be > 0)")

if __name__ == "__main__":
    test_causal_mask_integrity()
    test_asset_aware_causal_mask_integrity()
