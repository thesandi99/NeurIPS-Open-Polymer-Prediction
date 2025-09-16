
import pandas as pd
import numpy as np
import gc
import warnings
import os

from src.config import CFG
from src.untils import generate_mordred_descriptors, set_seed, run_xgb_pipeline, run_lgbm_pipeline

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


print("✅ Configuration and seeding function defined.")


print("Loading data...")
DATA_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025' 
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train_features_df = generate_mordred_descriptors(train_df['SMILES'], 'train')
test_features_df = generate_mordred_descriptors(test_df['SMILES'], 'test')

train_cols = set(train_features_df.columns)
test_cols = set(test_features_df.columns)
shared_cols = sorted(list(train_cols.intersection(test_cols)))

train_features_df = train_features_df[shared_cols]
test_features_df = test_features_df[shared_cols]


train_full_df = pd.concat([train_df, train_features_df], axis=1)
test_full_df = pd.concat([test_df, test_features_df], axis=1)

print(f"✅ Features generated. Train shape: {train_full_df.shape}, Test shape: {test_full_df.shape}")

final_preds = {}
ID_test = test_df['id']

MODEL_CHOICE = {
    'Tg': 'lgbm',
    'FFV': 'lgbm',
    'Tc': 'lgbm',
    'Density': 'lgbm',
    'Rg': 'lgbm',
}

# MODEL MAPPING 
MODEL_FUNCTIONS = {
    'xgb': run_xgb_pipeline,
    'lgbm': run_lgbm_pipeline,
}


print("\n=== Initiating Training ===")
for target in CFG.TARGET_COLS:
    print(f"\n[{target.upper()}]")
    
    model_name = MODEL_CHOICE[target] 
    model_func = MODEL_FUNCTIONS[model_name]
    
    seed_preds = []
    for seed in CFG.SEEDS:
        print(f"\n--- Training with seed: {seed} ---")
        preds = model_func(train_full_df, test_full_df, target, random_state=seed)
        seed_preds.append(preds)
        gc.collect()

    # Average predictions across different seeds
    final_preds[target] = np.mean(seed_preds, axis=0)

print("\n✅ Training complete.")

submission_df = pd.DataFrame({'id': ID_test})
for target in CFG.TARGET_COLS:
    # This ensures the correct predictions are assigned to the correct column
    submission_df[target] = final_preds[target]

submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")
print(submission_df.head())

