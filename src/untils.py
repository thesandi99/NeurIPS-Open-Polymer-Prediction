import random
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from mordred import Calculator, descriptors

import pandas as pd
import numpy as np
import os


def generate_mordred_descriptors(smiles_series, name):
    """Calculates and saves Mordred descriptors."""
    print(f"Generating Mordred descriptors for {len(smiles_series)} molecules ({name})...")
    
    # RDKit molecule conversion
    mols = [Chem.MolFromSmiles(s) for s in smiles_series]
    
    # Initialize Mordred calculator
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Use all available CPU cores for calculation
    # The .pandas() method shows the progress bar you are seeing
    df_desc = calc.pandas(mols, nproc=os.cpu_count())
    
    # Post-processing: ensure numeric, fill NaNs, remove constant columns
    df_desc = df_desc.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_desc = df_desc.loc[:, df_desc.nunique() > 1] # Important: remove zero-variance features
    
    print(f"Generated {df_desc.shape[1]} descriptors for {name}.")
    return df_desc


def set_seed(seed):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def select_features_in_fold(Xtr_df, ytr, k, corr_th):
    """Performs supervised feature selection using ONLY fold training data."""
    if Xtr_df.shape[1] <= k:
        return Xtr_df.columns.tolist()

    sel_f = SelectKBest(f_regression, k=min(k, Xtr_df.shape[1] - 1)).fit(Xtr_df, ytr)
    selected_cols_kbest = Xtr_df.columns[sel_f.get_support()]
    Xtr_df_selected = Xtr_df[selected_cols_kbest]

    corr = Xtr_df_selected.corr().abs()
    f_vals, _ = f_regression(Xtr_df_selected, ytr)
    strength = pd.Series(f_vals, index=Xtr_df_selected.columns).fillna(0.0)

    ordered_features = strength.sort_values(ascending=False).index
    kept_features = []
    for feature in ordered_features:
        if not kept_features:
            kept_features.append(feature)
            continue
        # Check correlation against already kept features
        if not (corr.loc[feature, kept_features] > corr_th).any():
            kept_features.append(feature)

    return kept_features

def get_transforms(y, target):
    """Applies target transformation for better model training."""
    if target == "FFV":
        eps = 1e-4
        y_clipped = np.clip(y, eps, 1 - eps)
        transform = lambda x: np.log(x / (1 - x)) # Logit transform
        inverse = lambda z: 1.0 / (1.0 + np.exp(-z))
        return transform(y_clipped), inverse
    if target == "Density":
        transform = lambda x: np.log(np.clip(x, 1e-4, None)) # Log transform
        inverse = lambda x: np.exp(x)
        return transform(y), inverse
    return y, lambda z: z # No transform for other targets


def run_xgb_pipeline(train_data, test_data, target, random_state):
    print(f"  Target: {target} | Model: XGBoost")
    set_seed(random_state)

    train_filtered = train_data[train_data[target].notna()].copy()
    y_raw = train_filtered[target].astype(np.float32).values
    
    # --- CORRECTED: Use shared feature columns ---
    feat_cols = shared_cols
    X_df = train_filtered[feat_cols]
    X_test_df = test_data[feat_cols]

    bins = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)

    test_preds = []
    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, bins), 1):
        print(f"    Fold {fold}/{CFG.N_SPLITS}")
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]

        imputer = SimpleImputer(strategy='median')
        Xtr_df = pd.DataFrame(imputer.fit_transform(Xtr_df), columns=Xtr_df.columns)
        Xva_df = pd.DataFrame(imputer.transform(Xva_df), columns=Xva_df.columns)
        X_test_fold_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
        
        selected_cols = select_features_in_fold(Xtr_df, ytr_raw, CFG.K_BEST_FEATURES, CFG.CORR_THRESHOLD)
        Xtr, Xva, X_test_fold = Xtr_df[selected_cols].values, Xva_df[selected_cols].values, X_test_fold_df[selected_cols].values
        
        model = xgb.XGBRegressor(
            random_state=random_state, objective='reg:absoluteerror', tree_method='hist',
            n_estimators=2500, learning_rate=0.015, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=150
        )
        model.fit(Xtr, ytr_raw, eval_set=[(Xva, yva_raw)], verbose=False)

        val_preds = model.predict(Xva)
        fold_test_preds = model.predict(X_test_fold)

        ytr_min, ytr_max = np.percentile(ytr_raw, [0.5, 99.5])
        test_preds.append(np.clip(fold_test_preds, ytr_min, ytr_max))
        
        fold_mae = mean_absolute_error(yva_raw, val_preds)
        fold_maes.append(fold_mae)
        print(f"      Fold {fold} MAE: {fold_mae:.5f}")

    print(f"    -> Average CV MAE for {target}: {np.mean(fold_maes):.5f} (+/- {np.std(fold_maes):.5f})")
    return np.mean(test_preds, axis=0)



def run_lgbm_pipeline(train_data, test_data, target, random_state):
    print(f"  Target: {target} | Model: LightGBM")
    set_seed(random_state)

    train_filtered = train_data[train_data[target].notna()].copy()
    y_raw = train_filtered[target].astype(np.float32).values
    
    # Use all feature columns
    feat_cols = [col for col in train_data.columns if col not in ['id', 'SMILES'] + CFG.TARGET_COLS]
    X_df = train_filtered[feat_cols]
    X_test_df = test_data[feat_cols]

    bins = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)

    test_preds = []
    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, bins), 1):
        print(f"    Fold {fold}/{CFG.N_SPLITS}")
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]

        imputer = SimpleImputer(strategy='median')
        Xtr = imputer.fit_transform(Xtr_df)
        Xva = imputer.transform(Xva_df)
        X_test_fold = imputer.transform(X_test_df)

        model = lgb.LGBMRegressor(
            random_state=random_state,
            objective='mae',
            metric='mae',
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(Xtr, ytr_raw,
                  eval_set=[(Xva, yva_raw)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        val_preds = model.predict(Xva)
        fold_test_preds = model.predict(X_test_fold)

        ytr_min, ytr_max = np.percentile(ytr_raw, [0.5, 99.5])
        test_preds.append(np.clip(fold_test_preds, ytr_min, ytr_max))
        
        fold_mae = mean_absolute_error(yva_raw, val_preds)
        fold_maes.append(fold_mae)
        print(f"      Fold {fold} MAE: {fold_mae:.5f}")

    print(f"    -> Average CV MAE for {target}: {np.mean(fold_maes):.5f} (+/- {np.std(fold_maes):.5f})")
    return np.mean(test_preds, axis=0)