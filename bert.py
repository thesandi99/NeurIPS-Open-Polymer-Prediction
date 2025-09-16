# train.py - With ChemBERT Integration

import pandas as pd
import numpy as np
import gc
import warnings
import os
import random
import torch
### NEW ###
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from mordred import Calculator, descriptors


# --- Global Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
### NEW ###
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class CFG:
    """Configuration class for hyperparameters and settings."""
    N_SPLITS = 5
    SEEDS = [42] 
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # --- LightGBM/XGBoost Hyperparameters ---
    K_BEST_FEATURES = 450
    CORR_THRESHOLD = 0.98

    CHEMBERT_MODEL_PATH = "DeepChem/ChemBERTa-77M-MTR"
    CHEMBERT_MAX_LEN = 128
    CHEMBERT_EPOCHS = 160 # Increased epochs since we have early stopping
    CHEMBERT_BATCH_SIZE = 32 # Can often increase batch size slightly
    CHEMBERT_LR = 2e-5 # A slightly higher LR is common for fine-tuning
    CHEMBERT_WEIGHT_DECAY = 1e-6
    CHEMBERT_PATIENCE = 40
    CHEMBERT_WARMUP_RATIO = 0.1 # For the learning rate scheduler

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

print("✅ Configuration and seeding function defined.")

# --- 1. Load Data ---
print("Loading data...")
# Assume data is in a standard input directory
DATA_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025' # Change to your data path
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))


# --- 2. Feature Engineering Function ---
def generate_mordred_descriptors(smiles_series, name):
    """Calculates and saves Mordred descriptors."""
    print(f"Generating Mordred descriptors for {len(smiles_series)} molecules ({name})...")
    mols = [Chem.MolFromSmiles(s) for s in smiles_series]
    calc = Calculator(descriptors, ignore_3D=True)
    df_desc = calc.pandas(mols, nproc=os.cpu_count())
    df_desc = df_desc.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_desc = df_desc.loc[:, df_desc.nunique() > 1]
    print(f"Generated {df_desc.shape[1]} descriptors for {name}.")
    return df_desc

# --- 3. Generate and Save Features ---
try:
    # Use pre-computed features if they exist
    train_features_df = pd.read_csv("/kaggle/input/polymer-dataset-525/train_features.csv")
except FileNotFoundError:
    print("Pre-computed train features not found. Generating them now.")
    train_features_df = generate_mordred_descriptors(train_df['SMILES'], 'train')
    
test_features_df = generate_mordred_descriptors(test_df['SMILES'], 'test')

# Align columns after generation
train_cols = set(train_features_df.columns)
test_cols = set(test_features_df.columns)
shared_cols = sorted(list(train_cols.intersection(test_cols)))

train_features_df = train_features_df[shared_cols]
test_features_df = test_features_df[shared_cols]

train_full_df = pd.concat([train_df.reset_index(drop=True), train_features_df.reset_index(drop=True)], axis=1)
test_full_df = pd.concat([test_df.reset_index(drop=True), test_features_df.reset_index(drop=True)], axis=1)

print(f"✅ Features generated. Train shape: {train_full_df.shape}, Test shape: {test_full_df.shape}")


# 1. Custom Dataset
class PolymerDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_len):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return inputs, target
        return inputs

# 2. Custom Model

class ChemBERTRegressor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        # ### MODIFIED ###: A more powerful MLP head
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.GELU(), # A common activation function in transformers
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_output)
        return prediction

# ### MODIFIED ###: This is the fully corrected function
from sklearn.preprocessing import StandardScaler

def run_chembert_pipeline(train_data, test_data, target, random_state):
    print(f"  Target: {target} | Model: ChemBERT")
    set_seed(random_state)

    tokenizer = AutoTokenizer.from_pretrained(CFG.CHEMBERT_MODEL_PATH)
    
    train_filtered = train_data[train_data[target].notna()].copy()
    
    y_raw = train_filtered[target].values.reshape(-1, 1) # Reshape for scaler
    smiles = train_filtered['SMILES'].values
    bins = pd.qcut(train_filtered[target].values, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)
    
    test_smiles = test_data['SMILES'].values
    test_dataset = PolymerDataset(test_smiles, None, tokenizer, CFG.CHEMBERT_MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=CFG.CHEMBERT_BATCH_SIZE * 2, shuffle=False)
    
    fold_maes = []
    test_preds_all_folds = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(smiles, bins), 1):
        print(f"    Fold {fold}/{CFG.N_SPLITS}")
        
        tr_smiles, va_smiles = smiles[tr_idx], smiles[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]
        
        # ### MODIFIED ###: Fit scaler on train data ONLY
        scaler = StandardScaler()
        ytr_scaled = scaler.fit_transform(ytr_raw)
        yva_scaled = scaler.transform(yva_raw)
        
        train_dataset = PolymerDataset(tr_smiles, ytr_scaled.flatten(), tokenizer, CFG.CHEMBERT_MAX_LEN)
        val_dataset = PolymerDataset(va_smiles, yva_scaled.flatten(), tokenizer, CFG.CHEMBERT_MAX_LEN)
        train_loader = DataLoader(train_dataset, batch_size=CFG.CHEMBERT_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.CHEMBERT_BATCH_SIZE * 2, shuffle=False)

        model = ChemBERTRegressor(CFG.CHEMBERT_MODEL_PATH).to(DEVICE)
        # ### MODIFIED ###: Setup for Layer-wise Learning Rate Decay
        optimizer_parameters = []
        # Give the BERT encoder layers a smaller learning rate
        optimizer_parameters.append({
            'params': model.bert.parameters(),
            'lr': 1e-5 # Smaller LR for the base model
        })
        # Give the new regressor head a larger learning rate
        optimizer_parameters.append({
            'params': model.regressor.parameters(),
            'lr': CFG.CHEMBERT_LR # Original (larger) LR for the head
        })

        optimizer = AdamW(optimizer_parameters, weight_decay=CFG.CHEMBERT_WEIGHT_DECAY)
        criterion = nn.L1Loss()
        
        # ### MODIFIED ###: Add Learning Rate Scheduler
        num_training_steps = len(train_loader) * CFG.CHEMBERT_EPOCHS
        num_warmup_steps = int(num_training_steps * CFG.CHEMBERT_WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        best_mae = float('inf')
        patience_counter = 0
        
        for epoch in range(CFG.CHEMBERT_EPOCHS):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                input_ids = inputs['input_ids'].to(DEVICE)
                attention_mask = inputs['attention_mask'].to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                scheduler.step() # ### MODIFIED ###: Step the scheduler

            model.eval()
            val_preds_scaled = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, _ = batch
                    input_ids = inputs['input_ids'].to(DEVICE)
                    attention_mask = inputs['attention_mask'].to(DEVICE)
                    outputs = model(input_ids, attention_mask)
                    val_preds_scaled.extend(outputs.squeeze().cpu().numpy())
            
            # ### MODIFIED ###: Inverse transform predictions to calculate true MAE
            val_preds_original_scale = scaler.inverse_transform(np.array(val_preds_scaled).reshape(-1, 1))
            epoch_mae = mean_absolute_error(yva_raw, val_preds_original_scale)
            print(f"      Epoch {epoch+1}/{CFG.CHEMBERT_EPOCHS}, Val MAE: {epoch_mae:.5f}")

            if epoch_mae < best_mae:
                best_mae = epoch_mae
                torch.save(model.state_dict(), f"chembert_best_fold_{fold}.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CFG.CHEMBERT_PATIENCE:
                    print(f"      Early stopping at epoch {epoch+1}.")
                    break
        
        fold_maes.append(best_mae)
        print(f"      Fold {fold} Best MAE: {best_mae:.5f}")

        model.load_state_dict(torch.load(f"chembert_best_fold_{fold}.pth"))
        model.eval()
        fold_test_preds_scaled = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                fold_test_preds_scaled.extend(outputs.squeeze().cpu().numpy())
        
        # ### MODIFIED ###: Inverse transform final test predictions
        fold_test_preds_original_scale = scaler.inverse_transform(np.array(fold_test_preds_scaled).reshape(-1, 1))
        test_preds_all_folds.append(fold_test_preds_original_scale.flatten())
        
        del model, scaler
        gc.collect()
        torch.cuda.empty_cache()

    print(f"    -> Average CV MAE for {target}: {np.mean(fold_maes):.5f} (+/- {np.std(fold_maes):.5f})")
    return np.mean(test_preds_all_folds, axis=0)

# --- 5. Main Execution ---
final_preds = {}
ID_test = test_df['id']

# --- DEFINE WHICH MODEL TO RUN FOR EACH TARGET ---
### NEW ###: Changed 'Tg' to use 'chembert'
MODEL_CHOICE = {
    'Tg': 'chembert',
    'Tc': 'chembert',
    'FFV': 'chembert',
    'Density': 'chembert',
    'Rg': 'chembert',
}

# --- MODEL MAPPING ---
### NEW ###: Added 'chembert' to the function map
MODEL_FUNCTIONS = {
   # 'lgbm': run_lgbm_pipeline,
    'chembert': run_chembert_pipeline,
    # 'xgb': run_xgb_pipeline, # You can add back XGB if you define it
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

    final_preds[target] = np.mean(seed_preds, axis=0)

print("\n✅ Training complete.")

# --- 6. Create Submission File ---
submission_df = pd.DataFrame({'id': ID_test})
for target in CFG.TARGET_COLS:
    submission_df[target] = final_preds[target]

submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")
print(submission_df.head())