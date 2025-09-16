

class CFG:
    """Configuration class for hyperparameters and settings."""
    N_SPLITS = 5
    SEEDS = [42] # Using more seeds for robustness provides better generalization
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # batch
    EPOCHS = 160
    BATCH_SIZE = 256
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 40
    HIDDEN_DIM = 512
    N_BLOCKS = 2
    DROPOUT_RATE = 0.3
    
    # Feature Selection
    K_BEST_FEATURES = 450
    CORR_THRESHOLD = 0.98