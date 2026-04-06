import os
import sys
import pandas as pd
from pathlib import Path

# Add repo root and core to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "core"))

from stacker import train_and_save
from sale_features import get_sale_X_y
from rent_features import get_rent_X_y

def train_sale_and_rent():
    proc_dir = ROOT / "data" / "processed"
    out_dir  = ROOT / "models"
    
    # 1. Train SALE model
    print("\n[train_models] Starting SALE model training...")
    X_train_sale = pd.read_csv(proc_dir / "sale_train.csv")
    X_test_sale  = pd.read_csv(proc_dir / "sale_test.csv")
    
    from sale_features import get_sale_X_y
    X_s_tr, y_s_tr = get_sale_X_y(X_train_sale)
    X_s_te, y_s_te = get_sale_X_y(X_test_sale)
    
    train_and_save(X_s_tr, y_s_tr, X_s_te, y_s_te, "sale", out_dir=str(out_dir), run_cv=False)
    
    # 2. Train RENT model
    print("\n[train_models] Starting RENT model training...")
    X_train_rent = pd.read_csv(proc_dir / "rent_train.csv")
    X_test_rent  = pd.read_csv(proc_dir / "rent_test.csv")
    
    from rent_features import get_rent_X_y
    X_r_tr, y_r_tr = get_rent_X_y(X_train_rent)
    X_r_te, y_r_te = get_rent_X_y(X_test_rent)
    
    train_and_save(X_r_tr, y_r_tr, X_r_te, y_r_te, "rent", out_dir=str(out_dir), run_cv=False)
    
    print("\n[train_models] All models trained and saved to", out_dir)

if __name__ == "__main__":
    train_sale_and_rent()
