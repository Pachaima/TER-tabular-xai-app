import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import argparse

# Ensure TALENT is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
TALENT_ROOT = os.path.join(PROJECT_ROOT, "TALENT")
if TALENT_ROOT not in sys.path:
    sys.path.append(TALENT_ROOT)

from TALENT.model.utils import get_deep_args, get_method

class TalentModel:
    # Mapping for user-friendly names to TALENT internal names
    MODEL_MAPPING = {
        'FT-Transformer': 'ftt',
        'ResNet': 'resnet',
        'SwitchTab': 'switchtab',
        'TabNet': 'tabnet',
        'TabR': 'tabr',
        'TabPFN': 'tabpfn',
        # Fallbacks or internal names
        'ftt': 'ftt',
        'resnet': 'resnet',
        'switchtab': 'switchtab',
        'tabnet': 'tabnet',
        'tabr': 'tabr',
        'tabpfn': 'tabpfn'
    }

    def __init__(self, model_type, data_dir="data/current_dataset", random_state=42):
        # Normalize model type
        self.model_type = self.MODEL_MAPPING.get(model_type, model_type.lower())
        self.data_dir = data_dir
        self.random_state = random_state
        self.output_dir = os.path.join(data_dir, "models", self.model_type)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        with open(os.path.join(data_dir, "info.json"), 'r') as f:
            self.info = json.load(f)
            
        self.task_type = self.info['task_type']
        self.method = None
        self.args = None
        
    def _prepare_args(self):
        # Mock Argparse Namespace
        # We use get_deep_args but we need to override many things
        # Actually get_deep_args expects CLI arguments. 
        # We can simulate CLI args or just manually construct the namespace if we know what it needs.
        # But get_deep_args does a lot of loading configs.
        # Let's try to simulate CLI args by manipulating sys.argv temporarily OR just constructing the object manually if possible.
        # The easiest way with `get_deep_args` is to mock `sys.argv` before calling it, 
        # OR better: look at `get_deep_args` again. It uses `parser.parse_args()`.
        
        # We will mock sys.argv
        original_argv = sys.argv
        sys.argv = ['talent_wrapper.py', '--model_type', self.model_type, '--dataset', 'current_dataset']
        
        try:
            # We must trick TALENT into finding a config.
            # get_deep_args loads `TALENT/configs/deep_configs.json` for defaults.
            # Then it tries to load `TALENT/configs/dataset/{model}.json`.
            # This is problematic for dynamic datasets.
            
            # WORKAROUND: We will parse args, then manually inject our config.
            
            args, default_para, opt_space = get_deep_args()
            
            # Now override args
            args.dataset_path = self.data_dir
            args.model_path = os.path.dirname(self.output_dir) # Model path usually is root for results
            args.save_path = self.output_dir
            
            # CRITICAL: Force CPU at the torch level before model initialization
            args.gpu = "-1" 
            if 'training' not in args.config: args.config['training'] = {}
            args.config['training']['device'] = 'cpu'  # TabNet specific config
            
            # Ensure specific model configs also respect CPU (TabR/TabNet)
            if 'device' in args.config:
                args.config['device'] = 'cpu'
            
            args.seed = self.random_state
            
            # Disable tuning for speed in this demo, unless requested
            args.tune = False 
            
            # Update config with data specific info
             # n_bins is used for some numerical settings
            args.n_bins = 20 # Default
            args.config['training']['n_bins'] = args.n_bins
            
            # Reduce complexity for stable CPU execution
            args.max_epoch = 5 
            args.config['training']['max_epoch'] = 5
            
            self.args = args
            
        finally:
            sys.argv = original_argv

    def fit(self):
        if self.args is None:
            self._prepare_args()
            
        # Get Method Class
        try:
            MethodClass = get_method(self.model_type)
        except Exception as e:
            if self.model_type == 'tabpfn':
                 raise ValueError(f"TabPFN not found in TALENT installation. It might require a separate extension or newer version. Error: {e}")
            raise ValueError(f"Model '{self.model_type}' not supported or found in TALENT. Error: {e}")
            
        self.method = MethodClass(self.args, self.task_type == 'regression')
        
        # Load Data
        # method.fit expects (N, C, y) tuple + info dict
        
        N_train = np.load(os.path.join(self.data_dir, 'N_train.npy')).astype(np.float32)
        C_train = np.load(os.path.join(self.data_dir, 'C_train.npy')).astype(np.int32)
        y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'))
        
        N_val = np.load(os.path.join(self.data_dir, 'N_val.npy')).astype(np.float32)
        C_val = np.load(os.path.join(self.data_dir, 'C_val.npy')).astype(np.int32)
        y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'))

        # Prepare Data Dicts as expected by Method.fit
        # Expects: N={'train': ..., 'val': ...}, C=..., y=...
        
        data = (
            {'train': N_train, 'val': N_val},
            {'train': C_train, 'val': C_val},
            {'train': y_train, 'val': y_val}
        )
        
        # Prepare Info Dict
        # Needs: task_type, n_num_features, n_cat_features
        # And y_info?
        # `data_label_process` extracts y_info.
        # But `Method.fit` creates `Dataset` object first.
        # `Dataset` constructor computes features.
        
        # Actually, look at `base.py`:
        # self.D = Dataset(N, C, y, info)
        # So we just pass raw dicts.
        
        start_fit = True
        try:
            self.method.fit(data, self.info, train=start_fit, config=self.args.config)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                raise RuntimeError(f"Hardware Error: {self.model_type} failed to fall back to CPU. Ensure torch is installed correctly and CUDA is disabled. Error: {e}")
            else:
                raise RuntimeError(f"Training Error with {self.model_type}: {e}")
        except Exception as e:
             raise RuntimeError(f"Unexpected Training Error with {self.model_type}: {e}")
        
        return self

    def predict(self, X_df):
        """
        Predicts on new data (Pandas DataFrame or Numpy Array).
        """
        if self.method is None:
            raise Exception("Model not trained")
            
        # Handle Numpy Input (LIME/SHAP often pass numpy)
        if isinstance(X_df, np.ndarray):
            # We assume order matches feature_names
            feature_names = self.info.get('feature_names', [])
            if not feature_names:
                # If feature names missing, we can't reliably map unless we assume strict order?
                # Best effort: use numeric then categorical? No, impossible to know.
                # However, LIME usually respects feature_names order if provided.
                # X_df usually comes from X_test so it has feature_names order.
                # If feature_names is missing in info (old version), we might fail.
                # But we just added it.
                pass
            X_df = pd.DataFrame(X_df, columns=feature_names)
            
        # Preprocess X_df to N and C
        # We must use the info from self.info to know which cols are which and encoding.
        
        numeric_features = self.info['numeric_features']
        categorical_features = self.info['categorical_features']
        cat_mappings = self.info.get('cat_mappings', {})
        
        # Fill NA
        # Copy to avoid setting on view
        X_df = X_df.copy()
        
        X_df[numeric_features] = X_df[numeric_features].fillna(0) # Simple fill
        # For Cats
        for col in categorical_features:
            X_df[col] = X_df[col].fillna("Missing")
        
        # Encode Cats
        # We need to map strings to ints using the saved mappings.
        X_C_list = []
        for col in categorical_features:
            mapping = cat_mappings.get(col, {})
            # Reverse mapping {int: str} -> {str: int}
            str_to_int = {v: int(k) for k, v in mapping.items()}
            
            # Map values, default to -1 or something if unseen?
            # LabelEncoder usually strictly requires known labels.
            # We'll map unknown to 0 for safety but this is bad ML.
            # Ideally we used "unknown" token.
            
            def map_val(x):
                return str_to_int.get(str(x), 0) # Default to 0?
            
            X_C_list.append(X_df[col].apply(map_val).values)
            
        if X_C_list:
            X_C = np.stack(X_C_list, axis=1).astype(np.int32)
        else:
            X_C = None
            
        X_N = X_df[numeric_features].values.astype(np.float32)
        
        # TALENT predict needs (N, C, y). y can be None or dummy.
        # But `predict` calls `data_format(False...)`.
        
        # We need a dummy y
        dummy_y = np.zeros(len(X_df))
        
        # Method.predict(data, info, model_name)
        # data = (N, C, y) keys? No, predict expects just inputs usually or tuple?
        # base.py: N, C, y = data.
        # And it expects them to be passed to data_format.
        # If test, data_format expects raw arrays usually?
        # base.py line 120: N_test, C_test... = data_nan_process(N, C...)
        # It handles them.
        
        data_tuple = (
            {'test': X_N}, 
            {'test': X_C} if X_C is not None else None, 
            {'test': dummy_y}
        )
        
        # Model name: 'epoch-last' usually or 'best-val'
        model_name = 'best-val'
        
        # We need to suppress print output or capture it
        
        loss, metric, metric_name, predictions = self.method.predict(
            data_tuple, self.info, model_name
        )
        
        # predictions are Tensor or numpy?
        # base.py line 229: `test_logit` is returned.
        # If classification, it returns probabilities if we check `check_softmax` usage?
        # `metric` function calls `check_softmax` but `predict` returns `test_logit`.
        # So it returns LOGITS for classification usually.
        
        preds = predictions.cpu().numpy()
        
        if self.task_type == 'binclass' or self.task_type == 'multiclass':
             # Return classes
             return np.argmax(preds, axis=1)
        else:
             return preds.flatten()

    def predict_proba(self, X_df):
        """
        Returns probabilities for classification.
        """
        if self.method is None:
            raise Exception("Model not trained")

        # Handle Numpy Input
        if isinstance(X_df, np.ndarray):
            feature_names = self.info.get('feature_names', [])
            X_df = pd.DataFrame(X_df, columns=feature_names)
            
        # ... (Duplicate preprocessing logic - refactor?) ...
        numeric_features = self.info['numeric_features']
        categorical_features = self.info['categorical_features']
        cat_mappings = self.info.get('cat_mappings', {})
        
        X_df = X_df.copy()
        
        X_df[numeric_features] = X_df[numeric_features].fillna(0)
        for col in categorical_features:
            X_df[col] = X_df[col].fillna("Missing")
            
        X_C_list = []
        for col in categorical_features:
            mapping = cat_mappings.get(col, {})
            str_to_int = {v: int(k) for k, v in mapping.items()}
            X_C_list.append(X_df[col].apply(lambda x: str_to_int.get(str(x), 0)).values)
            
        if X_C_list:
            X_C = np.stack(X_C_list, axis=1).astype(np.int32)
        else:
            X_C = None
        X_N = X_df[numeric_features].values.astype(np.float32)
        
        dummy_y = np.zeros(len(X_df))
        data_tuple = (
            {'test': X_N}, 
            {'test': X_C} if X_C is not None else None, 
            {'test': dummy_y}
        )
        
        loss, metric, metric_name, predictions = self.method.predict(
            data_tuple, self.info, 'best-val'
        )
        
        # Convert logits to proba
        preds = predictions.cpu().numpy()
        
        # Softmax
        # If binclass, we might get 2 cols or 1? TALENT usually 2 cols for binclass
        # Check base.py check_softmax: exps / sum.
        
        def softmax(x):
            try:
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)
            except:
                return x # Fallback
            
        return softmax(preds)

