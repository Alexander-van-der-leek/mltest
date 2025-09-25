import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pickle
from datetime import datetime

class CleanModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.models = []
        self.oof_predictions = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        
        self.feature_cols = [col for col in self.train_data.columns if col not in [
            'TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID', 'TX_FRAUD'
        ]]
        
        # Identify categorical features for CatBoost
        self.cat_features = [col for col in self.feature_cols if 'encoded' in col]
        
        print(f"Loaded data: Train {self.train_data.shape}, Test {self.test_data.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Categorical features: {len(self.cat_features)}")
        print(f"Fraud rate: {self.train_data['TX_FRAUD'].mean():.4f}")
        
    def get_model_params(self):
        if self.model_type == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'max_depth': 6,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'random_state': 42,
                'verbose': -1,
                'is_unbalance': True
            }
        
        elif self.model_type == 'xgboost':
            fraud_count = self.train_data['TX_FRAUD'].sum()
            non_fraud_count = len(self.train_data) - fraud_count
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'min_child_weight': 10,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'learning_rate': 0.05,
                'scale_pos_weight': non_fraud_count / fraud_count,
                'random_state': 42,
                'tree_method': 'hist',
                'verbosity': 0
            }
        
        elif self.model_type == 'catboost':
            return {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.8,
                'random_strength': 1,
                'od_type': 'Iter',
                'od_wait': 100,
                'random_seed': 42,
                'logging_level': 'Silent',
                'auto_class_weights': 'Balanced'
            }
        
        elif self.model_type == 'randomforest':
            fraud_count = self.train_data['TX_FRAUD'].sum()
            non_fraud_count = len(self.train_data) - fraud_count
            return {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'max_features': 'sqrt',
                'class_weight': {0: 1, 1: non_fraud_count/fraud_count},
                'random_state': 42,
                'n_jobs': -1
            }
        
        elif self.model_type == 'logistic':
            fraud_count = self.train_data['TX_FRAUD'].sum()
            non_fraud_count = len(self.train_data) - fraud_count
            return {
                'penalty': 'l1',
                'solver': 'liblinear',
                'C': 0.1,
                'class_weight': {0: 1, 1: non_fraud_count/fraud_count},
                'random_state': 42,
                'max_iter': 1000
            }
        
        elif self.model_type == 'knn':
            return {
                'n_neighbors': 50,
                'weights': 'distance',
                'algorithm': 'ball_tree',
                'leaf_size': 30,
                'metric': 'minkowski',
                'p': 2,
                'n_jobs': -1
            }
        
        elif self.model_type == 'naivebayes':
            return {
                'var_smoothing': 1e-9  # Gaussian Naive Bayes smoothing parameter
            }
    
    def train(self, n_splits=10):
        X = self.train_data[self.feature_cols]
        y = self.train_data['TX_FRAUD']
        
        # Apply preprocessing for models that need it
        if self.model_type in ['logistic', 'knn', 'naivebayes']:
            print(f"Applying StandardScaler for {self.model_type}")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_cols, index=X.index)
            
            # Feature selection for KNN to reduce dimensionality
            if self.model_type == 'knn' and len(self.feature_cols) > 50:
                print(f"KNN: Selecting top 50 features from {len(self.feature_cols)} for better performance")
                self.feature_selector = SelectKBest(score_func=f_classif, k=50)
                X_selected = self.feature_selector.fit_transform(X, y)
                
                selected_mask = self.feature_selector.get_support()
                self.selected_features = [self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]]
                X = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
                print(f"KNN: Selected {len(self.selected_features)} features")
            else:
                self.selected_features = self.feature_cols
        else:
            self.selected_features = self.feature_cols
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        self.oof_predictions = np.zeros(len(X))
        params = self.get_model_params()
        
        print(f"Training {self.model_type} with {n_splits}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model based on type
            if self.model_type == 'lightgbm':
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            elif self.model_type == 'xgboost':
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=100,
                    verbose_eval=0
                )
                val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
            
            elif self.model_type == 'catboost':
                model = CatBoostClassifier(**params)
                model.fit(
                    X_train, 
                    y_train,
                    cat_features=self.cat_features if self.cat_features else None,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    verbose=False
                )
                val_pred = model.predict_proba(X_val)[:, 1]
            
            elif self.model_type == 'randomforest':
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                val_pred = model.predict_proba(X_val)[:, 1]
            
            elif self.model_type == 'logistic':
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                val_pred = model.predict_proba(X_val)[:, 1]
            
            elif self.model_type == 'naivebayes':
                model = GaussianNB(**params)
                model.fit(X_train, y_train)
                val_pred = model.predict_proba(X_val)[:, 1]
            
            elif self.model_type == 'knn':
                # Smart sampling for KNN to manage memory
                if len(X_train) > 100000:
                    print(f"  KNN: Subsampling from {len(X_train)} samples for efficiency")
                    
                    # Get fraud and normal indices
                    fraud_idx = X_train.index[y_train == 1]
                    normal_idx = X_train.index[y_train == 0]
                    
                    # Balanced sampling
                    n_fraud_sample = min(len(fraud_idx), 10000)
                    n_normal_sample = min(len(normal_idx), 40000)
                    
                    if n_fraud_sample > 0 and n_normal_sample > 0:
                        fraud_sample = np.random.choice(fraud_idx, n_fraud_sample, replace=False)
                        normal_sample = np.random.choice(normal_idx, n_normal_sample, replace=False)
                        sample_idx = np.concatenate([fraud_sample, normal_sample])
                        np.random.shuffle(sample_idx)
                        
                        X_train_sample = X_train.loc[sample_idx]
                        y_train_sample = y_train.loc[sample_idx]
                        print(f"  KNN: Using {len(sample_idx)} samples ({n_fraud_sample} fraud, {n_normal_sample} normal)")
                    else:
                        X_train_sample = X_train
                        y_train_sample = y_train
                else:
                    X_train_sample = X_train
                    y_train_sample = y_train
                
                model = KNeighborsClassifier(**params)
                model.fit(X_train_sample, y_train_sample)
                val_pred = model.predict_proba(X_val)[:, 1]
            
            # Calculate fold performance
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            self.oof_predictions[val_idx] = val_pred
            self.models.append(model)
            
            print(f"  Fold {fold + 1} AUC: {fold_auc:.4f}")
        
        # Calculate overall performance
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        oof_auc = roc_auc_score(y, self.oof_predictions)
        
        print(f"\n{self.model_type.upper()} Results:")
        print(f"CV AUC: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        print(f"OOF AUC: {oof_auc:.4f}")
        print(f"Gap: {abs(cv_mean - oof_auc):.4f}")
        
        return cv_mean, oof_auc
    
    def predict(self):
        X_test = self.test_data[self.feature_cols]
        
        # Apply same preprocessing as training
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=self.feature_cols, index=X_test.index)
            
            # Apply feature selection if used
            if self.feature_selector is not None:
                X_test_selected = self.feature_selector.transform(X_test)
                X_test = pd.DataFrame(X_test_selected, columns=self.selected_features, index=X_test.index)
        
        # Generate predictions from all folds
        test_preds = np.zeros(len(X_test))
        print(f"Generating predictions with {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            if self.model_type == 'lightgbm':
                fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
            elif self.model_type == 'xgboost':
                dtest = xgb.DMatrix(X_test)
                fold_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
            elif self.model_type in ['catboost', 'randomforest', 'logistic', 'naivebayes', 'knn']:
                fold_pred = model.predict_proba(X_test)[:, 1]
            
            test_preds += fold_pred
            if (i + 1) % 2 == 0:
                print(f"  Completed {i + 1}/{len(self.models)} models")
        
        # Average predictions across folds
        test_preds /= len(self.models)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'TX_ID': self.test_data['TX_ID'],
            'TX_FRAUD': test_preds
        })
        
        print(f"Prediction statistics:")
        print(f"  Mean: {test_preds.mean():.4f}")
        print(f"  Min: {test_preds.min():.4f}")
        print(f"  Max: {test_preds.max():.4f}")
        
        return submission
    
    def save_model(self, filepath):
        model_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'cat_features': self.cat_features,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'model_type': self.model_type,
            'oof_predictions': self.oof_predictions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved: {filepath}")

def load_model(filepath):
    """Load a saved model"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = CleanModel(data['model_type'])
    model.models = data['models']
    model.feature_cols = data['feature_cols']
    model.cat_features = data.get('cat_features', [])
    model.scaler = data.get('scaler', None)
    model.feature_selector = data.get('feature_selector', None)
    model.selected_features = data.get('selected_features', model.feature_cols)
    model.oof_predictions = data.get('oof_predictions', None)
    
    print(f"Loaded {data['model_type']} model with {len(model.models)} folds")
    return model