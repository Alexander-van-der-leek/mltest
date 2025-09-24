import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import pickle
from datetime import datetime

class CleanModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.models = []
        self.oof_predictions = None
        
    def load_data(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        
        self.feature_cols = [col for col in self.train_data.columns if col not in [
            'TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID', 'TX_FRAUD'
        ]]
        
        print(f"Loaded data: Train {self.train_data.shape}, Test {self.test_data.shape}")
        print(f"Features: {len(self.feature_cols)}")
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
        
    def train(self, n_splits=5):
        X = self.train_data[self.feature_cols]
        y = self.train_data['TX_FRAUD']
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        self.oof_predictions = np.zeros(len(X))
        
        params = self.get_model_params()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
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
                
            elif self.model_type == 'randomforest':
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                val_pred = model.predict_proba(X_val)[:, 1]
            
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            self.oof_predictions[val_idx] = val_pred
            self.models.append(model)
            
            print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
        
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
        test_preds = np.zeros(len(X_test))
        
        for model in self.models:
            if self.model_type == 'lightgbm':
                fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
            elif self.model_type == 'xgboost':
                dtest = xgb.DMatrix(X_test)
                fold_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
            elif self.model_type == 'randomforest':
                fold_pred = model.predict_proba(X_test)[:, 1]
            
            test_preds += fold_pred
        
        test_preds /= len(self.models)
        
        submission = pd.DataFrame({
            'TX_ID': self.test_data['TX_ID'],
            'TX_FRAUD': test_preds
        })
        
        return submission
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_cols': self.feature_cols,
                'model_type': self.model_type
            }, f)
        print(f"Model saved: {filepath}")

def load_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = CleanModel(data['model_type'])
    model.models = data['models']
    model.feature_cols = data['feature_cols']
    
    return model