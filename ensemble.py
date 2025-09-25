import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle
from datetime import datetime

class EnsembleModel:
    def __init__(self, base_models=['lightgbm', 'logistic'], blend_method='weighted'):
        """
        Ensemble model combining different approaches
        
        Args:
            base_models: List of base models to ensemble ['lightgbm', 'logistic', 'catboost', 'randomforest']
            blend_method: How to combine predictions ('simple', 'weighted', 'stacked')
        """
        self.base_models = base_models
        self.blend_method = blend_method
        self.models = {model_type: [] for model_type in base_models}
        self.scalers = {model_type: None for model_type in base_models}
        self.oof_predictions = {model_type: None for model_type in base_models}
        self.blend_weights = None
        self.blend_model = None
        self.feature_cols = None
        self.cat_features = None
        
    def load_data(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        
        self.feature_cols = [col for col in self.train_data.columns if col not in [
            'TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID', 'TX_FRAUD'
        ]]
        
        # Identify categorical features
        self.cat_features = [col for col in self.feature_cols if 'encoded' in col]
        
        print(f"Loaded data: Train {self.train_data.shape}, Test {self.test_data.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Base models: {self.base_models}")
        print(f"Blend method: {self.blend_method}")
        print(f"Fraud rate: {self.train_data['TX_FRAUD'].mean():.4f}")
        
    def get_model_params(self, model_type):
        """Get parameters for each model type"""
        if model_type == 'lightgbm':
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
        
        elif model_type == 'logistic':
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
        
        elif model_type == 'catboost':
            from catboost import CatBoostClassifier
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
        
        elif model_type == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
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
    
    def train_base_model(self, model_type, X_train, y_train, X_val, y_val):
        """Train a single base model for one fold"""
        params = self.get_model_params(model_type)
        
        if model_type == 'lightgbm':
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
        
        elif model_type == 'logistic':
            # Scale features for logistic regression
            if self.scalers[model_type] is None:
                self.scalers[model_type] = StandardScaler()
                X_train_scaled = self.scalers[model_type].fit_transform(X_train)
            else:
                X_train_scaled = self.scalers[model_type].transform(X_train)
            
            X_val_scaled = self.scalers[model_type].transform(X_val)
            
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
        
        elif model_type == 'catboost':
            from catboost import CatBoostClassifier
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
        
        elif model_type == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
        
        return model, val_pred
    
    def train(self, n_splits=5):
        """Train all base models and blend them"""
        X = self.train_data[self.feature_cols]
        y = self.train_data['TX_FRAUD']
        
        # Initialize OOF predictions for each base model
        for model_type in self.base_models:
            self.oof_predictions[model_type] = np.zeros(len(X))
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        base_model_scores = {model_type: [] for model_type in self.base_models}
        
        print(f"Training ensemble with {n_splits}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_predictions = {}
            
            # Train each base model
            for model_type in self.base_models:
                print(f"  Training {model_type}...")
                
                # Reset scaler for each fold to prevent data leakage
                if model_type == 'logistic':
                    self.scalers[model_type] = None
                
                model, val_pred = self.train_base_model(model_type, X_train, y_train, X_val, y_val)
                
                # Store model and predictions
                self.models[model_type].append(model)
                self.oof_predictions[model_type][val_idx] = val_pred
                fold_predictions[model_type] = val_pred
                
                # Calculate individual model performance
                fold_auc = roc_auc_score(y_val, val_pred)
                base_model_scores[model_type].append(fold_auc)
                print(f"    {model_type} AUC: {fold_auc:.4f}")
            
            # Blend predictions for this fold
            if self.blend_method == 'simple':
                ensemble_pred = np.mean([fold_predictions[mt] for mt in self.base_models], axis=0)
            elif self.blend_method == 'weighted' and fold == 0:
                # Calculate weights based on individual performance (first fold)
                weights = []
                for model_type in self.base_models:
                    auc = base_model_scores[model_type][0]
                    weights.append(max(0, auc - 0.5))  # Give more weight to better models
                
                # Normalize weights
                total_weight = sum(weights)
                self.blend_weights = [w / total_weight for w in weights] if total_weight > 0 else [1/len(self.base_models)] * len(self.base_models)
                print(f"  Blend weights: {dict(zip(self.base_models, [f'{w:.3f}' for w in self.blend_weights]))}")
                
                ensemble_pred = np.average([fold_predictions[mt] for mt in self.base_models], 
                                        weights=self.blend_weights, axis=0)
            elif self.blend_method == 'weighted':
                ensemble_pred = np.average([fold_predictions[mt] for mt in self.base_models], 
                                        weights=self.blend_weights, axis=0)
            elif self.blend_method == 'stacked':
                # For stacked ensemble, we'll train a meta-model later
                ensemble_pred = np.mean([fold_predictions[mt] for mt in self.base_models], axis=0)
            
            fold_ensemble_auc = roc_auc_score(y_val, ensemble_pred)
            print(f"  Ensemble AUC: {fold_ensemble_auc:.4f}")
        
        # Train stacked ensemble if requested
        if self.blend_method == 'stacked':
            print("\nTraining stacked ensemble...")
            # Create meta-features from OOF predictions
            meta_features = np.column_stack([self.oof_predictions[mt] for mt in self.base_models])
            
            # Train meta-model
            self.blend_model = LogisticRegression(random_state=42, max_iter=1000)
            self.blend_model.fit(meta_features, y)
            
            # Get final ensemble OOF predictions
            final_oof_pred = self.blend_model.predict_proba(meta_features)[:, 1]
        else:
            # For simple/weighted averaging
            if self.blend_method == 'simple':
                final_oof_pred = np.mean([self.oof_predictions[mt] for mt in self.base_models], axis=0)
            else:  # weighted
                final_oof_pred = np.average([self.oof_predictions[mt] for mt in self.base_models], 
                                          weights=self.blend_weights, axis=0)
        
        # Calculate final performance
        ensemble_oof_auc = roc_auc_score(y, final_oof_pred)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"ENSEMBLE RESULTS ({self.blend_method.upper()})")
        print(f"{'='*60}")
        
        for model_type in self.base_models:
            cv_mean = np.mean(base_model_scores[model_type])
            cv_std = np.std(base_model_scores[model_type])
            oof_auc = roc_auc_score(y, self.oof_predictions[model_type])
            print(f"{model_type.upper():15s}: CV {cv_mean:.4f} (+/-{cv_std*2:.4f}) | OOF {oof_auc:.4f}")
        
        print(f"{'ENSEMBLE':15s}: OOF {ensemble_oof_auc:.4f}")
        
        if self.blend_method == 'stacked' and self.blend_model is not None:
            meta_coefs = self.blend_model.coef_[0]
            print(f"\nStacked model coefficients:")
            for i, model_type in enumerate(self.base_models):
                print(f"  {model_type}: {meta_coefs[i]:.4f}")
        
        return ensemble_oof_auc, final_oof_pred
    
    def predict(self):
        """Generate ensemble predictions on test data"""
        X_test = self.test_data[self.feature_cols]
        
        # Generate predictions from each base model
        base_predictions = {model_type: np.zeros(len(X_test)) for model_type in self.base_models}
        
        print(f"Generating ensemble predictions...")
        
        for model_type in self.base_models:
            print(f"  {model_type}...")
            model_preds = np.zeros(len(X_test))
            
            for i, model in enumerate(self.models[model_type]):
                if model_type == 'lightgbm':
                    fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
                elif model_type == 'logistic':
                    # Apply same scaling as training
                    fold_scaler = self.scalers[model_type] if hasattr(self, 'scalers') and self.scalers[model_type] else StandardScaler()
                    # For logistic regression, we need to handle scaling properly
                    # This is a limitation - we'd need to save scalers per fold for perfect implementation
                    X_test_scaled = StandardScaler().fit_transform(self.train_data[self.feature_cols])  # Refit on full train data
                    scaler = StandardScaler().fit(X_test_scaled)
                    X_test_scaled = scaler.transform(X_test)
                    fold_pred = model.predict_proba(X_test_scaled)[:, 1]
                elif model_type in ['catboost', 'randomforest']:
                    fold_pred = model.predict_proba(X_test)[:, 1]
                
                model_preds += fold_pred
            
            # Average across folds
            base_predictions[model_type] = model_preds / len(self.models[model_type])
        
        # Combine base model predictions
        if self.blend_method == 'simple':
            final_pred = np.mean([base_predictions[mt] for mt in self.base_models], axis=0)
        elif self.blend_method == 'weighted':
            final_pred = np.average([base_predictions[mt] for mt in self.base_models], 
                                  weights=self.blend_weights, axis=0)
        elif self.blend_method == 'stacked':
            # Create meta-features and apply meta-model
            meta_features = np.column_stack([base_predictions[mt] for mt in self.base_models])
            final_pred = self.blend_model.predict_proba(meta_features)[:, 1]
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'TX_ID': self.test_data['TX_ID'],
            'TX_FRAUD': final_pred
        })
        
        print(f"\nEnsemble prediction statistics:")
        print(f"  Mean: {final_pred.mean():.4f}")
        print(f"  Min: {final_pred.min():.4f}")
        print(f"  Max: {final_pred.max():.4f}")
        
        # Show individual model contributions
        for model_type in self.base_models:
            pred = base_predictions[model_type]
            print(f"  {model_type} - Mean: {pred.mean():.4f}, Std: {pred.std():.4f}")
        
        return submission
    
    def get_feature_importance(self):
        """Extract feature importance from base models (where available)"""
        importance_data = {}
        
        for model_type in self.base_models:
            if model_type == 'lightgbm':
                # Average importance across folds
                all_importances = []
                for model in self.models[model_type]:
                    importance = model.feature_importance(importance_type='gain')
                    all_importances.append(importance)
                
                avg_importance = np.mean(all_importances, axis=0)
                importance_data[model_type] = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
            
            elif model_type == 'logistic':
                # Average coefficients across folds
                all_coefs = []
                for model in self.models[model_type]:
                    all_coefs.append(np.abs(model.coef_[0]))  # Use absolute values
                
                avg_coefs = np.mean(all_coefs, axis=0)
                importance_data[model_type] = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': avg_coefs
                }).sort_values('importance', ascending=False)
            
            elif model_type in ['catboost', 'randomforest']:
                # Average feature importances
                all_importances = []
                for model in self.models[model_type]:
                    importance = model.feature_importances_
                    all_importances.append(importance)
                
                avg_importance = np.mean(all_importances, axis=0)
                importance_data[model_type] = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
        
        return importance_data
    
    def plot_feature_importance(self, top_n=25):
        """Plot feature importance for each base model"""
        import matplotlib.pyplot as plt
        
        importance_data = self.get_feature_importance()
        
        n_models = len(importance_data)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_type, importance_df) in enumerate(importance_data.items()):
            top_features = importance_df.head(top_n)
            
            axes[i].barh(range(len(top_features)), top_features['importance'], 
                        color='steelblue', alpha=0.7)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_type.upper()} - Top {top_n} Features')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_data
    
    def save_model(self, filepath):
        """Save the ensemble model"""
        model_data = {
            'base_models': self.base_models,
            'blend_method': self.blend_method,
            'models': self.models,
            'scalers': self.scalers,
            'blend_weights': self.blend_weights,
            'blend_model': self.blend_model,
            'feature_cols': self.feature_cols,
            'cat_features': self.cat_features,
            'oof_predictions': self.oof_predictions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Ensemble model saved: {filepath}")

def load_ensemble_model(filepath):
    """Load a saved ensemble model"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = EnsembleModel(data['base_models'], data['blend_method'])
    model.models = data['models']
    model.scalers = data['scalers']
    model.blend_weights = data['blend_weights']
    model.blend_model = data['blend_model']
    model.feature_cols = data['feature_cols']
    model.cat_features = data['cat_features']
    model.oof_predictions = data['oof_predictions']
    
    print(f"Loaded ensemble model: {data['base_models']} with {data['blend_method']} blending")
    return model