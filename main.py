import sys
from featureengineering import CleanFeatureEngine
from model import CleanModel
from ensemble import EnsembleModel
from dataviz import run_visualization
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python main.py features                    # Create features")
        print("python main.py lightgbm [train] [test]    # Train LightGBM")
        print("python main.py xgboost [train] [test]     # Train XGBoost")
        print("python main.py catboost [train] [test]    # Train CatBoost")
        print("python main.py randomforest [train] [test]# Train RandomForest")
        print("python main.py logistic [train] [test]    # Train Logistic Regression")
        print("python main.py naivebayes [train] [test]  # Train Naive Bayes")
        print("python main.py knn [train] [test]         # Train K-Nearest Neighbors")
        print("python main.py ensemble [train] [test] [method] [models]  # Train Ensemble")
        print("  - method: simple, weighted, stacked (default: weighted)")
        print("  - models: comma-separated list (default: lightgbm,logistic)")
        print("  - example: python main.py ensemble train.csv test.csv weighted lightgbm,naivebayes")
        print("python main.py visualize [model_file]     # Generate visualizations")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'features':
        print("Creating features...")
        engine = CleanFeatureEngine()
        engine.load_data()
        train_file, test_file = engine.create_features()
        print(f"Features created: {train_file}, {test_file}")
        
    elif command in ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'logistic', 'naivebayes', 'knn']:
        if len(sys.argv) != 4:
            print(f"Usage: python main.py {command} <train_file> <test_file>")
            return
            
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        
        print(f"Training {command}...")
        model = CleanModel(command)
        model.load_data(train_file, test_file)
        
        cv_auc, oof_auc = model.train()
        
        if abs(cv_auc - oof_auc) < 0.05:
            print("Model validation looks good, generating predictions...")
            
            submission = model.predict()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f'{command}_submission_{timestamp}.csv'
            submission.to_csv(submission_file, index=False)
            
            model_file = f'{command}_model_{timestamp}.pkl'
            model.save_model(model_file)
            
            print(f"Submission: {submission_file}")
            print(f"Model: {model_file}")
            print(f"Prediction mean: {submission['TX_FRAUD'].mean():.4f}")
        else:
            print(f"Validation gap too large: {abs(cv_auc - oof_auc):.4f}")
            print("Check for data issues before generating predictions")
    
    elif command == 'ensemble':
        if len(sys.argv) < 4:
            print("Usage: python main.py ensemble <train_file> <test_file> [method] [models]")
            print("  - method: simple, weighted, stacked (default: weighted)")
            print("  - models: comma-separated list (default: lightgbm,logistic)")
            return
        
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        
        # Parse optional arguments
        blend_method = sys.argv[4] if len(sys.argv) > 4 else 'weighted'
        base_models = sys.argv[5].split(',') if len(sys.argv) > 5 else ['lightgbm', 'logistic']
        
        # Validate blend method
        if blend_method not in ['simple', 'weighted', 'stacked']:
            print(f"Invalid blend method: {blend_method}. Use: simple, weighted, or stacked")
            return
        
        # Validate base models
        valid_models = ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'logistic', 'naivebayes', 'knn']
        for model in base_models:
            if model not in valid_models:
                print(f"Invalid model: {model}. Valid models: {valid_models}")
                return
        
        print(f"Training ensemble with {base_models} using {blend_method} blending...")
        
        ensemble = EnsembleModel(base_models=base_models, blend_method=blend_method)
        ensemble.load_data(train_file, test_file)
        
        ensemble_auc, oof_predictions = ensemble.train()
        
        # Generate predictions
        print("\nGenerating ensemble predictions...")
        submission = ensemble.predict()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_str = '_'.join(base_models)
        submission_file = f'ensemble_{models_str}_{blend_method}_submission_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        
        model_file = f'ensemble_{models_str}_{blend_method}_model_{timestamp}.pkl'
        ensemble.save_model(model_file)
        
        print(f"Submission: {submission_file}")
        print(f"Model: {model_file}")
        print(f"Prediction mean: {submission['TX_FRAUD'].mean():.4f}")
        
        # Generate feature importance plots
        print("\nGenerating feature importance analysis...")
        try:
            importance_data = ensemble.plot_feature_importance()
            print("Feature importance plots saved as 'ensemble_feature_importance.png'")
        except Exception as e:
            print(f"Could not generate feature importance plots: {e}")
    
    elif command == 'visualize':
        model_file = sys.argv[2] if len(sys.argv) > 2 else None
        print("Generating visualizations...")
        results = run_visualization(model_file)
        print("Visualizations complete. Check PNG files.")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()