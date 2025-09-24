import sys
from featureengineering import CleanFeatureEngine
from model import CleanModel
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
        print("python main.py knn [train] [test]         # Train K-Nearest Neighbors")
        print("python main.py visualize [model_file]     # Generate visualizations")
        print("python main.py visualize [model_file]     # Generate visualizations")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'features':
        print("Creating features...")
        engine = CleanFeatureEngine()
        engine.load_data()
        train_file, test_file = engine.create_features()
        print(f"Features created: {train_file}, {test_file}")
        
    elif command in ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'logistic', 'knn']:
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
    
    elif command == 'visualize':
        model_file = sys.argv[2] if len(sys.argv) > 2 else None
        print("Generating visualizations...")
        results = run_visualization(model_file)
        print("Visualizations complete. Check PNG files.")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()