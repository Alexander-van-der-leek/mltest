import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

class CleanFeatureEngine:
    def __init__(self):
        self.encoders = {}
        self.global_stats = {}
        
    def load_data(self):
        self.customers = pd.read_csv('data/customers.csv')
        self.terminals = pd.read_csv('data/terminals.csv')
        self.merchants = pd.read_csv('data/merchants.csv')
        self.train_tx = pd.read_csv('data/transactions_train.csv')
        self.test_tx = pd.read_csv('data/transactions_test.csv')
        
        self.train_tx['TX_TS'] = pd.to_datetime(self.train_tx['TX_TS'])
        self.test_tx['TX_TS'] = pd.to_datetime(self.test_tx['TX_TS'])
        
        self.global_stats = {
            'fraud_rate': self.train_tx['TX_FRAUD'].mean(),
            'amount_mean': self.train_tx['TX_AMOUNT'].mean(),
            'amount_std': self.train_tx['TX_AMOUNT'].std(),
            'amount_percentiles': self.train_tx['TX_AMOUNT'].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        }
        
    def add_basic_features(self, df):
        df = df.copy()
        
        # Time features
        df['hour'] = df['TX_TS'].dt.hour
        df['day_of_week'] = df['TX_TS'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Amount features
        for p, val in self.global_stats['amount_percentiles'].items():
            df[f'amount_above_p{int(p*100)}'] = (df['TX_AMOUNT'] > val).astype(int)
        
        df['amount_zscore'] = (df['TX_AMOUNT'] - self.global_stats['amount_mean']) / self.global_stats['amount_std']
        df['amount_zscore_abs'] = np.abs(df['amount_zscore'])
        df['is_extreme_amount'] = (df['amount_zscore_abs'] > 3).astype(int)
        
        # Transaction components
        df['goods_ratio'] = df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
        df['has_cashback'] = (df['TRANSACTION_CASHBACK_AMOUNT'] > 0).astype(int)
        
        return df
    
    def add_spatial_features(self, df):
        df = df.merge(self.customers, on='CUSTOMER_ID', how='left')
        df = df.merge(self.terminals, on='TERMINAL_ID', how='left')
        
        df['distance'] = np.sqrt(
            (df['x_customer_id'] - df['x_terminal_id'])**2 + 
            (df['y_customer_id'] - df['y_terminal_id'])**2
        )
        
        df['is_close'] = (df['distance'] < 2).astype(int)
        df['is_far'] = (df['distance'] > 4).astype(int)
        
        return df
    
    def add_merchant_features(self, df):
        df = df.merge(self.merchants, on='MERCHANT_ID', how='left')
        
        # Encode categoricals properly
        cat_cols = ['CARD_BRAND', 'TRANSACTION_TYPE', 'CARDHOLDER_AUTH_METHOD', 'MCC_CODE', 'CARD_COUNTRY_CODE']
        
        for col in cat_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    all_values = pd.concat([
                        self.train_tx[col] if col in self.train_tx.columns else pd.Series(dtype='object'),
                        self.test_tx[col] if col in self.test_tx.columns else pd.Series(dtype='object')
                    ]).astype(str).fillna('unknown')
                    # Ensure 'unknown' is always in the classes
                    all_values = pd.concat([all_values, pd.Series(['unknown'])])
                    self.encoders[col].fit(all_values)
                
                # Handle unseen values by replacing with 'unknown'
                values = df[col].astype(str).fillna('unknown')
                unseen_mask = ~values.isin(self.encoders[col].classes_)
                values.loc[unseen_mask] = 'unknown'
                
                df[col + '_encoded'] = self.encoders[col].transform(values)
        
        # Boolean features - convert to int
        bool_cols = ['TAX_EXCEMPT_INDICATOR', 'IS_RECURRING_TRANSACTION']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({'Y': 1, 'N': 0, True: 1, False: 0}).fillna(0).astype(int)
        
        # High-risk indicators
        high_risk_mccs = ['5122', '7801', '5993', '4119', '7995']
        df['is_high_risk_mcc'] = df['MCC_CODE'].isin(high_risk_mccs).astype(int)
        
        return df
    
    def add_historical_features(self, train_df, test_df):
        train_df['is_train'] = True
        test_df['is_train'] = False
        
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined = combined.sort_values('TX_TS').reset_index(drop=True)
        
        # Initialize historical features including new velocity count features
        hist_features = [
            'customer_tx_count_hist', 'customer_fraud_count_hist', 'customer_fraud_rate_hist',
            'customer_avg_amount_hist', 'terminal_tx_count_hist', 'terminal_fraud_count_hist',
            'terminal_fraud_rate_hist', 'prev_tx_time_diff', 'customer_tx_last_hour',
            # NEW VELOCITY COUNT FEATURES
            'customer_tx_last_30min', 'customer_tx_last_15min', 'customer_tx_last_5min',
            'terminal_tx_last_30min', 'terminal_tx_last_15min',
            # NEW QUICK SUCCESSION CUMULATIVE COUNT
            'customer_quick_succession_count_hist'
        ]
        
        for feature in hist_features:
            combined[feature] = 0.0
        
        customer_hist = {}
        terminal_hist = {}
        
        global_fraud_rate = self.global_stats['fraud_rate']
        smoothing = 5
        
        for idx in combined.index:
            row = combined.iloc[idx]
            cust_id = row['CUSTOMER_ID']
            term_id = row['TERMINAL_ID']
            curr_time = row['TX_TS']
            curr_amount = row['TX_AMOUNT']
            is_train = row['is_train']
            
            if cust_id not in customer_hist:
                customer_hist[cust_id] = {
                    'tx_count': 0, 
                    'fraud_count': 0, 
                    'amounts': [], 
                    'timestamps': [],
                    'quick_succession_count': 0  # NEW: Track cumulative quick succession events
                }
            
            if term_id not in terminal_hist:
                terminal_hist[term_id] = {'tx_count': 0, 'fraud_count': 0, 'timestamps': []}
            
            # Calculate features using current history
            cust_h = customer_hist[cust_id]
            term_h = terminal_hist[term_id]
            
            combined.at[idx, 'customer_tx_count_hist'] = cust_h['tx_count']
            combined.at[idx, 'customer_fraud_count_hist'] = cust_h['fraud_count']
            combined.at[idx, 'customer_quick_succession_count_hist'] = cust_h['quick_succession_count']
            
            if cust_h['tx_count'] > 0:
                fraud_rate = (cust_h['fraud_count'] + smoothing * global_fraud_rate) / (cust_h['tx_count'] + smoothing)
                combined.at[idx, 'customer_fraud_rate_hist'] = fraud_rate
                
                if cust_h['amounts']:
                    combined.at[idx, 'customer_avg_amount_hist'] = np.mean(cust_h['amounts'])
                
                if cust_h['timestamps']:
                    last_time = max(cust_h['timestamps'])
                    time_diff = (curr_time - last_time).total_seconds()
                    combined.at[idx, 'prev_tx_time_diff'] = time_diff
                    
                    # ENHANCED VELOCITY FEATURES - Count transactions in multiple time windows
                    # Customer velocity counts
                    hour_ago = curr_time - timedelta(hours=1)
                    min_30_ago = curr_time - timedelta(minutes=30)
                    min_15_ago = curr_time - timedelta(minutes=15)
                    min_5_ago = curr_time - timedelta(minutes=5)
                    
                    combined.at[idx, 'customer_tx_last_hour'] = sum(1 for t in cust_h['timestamps'] if t >= hour_ago)
                    combined.at[idx, 'customer_tx_last_30min'] = sum(1 for t in cust_h['timestamps'] if t >= min_30_ago)
                    combined.at[idx, 'customer_tx_last_15min'] = sum(1 for t in cust_h['timestamps'] if t >= min_15_ago)
                    combined.at[idx, 'customer_tx_last_5min'] = sum(1 for t in cust_h['timestamps'] if t >= min_5_ago)
            else:
                combined.at[idx, 'customer_fraud_rate_hist'] = global_fraud_rate
                combined.at[idx, 'customer_avg_amount_hist'] = self.global_stats['amount_mean']
            
            combined.at[idx, 'terminal_tx_count_hist'] = term_h['tx_count']
            combined.at[idx, 'terminal_fraud_count_hist'] = term_h['fraud_count']
            
            if term_h['tx_count'] > 0:
                fraud_rate = (term_h['fraud_count'] + smoothing * global_fraud_rate) / (term_h['tx_count'] + smoothing)
                combined.at[idx, 'terminal_fraud_rate_hist'] = fraud_rate
                
                # Terminal velocity counts
                if term_h['timestamps']:
                    min_30_ago = curr_time - timedelta(minutes=30)
                    min_15_ago = curr_time - timedelta(minutes=15)
                    
                    combined.at[idx, 'terminal_tx_last_30min'] = sum(1 for t in term_h['timestamps'] if t >= min_30_ago)
                    combined.at[idx, 'terminal_tx_last_15min'] = sum(1 for t in term_h['timestamps'] if t >= min_15_ago)
            else:
                combined.at[idx, 'terminal_fraud_rate_hist'] = global_fraud_rate
            
            # Update histories only for training transactions
            if is_train and pd.notna(row.get('TX_FRAUD')):
                # Check if current transaction is quick succession (< 30 minutes from previous)
                if cust_h['timestamps']:
                    last_time = max(cust_h['timestamps'])
                    time_since_last = (curr_time - last_time).total_seconds()
                    if time_since_last < 300:  # 30 minutes = 1800 seconds
                        cust_h['quick_succession_count'] += 1
                
                cust_h['tx_count'] += 1
                cust_h['fraud_count'] += int(row['TX_FRAUD'])
                cust_h['amounts'].append(curr_amount)
                cust_h['timestamps'].append(curr_time)
                
                term_h['tx_count'] += 1
                term_h['fraud_count'] += int(row['TX_FRAUD'])
                term_h['timestamps'].append(curr_time)
        
        # Enhanced derived features using the new velocity counts
        combined['is_quick_repeat'] = (combined['prev_tx_time_diff'] < 300).astype(int)  # 30 minutes
        combined['multiple_tx_hour'] = (combined['customer_tx_last_hour'] > 0).astype(int)
        combined['high_risk_customer'] = (combined['customer_fraud_rate_hist'] > 0.05).astype(int)
        combined['high_risk_terminal'] = (combined['terminal_fraud_rate_hist'] > 0.1).astype(int)
        combined['amount_vs_hist_ratio'] = combined['TX_AMOUNT'] / (combined['customer_avg_amount_hist'] + 1)
        
        # NEW ENHANCED VELOCITY INDICATORS
        # Multiple transaction count categories
        combined['customer_burst_30min'] = (combined['customer_tx_last_30min'] >= 3).astype(int)  # 3+ in 30min
        combined['customer_burst_15min'] = (combined['customer_tx_last_15min'] >= 2).astype(int)  # 2+ in 15min
        combined['customer_burst_5min'] = (combined['customer_tx_last_5min'] >= 2).astype(int)   # 2+ in 5min
        combined['customer_high_velocity'] = (combined['customer_tx_last_hour'] >= 5).astype(int) # 5+ in 1hour
        
        # Terminal velocity indicators
        combined['terminal_burst_30min'] = (combined['terminal_tx_last_30min'] >= 5).astype(int)  # 5+ in 30min
        combined['terminal_burst_15min'] = (combined['terminal_tx_last_15min'] >= 3).astype(int)  # 3+ in 15min
        
        # Velocity ratio features (continuous values for models to learn thresholds)
        combined['customer_velocity_30min_ratio'] = combined['customer_tx_last_30min'] / 30.0  # transactions per minute
        combined['customer_velocity_15min_ratio'] = combined['customer_tx_last_15min'] / 15.0
        combined['customer_velocity_5min_ratio'] = combined['customer_tx_last_5min'] / 5.0
        
        # Cross-velocity features (customer + terminal patterns)
        combined['customer_terminal_burst'] = ((combined['customer_tx_last_30min'] >= 2) & 
                                             (combined['terminal_tx_last_30min'] >= 3)).astype(int)
        
        # ENHANCED QUICK SUCCESSION DERIVED FEATURES
        combined['has_quick_succession_history'] = (combined['customer_quick_succession_count_hist'] > 0).astype(int)
        combined['high_quick_succession_customer'] = (combined['customer_quick_succession_count_hist'] >= 3).astype(int)
        combined['quick_succession_rate'] = combined['customer_quick_succession_count_hist'] / (combined['customer_tx_count_hist'] + 1)
        
        train_processed = combined[combined['is_train']].drop('is_train', axis=1)
        test_processed = combined[~combined['is_train']].drop(['is_train', 'TX_FRAUD'], axis=1, errors='ignore')
        
        return train_processed, test_processed
    
    def create_features(self):
        print("Creating features...")
        
        train_basic = self.add_basic_features(self.train_tx)
        test_basic = self.add_basic_features(self.test_tx)
        
        train_spatial = self.add_spatial_features(train_basic)
        test_spatial = self.add_spatial_features(test_basic)
        
        train_merchant = self.add_merchant_features(train_spatial)
        test_merchant = self.add_merchant_features(test_spatial)
        
        train_final, test_final = self.add_historical_features(train_merchant, test_merchant)
        
        # Ensure all features are numeric
        exclude_cols = ['TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID']
        
        for col in train_final.columns:
            if col not in exclude_cols and col != 'TX_FRAUD':
                train_final[col] = pd.to_numeric(train_final[col], errors='coerce').fillna(0)
                if col in test_final.columns:
                    test_final[col] = pd.to_numeric(test_final[col], errors='coerce').fillna(0)
        
        # Drop original categorical columns
        drop_cols = [
            'CARD_BRAND', 'TRANSACTION_TYPE', 'CARDHOLDER_AUTH_METHOD', 'MCC_CODE',
            'CARD_EXPIRY_DATE', 'CARD_DATA', 'TRANSACTION_STATUS', 'FAILURE_CODE', 
            'FAILURE_REASON', 'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE',
            'ACQUIRER_ID', 'BUSINESS_TYPE', 'LEGAL_NAME', 'FOUNDATION_DATE',
            'OUTLET_TYPE', 'ACTIVE_FROM', 'TRADING_FROM', 'x_customer_id', 
            'y_customer_id', 'x_terminal_id', 'y_terminal_id'
        ]
        
        train_final = train_final.drop(columns=[col for col in drop_cols if col in train_final.columns])
        test_final = test_final.drop(columns=[col for col in drop_cols if col in test_final.columns])
        
        # Print feature summary including new velocity features
        print(f"Clean train: {train_final.shape}")
        print(f"Clean test: {test_final.shape}")
        print(f"Fraud rate: {train_final['TX_FRAUD'].mean():.4f}")
        
        # Show new velocity feature statistics
        velocity_features = [
            'customer_tx_last_30min', 'customer_tx_last_15min', 'customer_tx_last_5min',
            'customer_burst_30min', 'customer_burst_15min', 'customer_burst_5min',
            'customer_quick_succession_count_hist', 'has_quick_succession_history', 
            'high_quick_succession_customer'
        ]
        
        print("\nNew velocity feature statistics:")
        for feature in velocity_features:
            if feature in train_final.columns:
                if 'burst' in feature or 'has_quick_succession_history' in feature or 'high_quick_succession_customer' in feature:
                    # Binary features - show percentage
                    pct = train_final[feature].mean() * 100
                    fraud_rate = train_final[train_final[feature] == 1]['TX_FRAUD'].mean() if train_final[feature].sum() > 0 else 0
                    print(f"  {feature}: {pct:.2f}% of transactions, fraud rate: {fraud_rate:.3f}")
                elif 'quick_succession_count_hist' in feature:
                    # Quick succession count - show distribution
                    max_count = train_final[feature].max()
                    mean_count = train_final[feature].mean()
                    non_zero_pct = (train_final[feature] > 0).mean() * 100
                    print(f"  {feature}: max={max_count}, mean={mean_count:.2f}, {non_zero_pct:.1f}% have >0")
                else:
                    # Count features - show distribution
                    max_count = train_final[feature].max()
                    mean_count = train_final[feature].mean()
                    print(f"  {feature}: max={max_count}, mean={mean_count:.2f}")
        
        train_final.to_csv('train_clean.csv', index=False)
        test_final.to_csv('test_clean.csv', index=False)
        
        return 'train_clean.csv', 'test_clean.csv'

if __name__ == "__main__":
    engine = CleanFeatureEngine()
    engine.load_data()
    train_file, test_file = engine.create_features()