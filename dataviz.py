import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDataVisualizer:
    def __init__(self):
        self.fig_count = 0
        
    def load_data(self):
        self.customers = pd.read_csv('data/customers.csv')
        self.terminals = pd.read_csv('data/terminals.csv')
        self.merchants = pd.read_csv('data/merchants.csv')
        self.train_tx = pd.read_csv('data/transactions_train.csv')
        
        self.train_tx['TX_TS'] = pd.to_datetime(self.train_tx['TX_TS'])
        self.train_tx['hour'] = self.train_tx['TX_TS'].dt.hour
        self.train_tx['day_of_week'] = self.train_tx['TX_TS'].dt.dayofweek
        self.train_tx['month'] = self.train_tx['TX_TS'].dt.month
        
        print(f"Data loaded: {len(self.train_tx):,} transactions, {self.train_tx['TX_FRAUD'].mean():.3%} fraud rate")
        
    def create_figure(self, figsize=(15, 10)):
        self.fig_count += 1
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Fraud Analysis - Part {self.fig_count}', fontsize=16, y=0.95)
        return fig, axes.flatten()
    
    def temporal_analysis(self):
        fig, axes = self.create_figure()
        
        # Hourly patterns
        hourly_stats = self.train_tx.groupby('hour').agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': 'mean'
        }).round(4)
        hourly_stats.columns = ['total_tx', 'fraud_tx', 'fraud_rate', 'avg_amount']
        
        axes[0].plot(hourly_stats.index, hourly_stats['fraud_rate'], 'b-', linewidth=2)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Fraud Rate')
        axes[0].set_title('Fraud Rate by Hour')
        axes[0].grid(True, alpha=0.3)
        
        # Daily patterns
        daily_stats = self.train_tx.groupby('day_of_week').agg({
            'TX_FRAUD': ['count', 'mean'],
            'TX_AMOUNT': 'mean'
        }).round(4)
        daily_stats.columns = ['total_tx', 'fraud_rate', 'avg_amount']
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1].bar(range(7), daily_stats['fraud_rate'], color='skyblue')
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel('Fraud Rate')
        axes[1].set_title('Fraud Rate by Day of Week')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_names)
        
        # Monthly patterns
        monthly_stats = self.train_tx.groupby('month').agg({
            'TX_FRAUD': ['count', 'mean']
        }).round(4)
        monthly_stats.columns = ['total_tx', 'fraud_rate']
        
        axes[2].bar(monthly_stats.index, monthly_stats['fraud_rate'], color='lightcoral')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Fraud Rate')
        axes[2].set_title('Fraud Rate by Month')
        
        # Volume vs fraud overlay
        ax2_twin = axes[3].twinx()
        axes[3].bar(hourly_stats.index, hourly_stats['total_tx']/1000, alpha=0.6, color='lightgray', label='Volume (K)')
        ax2_twin.plot(hourly_stats.index, hourly_stats['fraud_rate'], 'r-', linewidth=2, label='Fraud Rate')
        axes[3].set_xlabel('Hour of Day')
        axes[3].set_ylabel('Transaction Volume (K)')
        ax2_twin.set_ylabel('Fraud Rate')
        axes[3].set_title('Transaction Volume vs Fraud Rate')
        axes[3].legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('fraud_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return hourly_stats, daily_stats, monthly_stats
    
    def amount_analysis(self):
        fig, axes = self.create_figure()
        
        fraud_amounts = self.train_tx[self.train_tx['TX_FRAUD'] == 1]['TX_AMOUNT']
        normal_amounts = self.train_tx[self.train_tx['TX_FRAUD'] == 0]['TX_AMOUNT']
        
        # Amount distributions
        bins = np.linspace(0, 500, 50)
        axes[0].hist(normal_amounts, bins=bins, alpha=0.7, label='Normal', density=True, color='blue')
        axes[0].hist(fraud_amounts, bins=bins, alpha=0.7, label='Fraud', density=True, color='red')
        axes[0].set_xlabel('Transaction Amount ($)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Amount Distribution: Normal vs Fraud')
        axes[0].legend()
        
        # Amount percentiles comparison
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        fraud_pct = [np.percentile(fraud_amounts, p) for p in percentiles]
        normal_pct = [np.percentile(normal_amounts, p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        axes[1].bar(x - width/2, normal_pct, width, label='Normal', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, fraud_pct, width, label='Fraud', color='red', alpha=0.7)
        axes[1].set_xlabel('Percentile')
        axes[1].set_ylabel('Amount ($)')
        axes[1].set_title('Amount Percentiles: Normal vs Fraud')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'P{p}' for p in percentiles])
        axes[1].legend()
        
        # Fraud rate by amount bins
        self.train_tx['amount_bin'] = pd.cut(self.train_tx['TX_AMOUNT'], bins=20)
        amount_fraud_rate = self.train_tx.groupby('amount_bin')['TX_FRAUD'].agg(['count', 'mean'])
        amount_fraud_rate = amount_fraud_rate[amount_fraud_rate['count'] >= 100]
        
        bin_centers = [interval.mid for interval in amount_fraud_rate.index]
        axes[2].plot(bin_centers, amount_fraud_rate['mean'], 'go-', linewidth=2)
        axes[2].set_xlabel('Transaction Amount ($)')
        axes[2].set_ylabel('Fraud Rate')
        axes[2].set_title('Fraud Rate by Amount Range')
        axes[2].grid(True, alpha=0.3)
        
        # Round amounts analysis
        self.train_tx['is_round'] = (self.train_tx['TX_AMOUNT'] % 10 == 0)
        round_analysis = self.train_tx.groupby('is_round')['TX_FRAUD'].agg(['count', 'mean'])
        
        axes[3].bar(['Non-Round', 'Round'], round_analysis['mean'], color=['lightblue', 'orange'])
        axes[3].set_ylabel('Fraud Rate')
        axes[3].set_title('Fraud Rate: Round vs Non-Round Amounts')
        
        plt.tight_layout()
        plt.savefig('fraud_amount_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fraud_amounts.describe(), normal_amounts.describe()
    
    def geographic_analysis(self):
        fig, axes = self.create_figure(figsize=(16, 10))
        
        # Merge with location data
        tx_with_locations = self.train_tx.merge(self.customers, on='CUSTOMER_ID', how='left')
        tx_with_locations = tx_with_locations.merge(self.terminals, on='TERMINAL_ID', how='left')
        
        # Calculate distance
        tx_with_locations['distance'] = np.sqrt(
            (tx_with_locations['x_customer_id'] - tx_with_locations['x_terminal_id'])**2 +
            (tx_with_locations['y_customer_id'] - tx_with_locations['y_terminal_id'])**2
        )
        
        # Customer locations heatmap
        customer_fraud = tx_with_locations.groupby(['x_customer_id', 'y_customer_id']).agg({
            'TX_FRAUD': ['count', 'sum']
        })
        customer_fraud.columns = ['tx_count', 'fraud_count']
        customer_fraud = customer_fraud[customer_fraud['tx_count'] >= 10]
        customer_fraud['fraud_rate'] = customer_fraud['fraud_count'] / customer_fraud['tx_count']
        
        scatter = axes[0].scatter(customer_fraud.index.get_level_values(0), 
                                customer_fraud.index.get_level_values(1),
                                c=customer_fraud['fraud_rate'], 
                                s=np.sqrt(customer_fraud['tx_count']), 
                                alpha=0.6, cmap='Reds')
        axes[0].set_xlabel('Customer X Coordinate')
        axes[0].set_ylabel('Customer Y Coordinate')
        axes[0].set_title('Customer Locations (Size=Volume, Color=Fraud Rate)')
        plt.colorbar(scatter, ax=axes[0])
        
        # Terminal locations
        terminal_fraud = tx_with_locations.groupby(['x_terminal_id', 'y_terminal_id']).agg({
            'TX_FRAUD': ['count', 'sum']
        })
        terminal_fraud.columns = ['tx_count', 'fraud_count']
        terminal_fraud = terminal_fraud[terminal_fraud['tx_count'] >= 50]
        terminal_fraud['fraud_rate'] = terminal_fraud['fraud_count'] / terminal_fraud['tx_count']
        
        scatter2 = axes[1].scatter(terminal_fraud.index.get_level_values(0),
                                 terminal_fraud.index.get_level_values(1),
                                 c=terminal_fraud['fraud_rate'],
                                 s=np.sqrt(terminal_fraud['tx_count']),
                                 alpha=0.6, cmap='Blues')
        axes[1].set_xlabel('Terminal X Coordinate')
        axes[1].set_ylabel('Terminal Y Coordinate')
        axes[1].set_title('Terminal Locations (Size=Volume, Color=Fraud Rate)')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Distance analysis
        distance_bins = [0, 1, 2, 3, 4, 5, 10, 100]
        tx_with_locations['distance_bin'] = pd.cut(tx_with_locations['distance'], bins=distance_bins)
        distance_fraud = tx_with_locations.groupby('distance_bin')['TX_FRAUD'].agg(['count', 'mean'])
        distance_fraud = distance_fraud[distance_fraud['count'] >= 100]
        
        bin_labels = [f'{distance_bins[i]}-{distance_bins[i+1]}' for i in range(len(distance_bins)-1)]
        valid_bins = distance_fraud.index
        valid_labels = [str(bin_val) for bin_val in valid_bins]
        
        axes[2].bar(range(len(valid_labels)), distance_fraud['mean'], color='green', alpha=0.7)
        axes[2].set_xlabel('Distance Range')
        axes[2].set_ylabel('Fraud Rate')
        axes[2].set_title('Fraud Rate by Customer-Terminal Distance')
        axes[2].set_xticks(range(len(valid_labels)))
        axes[2].set_xticklabels(valid_labels, rotation=45)
        
        # Distance distribution
        axes[3].hist(tx_with_locations[tx_with_locations['TX_FRAUD'] == 0]['distance'], 
                    bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        axes[3].hist(tx_with_locations[tx_with_locations['TX_FRAUD'] == 1]['distance'], 
                    bins=50, alpha=0.7, label='Fraud', density=True, color='red')
        axes[3].set_xlabel('Distance')
        axes[3].set_ylabel('Density')
        axes[3].set_title('Distance Distribution: Normal vs Fraud')
        axes[3].legend()
        axes[3].set_xlim(0, 20)
        
        plt.tight_layout()
        plt.savefig('fraud_geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return distance_fraud
    
    def velocity_analysis(self):
        fig, axes = self.create_figure()
        
        # Calculate velocity features
        tx_sorted = self.train_tx.sort_values(['CUSTOMER_ID', 'TX_TS'])
        tx_sorted['prev_tx_time'] = tx_sorted.groupby('CUSTOMER_ID')['TX_TS'].shift(1)
        tx_sorted['time_diff_minutes'] = (tx_sorted['TX_TS'] - tx_sorted['prev_tx_time']).dt.total_seconds() / 60
        tx_sorted = tx_sorted.dropna(subset=['time_diff_minutes'])
        
        # Quick repeats analysis
        time_bins = [0, 5, 15, 60, 360, 1440, 10080]  # 5min, 15min, 1hr, 6hr, 1day, 1week
        tx_sorted['time_bin'] = pd.cut(tx_sorted['time_diff_minutes'], bins=time_bins)
        velocity_fraud = tx_sorted.groupby('time_bin')['TX_FRAUD'].agg(['count', 'mean'])
        velocity_fraud = velocity_fraud[velocity_fraud['count'] >= 100]
        
        bin_labels = ['<5min', '5-15min', '15min-1hr', '1-6hr', '6hr-1day', '1-7day']
        valid_bins = velocity_fraud.index
        valid_labels = [bin_labels[i] for i, bin_val in enumerate(time_bins[:-1]) if pd.Interval(time_bins[i], time_bins[i+1]) in valid_bins]
        
        axes[0].bar(range(len(valid_labels)), velocity_fraud['mean'], color='purple', alpha=0.7)
        axes[0].set_xlabel('Time Since Last Transaction')
        axes[0].set_ylabel('Fraud Rate')
        axes[0].set_title('Fraud Rate by Transaction Velocity')
        axes[0].set_xticks(range(len(valid_labels)))
        axes[0].set_xticklabels(valid_labels, rotation=45)
        
        # Daily transaction count per customer
        daily_tx_count = self.train_tx.groupby(['CUSTOMER_ID', self.train_tx['TX_TS'].dt.date]).size()
        customer_daily_patterns = daily_tx_count.groupby('CUSTOMER_ID').agg(['mean', 'max', 'std']).fillna(0)
        customer_daily_patterns.columns = ['avg_daily_tx', 'max_daily_tx', 'std_daily_tx']
        
        # Merge back with fraud data
        customer_fraud_rate = self.train_tx.groupby('CUSTOMER_ID')['TX_FRAUD'].mean()
        customer_patterns = customer_daily_patterns.join(customer_fraud_rate)
        customer_patterns = customer_patterns.dropna()
        
        # High activity customers
        high_activity = customer_patterns[customer_patterns['max_daily_tx'] > 10]
        axes[1].scatter(high_activity['max_daily_tx'], high_activity['TX_FRAUD'], alpha=0.6)
        axes[1].set_xlabel('Max Daily Transactions')
        axes[1].set_ylabel('Customer Fraud Rate')
        axes[1].set_title('High Activity Customers: Volume vs Fraud Rate')
        
        # Time distribution analysis
        fraud_time_diff = tx_sorted[tx_sorted['TX_FRAUD'] == 1]['time_diff_minutes']
        normal_time_diff = tx_sorted[tx_sorted['TX_FRAUD'] == 0]['time_diff_minutes']
        
        bins = np.logspace(0, 4, 30)  # Log scale from 1 to 10000 minutes
        axes[2].hist(normal_time_diff, bins=bins, alpha=0.7, label='Normal', density=True, color='blue')
        axes[2].hist(fraud_time_diff, bins=bins, alpha=0.7, label='Fraud', density=True, color='red')
        axes[2].set_xscale('log')
        axes[2].set_xlabel('Time Since Last Transaction (minutes)')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Time Between Transactions Distribution')
        axes[2].legend()
        
        # Burst detection
        tx_sorted['is_quick_repeat'] = tx_sorted['time_diff_minutes'] < 30
        quick_repeat_analysis = tx_sorted.groupby('is_quick_repeat')['TX_FRAUD'].agg(['count', 'mean'])
        
        axes[3].bar(['Normal Timing', 'Quick Repeat (<30min)'], quick_repeat_analysis['mean'], 
                   color=['lightblue', 'red'])
        axes[3].set_ylabel('Fraud Rate')
        axes[3].set_title('Quick Repeat Transactions Fraud Rate')
        
        plt.tight_layout()
        plt.savefig('fraud_velocity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return velocity_fraud, quick_repeat_analysis
    
    def merchant_analysis(self):
        fig, axes = self.create_figure(figsize=(16, 12))
        
        # Merge with merchant data
        tx_with_merchants = self.train_tx.merge(self.merchants, on='MERCHANT_ID', how='left')
        
        # MCC analysis
        mcc_fraud = tx_with_merchants.groupby('MCC_CODE').agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': 'mean'
        })
        mcc_fraud.columns = ['tx_count', 'fraud_count', 'fraud_rate', 'avg_amount']
        mcc_fraud = mcc_fraud[mcc_fraud['tx_count'] >= 1000].sort_values('fraud_rate', ascending=False)
        
        top_risk_mccs = mcc_fraud.head(10)
        axes[0].barh(range(len(top_risk_mccs)), top_risk_mccs['fraud_rate'], color='red', alpha=0.7)
        axes[0].set_yticks(range(len(top_risk_mccs)))
        axes[0].set_yticklabels(top_risk_mccs.index)
        axes[0].set_xlabel('Fraud Rate')
        axes[0].set_title('Top 10 Highest Risk MCC Codes')
        
        # Business type analysis
        business_fraud = tx_with_merchants.groupby('BUSINESS_TYPE').agg({
            'TX_FRAUD': ['count', 'mean'],
            'TX_AMOUNT': 'mean'
        })
        business_fraud.columns = ['tx_count', 'fraud_rate', 'avg_amount']
        business_fraud = business_fraud[business_fraud['tx_count'] >= 5000].sort_values('fraud_rate', ascending=False)
        
        axes[1].bar(range(len(business_fraud)), business_fraud['fraud_rate'], color='orange', alpha=0.7)
        axes[1].set_xticks(range(len(business_fraud)))
        axes[1].set_xticklabels(business_fraud.index, rotation=45, ha='right')
        axes[1].set_ylabel('Fraud Rate')
        axes[1].set_title('Fraud Rate by Business Type')
        
        # Payment channel analysis
        if 'PAYMENT_PERCENTAGE_ECOM' in tx_with_merchants.columns:
            tx_with_merchants['ecom_heavy'] = tx_with_merchants['PAYMENT_PERCENTAGE_ECOM'] > 70
            tx_with_merchants['f2f_heavy'] = tx_with_merchants['PAYMENT_PERCENTAGE_FACE_TO_FACE'] > 70
            
            channel_analysis = tx_with_merchants.groupby(['ecom_heavy', 'f2f_heavy'])['TX_FRAUD'].agg(['count', 'mean'])
            channel_labels = ['Mixed', 'F2F Heavy', 'Ecom Heavy', 'Other']
            
            if len(channel_analysis) >= 2:
                axes[2].bar(range(len(channel_analysis)), channel_analysis['mean'], color='green', alpha=0.7)
                axes[2].set_xticks(range(len(channel_analysis)))
                axes[2].set_xticklabels([str(idx) for idx in channel_analysis.index], rotation=45)
                axes[2].set_ylabel('Fraud Rate')
                axes[2].set_title('Fraud Rate by Payment Channel Preference')
        
        # Merchant size vs fraud
        if 'ANNUAL_TURNOVER' in tx_with_merchants.columns:
            tx_with_merchants['turnover_bin'] = pd.qcut(tx_with_merchants['ANNUAL_TURNOVER'].fillna(0), 
                                                       q=5, duplicates='drop')
            turnover_fraud = tx_with_merchants.groupby('turnover_bin')['TX_FRAUD'].agg(['count', 'mean'])
            turnover_fraud = turnover_fraud[turnover_fraud['count'] >= 100]
            
            axes[3].plot(range(len(turnover_fraud)), turnover_fraud['mean'], 'bo-', linewidth=2)
            axes[3].set_xlabel('Merchant Size Quintile (by Annual Turnover)')
            axes[3].set_ylabel('Fraud Rate')
            axes[3].set_title('Fraud Rate by Merchant Size')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_merchant_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return mcc_fraud.head(20), business_fraud
    
    def business_impact_analysis(self):
        fig, axes = self.create_figure(figsize=(16, 10))
        
        # Financial impact
        total_fraud_amount = self.train_tx[self.train_tx['TX_FRAUD'] == 1]['TX_AMOUNT'].sum()
        total_amount = self.train_tx['TX_AMOUNT'].sum()
        fraud_percentage = total_fraud_amount / total_amount * 100
        
        # Monthly fraud losses
        monthly_losses = self.train_tx[self.train_tx['TX_FRAUD'] == 1].groupby('month')['TX_AMOUNT'].sum()
        axes[0].bar(monthly_losses.index, monthly_losses.values/1000, color='red', alpha=0.7)
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Fraud Losses (K$)')
        axes[0].set_title(f'Monthly Fraud Losses\nTotal: ${total_fraud_amount:,.0f} ({fraud_percentage:.2f}% of volume)')
        
        # Top fraud merchants by loss
        merchant_losses = self.train_tx[self.train_tx['TX_FRAUD'] == 1].groupby('MERCHANT_ID').agg({
            'TX_AMOUNT': ['sum', 'count'],
            'TX_FRAUD': 'sum'
        })
        merchant_losses.columns = ['total_loss', 'fraud_tx_count', 'fraud_count']
        merchant_losses = merchant_losses.sort_values('total_loss', ascending=False).head(15)
        
        axes[1].barh(range(len(merchant_losses)), merchant_losses['total_loss'], color='darkred', alpha=0.7)
        axes[1].set_yticks(range(len(merchant_losses)))
        axes[1].set_yticklabels([f'M_{str(idx)[:8]}' for idx in merchant_losses.index])
        axes[1].set_xlabel('Total Fraud Loss ($)')
        axes[1].set_title('Top 15 Merchants by Fraud Losses')
        
        # Customer risk distribution
        customer_fraud_stats = self.train_tx.groupby('CUSTOMER_ID').agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': 'sum'
        })
        customer_fraud_stats.columns = ['tx_count', 'fraud_count', 'fraud_rate', 'total_amount']
        customer_fraud_stats = customer_fraud_stats[customer_fraud_stats['tx_count'] >= 5]
        
        risk_categories = ['No Fraud', 'Low Risk (0-5%)', 'Medium Risk (5-20%)', 'High Risk (20%+)']
        risk_counts = [
            (customer_fraud_stats['fraud_rate'] == 0).sum(),
            ((customer_fraud_stats['fraud_rate'] > 0) & (customer_fraud_stats['fraud_rate'] <= 0.05)).sum(),
            ((customer_fraud_stats['fraud_rate'] > 0.05) & (customer_fraud_stats['fraud_rate'] <= 0.20)).sum(),
            (customer_fraud_stats['fraud_rate'] > 0.20).sum()
        ]
        
        colors = ['green', 'yellow', 'orange', 'red']
        axes[2].pie(risk_counts, labels=risk_categories, colors=colors, autopct='%1.1f%%')
        axes[2].set_title('Customer Risk Distribution\n(Among customers with 5+ transactions)')
        
        # Prevention impact simulation
        detection_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        false_positive_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
        
        prevented_losses = [total_fraud_amount * dr for dr in detection_rates]
        blocked_legitimate = [self.train_tx[self.train_tx['TX_FRAUD'] == 0]['TX_AMOUNT'].sum() * fpr 
                             for fpr in false_positive_rates]
        
        ax3_twin = axes[3].twinx()
        line1 = axes[3].plot(detection_rates, [pl/1000 for pl in prevented_losses], 
                           'g-', linewidth=2, label='Prevented Losses')
        line2 = ax3_twin.plot(detection_rates, [bl/1000 for bl in blocked_legitimate], 
                             'r--', linewidth=2, label='Blocked Legitimate')
        
        axes[3].set_xlabel('Detection Rate')
        axes[3].set_ylabel('Prevented Losses (K$)', color='g')
        ax3_twin.set_ylabel('Blocked Legitimate (K$)', color='r')
        axes[3].set_title('Fraud Prevention Impact Simulation')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[3].legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.savefig('fraud_business_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nBUSINESS IMPACT SUMMARY")
        print("=" * 50)
        print(f"Total Fraud Losses: ${total_fraud_amount:,.2f}")
        print(f"Fraud as % of Total Volume: {fraud_percentage:.2f}%")
        print(f"Average Fraud Amount: ${self.train_tx[self.train_tx['TX_FRAUD'] == 1]['TX_AMOUNT'].mean():.2f}")
        print(f"Average Normal Amount: ${self.train_tx[self.train_tx['TX_FRAUD'] == 0]['TX_AMOUNT'].mean():.2f}")
        print(f"High Risk Customers (>20% fraud rate): {risk_counts[3]:,}")
        
        return total_fraud_amount, merchant_losses
    
    def feature_importance_analysis(self, model_file=None):
        if model_file:
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Get feature importance from first model (assuming LightGBM)
                model = model_data['models'][0]
                feature_names = model_data['feature_cols']
                
                if hasattr(model, 'feature_importance'):
                    importance = model.feature_importance(importance_type='gain')
                    feature_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(min(20, len(feature_imp))), 
                           feature_imp.head(20)['importance'], 
                           color='steelblue', alpha=0.7)
                    plt.yticks(range(min(20, len(feature_imp))), 
                              feature_imp.head(20)['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title('Top 20 Most Important Features')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    return feature_imp
                    
            except Exception as e:
                print(f"Could not load model file: {e}")
        
        return None
    
    def run_full_analysis(self, model_file=None):
        print("Loading data...")
        self.load_data()
        
        print("Running temporal analysis...")
        hourly_stats, daily_stats, monthly_stats = self.temporal_analysis()
        
        print("Running amount analysis...")
        fraud_amounts_desc, normal_amounts_desc = self.amount_analysis()
        
        print("Running geographic analysis...")
        distance_fraud = self.geographic_analysis()
        
        print("Running velocity analysis...")
        velocity_fraud, quick_repeat_analysis = self.velocity_analysis()
        
        print("Running merchant analysis...")
        mcc_fraud, business_fraud = self.merchant_analysis()
        
        print("Running business impact analysis...")
        total_fraud_losses, merchant_losses = self.business_impact_analysis()
        
        if model_file:
            print("Analyzing feature importance...")
            feature_importance = self.feature_importance_analysis(model_file)
        
        print("Analysis complete. Charts saved as PNG files.")
        
        return {
            'temporal': (hourly_stats, daily_stats, monthly_stats),
            'amounts': (fraud_amounts_desc, normal_amounts_desc),
            'geographic': distance_fraud,
            'velocity': (velocity_fraud, quick_repeat_analysis),
            'merchants': (mcc_fraud, business_fraud),
            'business_impact': (total_fraud_losses, merchant_losses)
        }

def run_visualization(model_file=None):
    visualizer = FraudDataVisualizer()
    return visualizer.run_full_analysis(model_file)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_visualization(model_file)