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
        
        # Daily transaction count per customer - FIXED
        daily_tx_count = self.train_tx.groupby(['CUSTOMER_ID', self.train_tx['TX_TS'].dt.date]).size()
        customer_daily_patterns = daily_tx_count.groupby('CUSTOMER_ID').agg(['mean', 'max', 'std']).fillna(0)
        customer_daily_patterns.columns = ['avg_daily_tx', 'max_daily_tx', 'std_daily_tx']
        
        # Merge back with fraud data
        customer_fraud_rate = self.train_tx.groupby('CUSTOMER_ID')['TX_FRAUD'].mean()
        customer_patterns = customer_daily_patterns.join(customer_fraud_rate)
        customer_patterns = customer_patterns.dropna()
        
        # FIXED: Lower threshold and add debugging
        high_activity = customer_patterns[customer_patterns['max_daily_tx'] > 3]
        print(f"High activity customers (>3 daily): {len(high_activity)}")
        print(f"Max daily transactions range: {customer_patterns['max_daily_tx'].min():.1f} to {customer_patterns['max_daily_tx'].max():.1f}")
        
        if len(high_activity) > 10:  # Only plot if we have enough data
            axes[1].scatter(high_activity['max_daily_tx'], high_activity['TX_FRAUD'], alpha=0.6)
            axes[1].set_xlabel('Max Daily Transactions')
            axes[1].set_ylabel('Customer Fraud Rate')
            axes[1].set_title(f'High Activity Customers: Volume vs Fraud Rate\n({len(high_activity)} customers)')
        else:
            axes[1].text(0.5, 0.5, f'Insufficient high activity customers\n(only {len(high_activity)} found)', 
                        ha='center', va='center', transform=axes[1].transAxes)
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
    
    def business_impact_analysis(self, model_file=None):
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
        
        # Replace the fake precision-recall simulation with real data analysis
        if model_file:
            # If we have a model, show something more meaningful
            axes[3].text(0.5, 0.5, 'Model-based analysis available\n(Feature importance shown separately)', 
                        ha='center', va='center', transform=axes[3].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[3].set_title('Model Analysis Available')
        else:
            # Show fraud concentration by amount ranges
            amount_ranges = pd.cut(self.train_tx['TX_AMOUNT'], 
                                 bins=[0, 50, 100, 200, 500, np.inf], 
                                 labels=['$0-50', '$50-100', '$100-200', '$200-500', '$500+'])
            
            range_analysis = self.train_tx.groupby(amount_ranges).agg({
                'TX_FRAUD': ['count', 'sum'],
                'TX_AMOUNT': 'sum'
            })
            range_analysis.columns = ['total_tx', 'fraud_tx', 'total_amount']
            range_analysis['fraud_concentration'] = range_analysis['fraud_tx'] / range_analysis['fraud_tx'].sum()
            range_analysis['volume_concentration'] = range_analysis['total_amount'] / range_analysis['total_amount'].sum()
            
            x = np.arange(len(range_analysis))
            width = 0.35
            
            axes[3].bar(x - width/2, range_analysis['fraud_concentration'], width, 
                       label='Fraud Concentration', color='red', alpha=0.7)
            axes[3].bar(x + width/2, range_analysis['volume_concentration'], width, 
                       label='Volume Concentration', color='blue', alpha=0.7)
            
            axes[3].set_xlabel('Amount Range')
            axes[3].set_ylabel('Concentration')
            axes[3].set_title('Fraud vs Volume Distribution by Amount Range')
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(range_analysis.index)
            axes[3].legend()
        
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
    
    def network_analysis(self):
        """NEW: Customer-Terminal interaction heatmap"""
        fig, axes = self.create_figure(figsize=(16, 10))
        
        # Customer-Terminal interaction matrix
        cust_term_matrix = self.train_tx.groupby(['CUSTOMER_ID', 'TERMINAL_ID']).agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': 'sum'
        })
        cust_term_matrix.columns = ['tx_count', 'fraud_count', 'fraud_rate', 'total_amount']
        cust_term_matrix = cust_term_matrix[cust_term_matrix['tx_count'] >= 3]  # At least 3 transactions
        
        # Top customer-terminal pairs by transaction volume
        top_pairs = cust_term_matrix.nlargest(50, 'tx_count')
        
        # Create network visualization data
        pair_data = []
        for (cust, term), row in top_pairs.iterrows():
            pair_data.append({
                'customer': f"C_{str(cust)[:6]}", 
                'terminal': f"T_{str(term)[:6]}",
                'transactions': row['tx_count'],
                'fraud_rate': row['fraud_rate']
            })
        
        pair_df = pd.DataFrame(pair_data)
        
        # Plot 1: Customer-Terminal interaction bubble chart
        scatter = axes[0].scatter(pair_df.index, pair_df['transactions'], 
                                 c=pair_df['fraud_rate'], s=pair_df['transactions']*3,
                                 alpha=0.7, cmap='Reds')
        axes[0].set_xlabel('Customer-Terminal Pair Index')
        axes[0].set_ylabel('Transaction Count')
        axes[0].set_title('Top 50 Customer-Terminal Pairs\n(Size=Volume, Color=Fraud Rate)')
        plt.colorbar(scatter, ax=axes[0])
        
        # Plot 2: Customer diversification (number of terminals used)
        cust_terminals = self.train_tx.groupby('CUSTOMER_ID')['TERMINAL_ID'].nunique()
        cust_fraud_rate = self.train_tx.groupby('CUSTOMER_ID')['TX_FRAUD'].mean()
        cust_diversity = pd.DataFrame({
            'terminal_count': cust_terminals,
            'fraud_rate': cust_fraud_rate
        }).dropna()
        
        # Bin by terminal diversity
        cust_diversity['diversity_bin'] = pd.cut(cust_diversity['terminal_count'], 
                                               bins=[0, 1, 2, 3, 5, 100], 
                                               labels=['1', '2', '3', '4-5', '6+'])
        diversity_fraud = cust_diversity.groupby('diversity_bin')['fraud_rate'].mean()
        
        axes[1].bar(range(len(diversity_fraud)), diversity_fraud.values, color='skyblue', alpha=0.7)
        axes[1].set_xticks(range(len(diversity_fraud)))
        axes[1].set_xticklabels(diversity_fraud.index)
        axes[1].set_xlabel('Number of Different Terminals Used')
        axes[1].set_ylabel('Average Fraud Rate')
        axes[1].set_title('Fraud Rate by Customer Terminal Diversity')
        
        # Plot 3: Terminal customer diversity
        term_customers = self.train_tx.groupby('TERMINAL_ID')['CUSTOMER_ID'].nunique()
        term_fraud_rate = self.train_tx.groupby('TERMINAL_ID')['TX_FRAUD'].mean()
        term_diversity = pd.DataFrame({
            'customer_count': term_customers,
            'fraud_rate': term_fraud_rate
        }).dropna()
        
        # Only terminals with significant activity
        term_diversity_active = term_diversity[term_diversity['customer_count'] >= 10]
        
        axes[2].scatter(term_diversity_active['customer_count'], 
                       term_diversity_active['fraud_rate'], alpha=0.6)
        axes[2].set_xlabel('Number of Different Customers')
        axes[2].set_ylabel('Terminal Fraud Rate')
        axes[2].set_title('Terminal Customer Base vs Fraud Rate')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Merchant risk distribution
        merchant_stats = self.train_tx.groupby('MERCHANT_ID').agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': ['sum', 'mean']
        })
        merchant_stats.columns = ['tx_count', 'fraud_count', 'fraud_rate', 'total_amount', 'avg_amount']
        merchant_stats = merchant_stats[merchant_stats['tx_count'] >= 50]  # Minimum transactions
        
        # Create risk categories
        merchant_stats['risk_category'] = pd.cut(
            merchant_stats['fraud_rate'],
            bins=[0, 0.01, 0.05, 0.1, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
        
        risk_dist = merchant_stats['risk_category'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        
        axes[3].pie(risk_dist.values, labels=risk_dist.index, colors=colors[:len(risk_dist)], 
                   autopct='%1.1f%%')
        axes[3].set_title('Merchant Risk Distribution\n(Based on Fraud Rate)')
        
        plt.tight_layout()
        plt.savefig('fraud_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pair_df, cust_diversity, term_diversity, merchant_stats
    
    def temporal_deep_dive(self):
        """NEW: Advanced temporal pattern analysis"""
        fig, axes = self.create_figure(figsize=(16, 10))
        
        # Calculate time between transactions for velocity analysis
        tx_sorted = self.train_tx.sort_values(['CUSTOMER_ID', 'TX_TS'])
        tx_sorted['prev_tx_time'] = tx_sorted.groupby('CUSTOMER_ID')['TX_TS'].shift(1)
        tx_sorted['time_diff_seconds'] = (tx_sorted['TX_TS'] - tx_sorted['prev_tx_time']).dt.total_seconds()
        tx_sorted = tx_sorted.dropna(subset=['time_diff_seconds'])
        
        # Plot 1: Transaction velocity heatmap by hour and day
        tx_sorted['hour'] = tx_sorted['TX_TS'].dt.hour
        tx_sorted['day_of_week'] = tx_sorted['TX_TS'].dt.dayofweek
        
        # Create velocity categories
        tx_sorted['velocity_cat'] = pd.cut(
            tx_sorted['time_diff_seconds'],
            bins=[0, 300, 1800, 3600, 86400, np.inf],  # 5min, 30min, 1hr, 1day, inf
            labels=['<5min', '5-30min', '30min-1hr', '1hr-1day', '>1day']
        )
        
        # Quick transactions (< 30 min) fraud rate by hour and day
        quick_tx = tx_sorted[tx_sorted['time_diff_seconds'] < 1800]  # < 30 minutes
        if len(quick_tx) > 100:
            quick_heatmap = quick_tx.pivot_table(
                values='TX_FRAUD', 
                index='day_of_week', 
                columns='hour', 
                aggfunc='mean'
            )
            
            im = axes[0].imshow(quick_heatmap, cmap='Reds', aspect='auto')
            axes[0].set_xlabel('Hour of Day')
            axes[0].set_ylabel('Day of Week')
            axes[0].set_title('Quick Transaction (<30min) Fraud Rate Heatmap')
            axes[0].set_yticks(range(7))
            axes[0].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.colorbar(im, ax=axes[0])
        
        # Plot 2: Customer transaction count vs fraud rate
        customer_tx_count = self.train_tx.groupby('CUSTOMER_ID').agg({
            'TX_FRAUD': ['count', 'sum', 'mean'],
            'TX_AMOUNT': 'sum'
        })
        customer_tx_count.columns = ['tx_count', 'fraud_count', 'fraud_rate', 'total_amount']
        customer_tx_count = customer_tx_count[customer_tx_count['tx_count'] >= 5]
        
        # Create bins for transaction count
        customer_tx_count['count_bin'] = pd.cut(
            customer_tx_count['tx_count'],
            bins=[0, 10, 25, 50, 100, 1000],
            labels=['5-10', '11-25', '26-50', '51-100', '100+']
        )
        
        count_fraud = customer_tx_count.groupby('count_bin')['fraud_rate'].mean()
        
        axes[1].bar(range(len(count_fraud)), count_fraud.values, color='coral', alpha=0.7)
        axes[1].set_xticks(range(len(count_fraud)))
        axes[1].set_xticklabels(count_fraud.index)
        axes[1].set_xlabel('Customer Transaction Count Range')
        axes[1].set_ylabel('Average Fraud Rate')
        axes[1].set_title('Fraud Rate by Customer Activity Level')
        
        # Plot 3: Terminal utilization patterns
        terminal_hourly = self.train_tx.groupby(['TERMINAL_ID', 'hour']).agg({
            'TX_FRAUD': ['count', 'mean']
        })
        terminal_hourly.columns = ['tx_count', 'fraud_rate']
        terminal_hourly = terminal_hourly.reset_index()
        
        # Find terminals with unusual hour patterns (high fraud at certain hours)
        terminal_hour_fraud = terminal_hourly.pivot_table(
            values='fraud_rate',
            index='TERMINAL_ID',
            columns='hour',
            aggfunc='mean'
        ).fillna(0)
        
        # Calculate variance in fraud rate across hours for each terminal
        terminal_hour_fraud['fraud_variance'] = terminal_hour_fraud.var(axis=1)
        high_variance_terminals = terminal_hour_fraud.nlargest(20, 'fraud_variance')
        
        # Plot fraud rate patterns for top variable terminals
        for i, (terminal_id, row) in enumerate(high_variance_terminals.head(5).iterrows()):
            if i == 0:
                axes[2].plot(range(24), row[:-1], label=f'T_{str(terminal_id)[:6]}', alpha=0.7)
            else:
                axes[2].plot(range(24), row[:-1], alpha=0.7)
        
        axes[2].set_xlabel('Hour of Day')
        axes[2].set_ylabel('Fraud Rate')
        axes[2].set_title('Top 5 Terminals with Variable Fraud Patterns by Hour')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Recurring transaction analysis
        recurring_stats = self.train_tx.groupby('IS_RECURRING_TRANSACTION').agg({
            'TX_FRAUD': ['count', 'mean'],
            'TX_AMOUNT': 'mean'
        })
        recurring_stats.columns = ['tx_count', 'fraud_rate', 'avg_amount']
        
        # Also analyze by card brand
        card_fraud = self.train_tx.groupby('CARD_BRAND').agg({
            'TX_FRAUD': ['count', 'mean'],
            'TX_AMOUNT': 'mean'
        })
        card_fraud.columns = ['tx_count', 'fraud_rate', 'avg_amount']
        card_fraud = card_fraud[card_fraud['tx_count'] >= 1000].sort_values('fraud_rate', ascending=False)
        
        if len(card_fraud) > 0:
            axes[3].bar(range(len(card_fraud)), card_fraud['fraud_rate'], color='lightgreen', alpha=0.7)
            axes[3].set_xticks(range(len(card_fraud)))
            axes[3].set_xticklabels(card_fraud.index, rotation=45)
            axes[3].set_ylabel('Fraud Rate')
            axes[3].set_title('Fraud Rate by Card Brand')
        
        plt.tight_layout()
        plt.savefig('fraud_temporal_deep_dive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tx_sorted, customer_tx_count, terminal_hourly, card_fraud
    
    def behavioral_analysis(self):
        """NEW: Advanced behavioral pattern analysis"""
        fig, axes = self.create_figure(figsize=(16, 10))
        
        # Plot 1: Amount anomaly patterns
        # Calculate z-scores for amounts by customer
        customer_amount_stats = self.train_tx.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(['mean', 'std'])
        customer_amount_stats['std'] = customer_amount_stats['std'].fillna(0)
        
        # Merge back to get z-scores
        tx_with_stats = self.train_tx.merge(
            customer_amount_stats.reset_index(),
            on='CUSTOMER_ID',
            suffixes=('', '_cust')
        )
        
        # Calculate amount z-score per customer
        tx_with_stats['amount_zscore'] = np.where(
            tx_with_stats['std'] > 0,
            (tx_with_stats['TX_AMOUNT'] - tx_with_stats['mean']) / tx_with_stats['std'],
            0
        )
        
        # Fraud rate by amount z-score bins
        tx_with_stats['zscore_bin'] = pd.cut(
            tx_with_stats['amount_zscore'],
            bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
            labels=['<<-2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '>>2']
        )
        
        zscore_fraud = tx_with_stats.groupby('zscore_bin')['TX_FRAUD'].agg(['count', 'mean'])
        zscore_fraud = zscore_fraud[zscore_fraud['count'] >= 100]
        
        axes[0].bar(range(len(zscore_fraud)), zscore_fraud['mean'], color='purple', alpha=0.7)
        axes[0].set_xticks(range(len(zscore_fraud)))
        axes[0].set_xticklabels(zscore_fraud.index)
        axes[0].set_xlabel('Amount Z-Score (vs Customer History)')
        axes[0].set_ylabel('Fraud Rate')
        axes[0].set_title('Fraud Rate by Amount Anomaly Level')
        
        # Plot 2: Payment method risk analysis
        payment_methods = ['CARD_BRAND', 'TRANSACTION_TYPE', 'CARDHOLDER_AUTH_METHOD']
        
        for i, method in enumerate(payment_methods):
            if method in self.train_tx.columns and i < 3:  # Only show first 3
                method_fraud = self.train_tx.groupby(method).agg({
                    'TX_FRAUD': ['count', 'mean']
                })
                method_fraud.columns = ['tx_count', 'fraud_rate']
                method_fraud = method_fraud[method_fraud['tx_count'] >= 500].sort_values('fraud_rate', ascending=False)
                
                if len(method_fraud) > 0 and i == 1:  # Show TRANSACTION_TYPE
                    axes[1].bar(range(len(method_fraud)), method_fraud['fraud_rate'], 
                               color='orange', alpha=0.7)
                    axes[1].set_xticks(range(len(method_fraud)))
                    axes[1].set_xticklabels(method_fraud.index, rotation=45, ha='right')
                    axes[1].set_ylabel('Fraud Rate')
                    axes[1].set_title('Fraud Rate by Transaction Type')
                    break
        
        # Plot 3: Geographic fraud hotspots
        # Merge with customer locations
        tx_with_locations = self.train_tx.merge(self.customers, on='CUSTOMER_ID', how='left')
        
        # Create grid cells for heatmap
        tx_with_locations['x_grid'] = (tx_with_locations['x_customer_id'] / 10).astype(int) * 10
        tx_with_locations['y_grid'] = (tx_with_locations['y_customer_id'] / 10).astype(int) * 10
        
        # Grid fraud rates
        grid_fraud = tx_with_locations.groupby(['x_grid', 'y_grid']).agg({
            'TX_FRAUD': ['count', 'mean']
        })
        grid_fraud.columns = ['tx_count', 'fraud_rate']
        grid_fraud = grid_fraud[grid_fraud['tx_count'] >= 50]  # Minimum transactions per grid
        
        # Create heatmap data
        if len(grid_fraud) > 0:
            heatmap_data = grid_fraud['fraud_rate'].unstack(fill_value=0)
            im = axes[2].imshow(heatmap_data, cmap='Reds', aspect='auto', origin='lower')
            axes[2].set_xlabel('X Grid (0-100)')
            axes[2].set_ylabel('Y Grid (0-100)')
            axes[2].set_title('Geographic Fraud Rate Heatmap\n(10x10 grid cells)')
            plt.colorbar(im, ax=axes[2])
        
        # Plot 4: Customer loyalty vs fraud
        # Calculate customer tenure and transaction frequency
        customer_patterns = self.train_tx.groupby('CUSTOMER_ID').agg({
            'TX_TS': ['min', 'max', 'count'],
            'TX_FRAUD': 'mean',
            'TX_AMOUNT': 'mean'
        })
        customer_patterns.columns = ['first_tx', 'last_tx', 'tx_count', 'fraud_rate', 'avg_amount']
        
        # Calculate tenure in days
        customer_patterns['tenure_days'] = (
            customer_patterns['last_tx'] - customer_patterns['first_tx']
        ).dt.days
        customer_patterns['tenure_days'] = customer_patterns['tenure_days'].fillna(0)
        
        # Calculate transaction frequency (transactions per day)
        customer_patterns['tx_frequency'] = customer_patterns['tx_count'] / (customer_patterns['tenure_days'] + 1)
        
        # Only customers with multiple transactions
        active_customers = customer_patterns[customer_patterns['tx_count'] >= 5]
        
        # Create frequency bins
        active_customers['freq_bin'] = pd.cut(
            active_customers['tx_frequency'],
            bins=[0, 0.1, 0.5, 1.0, 5.0, np.inf],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        freq_fraud = active_customers.groupby('freq_bin')['fraud_rate'].mean()
        
        axes[3].bar(range(len(freq_fraud)), freq_fraud.values, color='teal', alpha=0.7)
        axes[3].set_xticks(range(len(freq_fraud)))
        axes[3].set_xticklabels(freq_fraud.index)
        axes[3].set_xlabel('Transaction Frequency Category')
        axes[3].set_ylabel('Average Fraud Rate')
        axes[3].set_title('Fraud Rate by Customer Transaction Frequency')
        
        plt.tight_layout()
        plt.savefig('fraud_behavioral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tx_with_stats, grid_fraud, customer_patterns, active_customers
    
    def feature_importance_analysis(self, model_file=None):
        if model_file:
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Get feature importance from models - UPDATED for all model types including naivebayes
                models = model_data['models']
                feature_names = model_data['feature_cols']
                model_type = model_data['model_type']
                
                # Average importance across all folds
                all_importances = []
                
                for model in models:
                    if model_type == 'lightgbm':
                        importance = model.feature_importance(importance_type='gain')
                    elif model_type == 'xgboost':
                        # XGBoost returns dict, need to align with feature names
                        importance_dict = model.get_score(importance_type='gain')
                        importance = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
                    elif model_type == 'catboost':
                        importance = model.feature_importances_
                    elif model_type in ['randomforest', 'logistic']:
                        importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
                    elif model_type in ['knn', 'naivebayes']:
                        # These models don't have meaningful feature importance
                        print(f"{model_type.upper()} doesn't have traditional feature importance - skipping visualization")
                        return None
                    else:
                        continue
                    
                    all_importances.append(importance)
                
                if all_importances:
                    # Average across folds
                    avg_importance = np.mean(all_importances, axis=0)
                    
                    feature_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance': avg_importance
                    }).sort_values('importance', ascending=False)
                    
                    plt.figure(figsize=(12, 8))
                    top_features = min(25, len(feature_imp))
                    plt.barh(range(top_features), 
                        feature_imp.head(top_features)['importance'], 
                        color='steelblue', alpha=0.7)
                    plt.yticks(range(top_features), 
                            feature_imp.head(top_features)['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top {top_features} Most Important Features ({model_type.upper()})')
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
        total_fraud_losses, merchant_losses = self.business_impact_analysis(model_file)
        
        print("Running network analysis...")
        network_results = self.network_analysis()
        
        print("Running temporal deep dive...")
        temporal_results = self.temporal_deep_dive()
        
        print("Running behavioral analysis...")
        behavioral_results = self.behavioral_analysis()
        
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
            'business_impact': (total_fraud_losses, merchant_losses),
            'network': network_results,
            'temporal_deep': temporal_results,
            'behavioral': behavioral_results
        }

def run_visualization(model_file=None):
    visualizer = FraudDataVisualizer()
    return visualizer.run_full_analysis(model_file)

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_visualization(model_file)