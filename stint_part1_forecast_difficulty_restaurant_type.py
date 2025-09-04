"""
Stint Data Science Technical Task - Part 1: Forecast Difficulty by Restaurant Type Analysis
Analyse how forecast difficulty varies by restaurant type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_and_prepare_data():
    """Load and prepare the restaurant demand dataset."""
    print("Loading restaurant demand data for forecast difficulty analysis by restaurant type...")
    df = pd.read_csv('ds_task_dataset.csv')
    
    df = df.dropna(subset=['restaurant_type'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create comprehensive features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
    
    # Calculate customer count
    df['customer_count'] = df['main_meal_count'].fillna(0) * 1.2
    
    # Create business context features
    df['is_lunch_period'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_dinner_period'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_peak_period'] = (df['is_lunch_period'] | df['is_dinner_period']).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Restaurant types: {df['restaurant_type'].unique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def analyze_forecast_difficulty_by_type(df):
    """Comprehensive analysis of forecast difficulty varying by restaurant type."""
    
    print("\n" + "="*80)
    print("FORECAST DIFFICULTY ANALYSIS BY RESTAURANT TYPE")
    print("="*80)
    
    restaurant_analysis = {}
    
    # Sample for computational efficiency
    df_sample = df.sample(n=min(50000, len(df)), random_state=42)
    
    for restaurant_type in df_sample['restaurant_type'].unique():
        print(f"\n🏪 ANALYZING: {restaurant_type.upper()}")
        print("-" * 60)
        
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant_type].copy()
        
        if len(rest_data) < 500:
            print(f"   ⚠️  Insufficient data ({len(rest_data)} records)")
            continue
        
        # Sort by timestamp for time series analysis
        rest_data = rest_data.sort_values('timestamp').reset_index(drop=True)
        
        # 1. BASIC VARIABILITY METRICS
        mean_demand = rest_data['customer_count'].mean()
        std_demand = rest_data['customer_count'].std()
        cv_demand = std_demand / mean_demand if mean_demand > 0 else 0
        
        # Quantile dispersion
        q75 = rest_data['customer_count'].quantile(0.75)
        q25 = rest_data['customer_count'].quantile(0.25)
        iqr = q75 - q25
        quartile_dispersion = iqr / (q75 + q25) if (q75 + q25) > 0 else 0
        
        # 2. TEMPORAL PATTERN ANALYSIS
        # Hourly pattern consistency
        hourly_means = rest_data.groupby('hour')['customer_count'].mean()
        hourly_stds = rest_data.groupby('hour')['customer_count'].std()
        hourly_cvs = hourly_stds / hourly_means
        hourly_consistency = 1 - hourly_cvs.mean()  # Higher = more consistent
        
        # Daily pattern consistency
        daily_means = rest_data.groupby('day_of_week')['customer_count'].mean()
        daily_stds = rest_data.groupby('day_of_week')['customer_count'].std()
        daily_cvs = daily_stds / daily_means
        daily_consistency = 1 - daily_cvs.mean()
        
        # Monthly seasonal consistency
        monthly_means = rest_data.groupby('month')['customer_count'].mean()
        seasonal_cv = monthly_means.std() / monthly_means.mean() if monthly_means.mean() > 0 else 0
        
        # 3. TREND AND STATIONARITY
        # Calculate rolling means to identify trends
        rest_data['rolling_7d'] = rest_data['customer_count'].rolling(window=336, min_periods=100).mean()  # 7 days = 336 periods
        rest_data['rolling_30d'] = rest_data['customer_count'].rolling(window=1440, min_periods=500).mean()  # 30 days
        
        # Trend strength
        if len(rest_data) > 1440:  # Need enough data for 30-day trend
            trend_correlation = rest_data['customer_count'].corr(pd.Series(range(len(rest_data)))) if len(rest_data) > 100 else 0
            trend_strength = abs(trend_correlation)
        else:
            trend_strength = 0
        
        # 4. AUTOCORRELATION ANALYSIS
        # Various lags for different time horizons
        autocorr_1period = rest_data['customer_count'].autocorr(lag=1) if len(rest_data) > 1 else 0
        autocorr_24h = rest_data['customer_count'].autocorr(lag=48) if len(rest_data) > 48 else 0  # Same time next day
        autocorr_7d = rest_data['customer_count'].autocorr(lag=336) if len(rest_data) > 336 else 0  # Same time next week
        
        # Average predictability from autocorrelations
        autocorr_predictability = np.mean([abs(autocorr_1period), abs(autocorr_24h), abs(autocorr_7d)])
        
        # 5. EXTERNAL FACTOR SENSITIVITY
        external_factors = ['temperature', 'precipitation', 'economic_indicator', 
                           'competitor_promo', 'social_trend', 'local_event']
        
        external_correlations = []
        for factor in external_factors:
            if factor in rest_data.columns:
                corr = abs(rest_data['customer_count'].corr(rest_data[factor]))
                if not np.isnan(corr):
                    external_correlations.append(corr)
        
        avg_external_sensitivity = np.mean(external_correlations) if external_correlations else 0
        max_external_sensitivity = np.max(external_correlations) if external_correlations else 0
        
        # 6. PEAK DETECTION AND REGULARITY
        demand_values = rest_data['customer_count'].values
        peaks, _ = find_peaks(demand_values, height=np.percentile(demand_values, 80))
        
        if len(peaks) > 2:
            peak_intervals = np.diff(peaks)
            peak_regularity = 1 - (np.std(peak_intervals) / np.mean(peak_intervals)) if np.mean(peak_intervals) > 0 else 0
            peak_regularity = max(0, peak_regularity)
            avg_peak_interval = np.mean(peak_intervals) * 0.5  # Convert to hours
        else:
            peak_regularity = 0
            avg_peak_interval = 0
        
        # 7. VOLATILITY IN DIFFERENT CONDITIONS
        # Weekend vs Weekday volatility
        weekend_data = rest_data[rest_data['is_weekend'] == 1]
        weekday_data = rest_data[rest_data['is_weekend'] == 0]
        
        weekend_cv = weekend_data['customer_count'].std() / weekend_data['customer_count'].mean() if len(weekend_data) > 10 and weekend_data['customer_count'].mean() > 0 else 0
        weekday_cv = weekday_data['customer_count'].std() / weekday_data['customer_count'].mean() if len(weekday_data) > 10 and weekday_data['customer_count'].mean() > 0 else 0
        
        weekend_volatility_ratio = weekend_cv / weekday_cv if weekday_cv > 0 else 1
        
        # Peak vs Off-peak volatility
        peak_data = rest_data[rest_data['is_peak_period'] == 1]
        offpeak_data = rest_data[rest_data['is_peak_period'] == 0]
        
        peak_cv = peak_data['customer_count'].std() / peak_data['customer_count'].mean() if len(peak_data) > 10 and peak_data['customer_count'].mean() > 0 else 0
        offpeak_cv = offpeak_data['customer_count'].std() / offpeak_data['customer_count'].mean() if len(offpeak_data) > 10 and offpeak_data['customer_count'].mean() > 0 else 0
        
        peak_volatility_ratio = peak_cv / offpeak_cv if offpeak_cv > 0 else 1
        
        # 8. BUSINESS-SPECIFIC CHARACTERISTICS
        # Revenue volatility (if customers and revenue don't move together, it's harder to predict)
        revenue_per_customer = rest_data['total_sales'] / (rest_data['customer_count'] + 0.01)
        revenue_consistency = 1 - (revenue_per_customer.std() / revenue_per_customer.mean()) if revenue_per_customer.mean() > 0 else 0
        
        # Capacity utilization patterns
        capacity_utilization = rest_data['customer_count'] / (rest_data['capacity_available'] + 1)
        capacity_cv = capacity_utilization.std() / capacity_utilization.mean() if capacity_utilization.mean() > 0 else 0
        
        # 9. COMPOSITE FORECAST DIFFICULTY SCORES
        
        # Volatility component (25%)
        volatility_score = min(cv_demand / 0.6, 1.0)  # Normalize to 0-1
        
        # Predictability component (25%)
        predictability_score = 1 - autocorr_predictability  # Lower autocorr = harder to predict
        
        # Pattern consistency component (20%)
        consistency_score = 1 - ((hourly_consistency + daily_consistency) / 2)
        
        # External sensitivity component (15%)
        external_score = avg_external_sensitivity
        
        # Peak irregularity component (10%)
        irregularity_score = 1 - peak_regularity
        
        # Conditional volatility component (5%)
        conditional_vol_score = min((weekend_volatility_ratio + peak_volatility_ratio - 2) / 2, 1.0)
        
        # Composite score
        composite_difficulty = (
            0.25 * volatility_score +
            0.25 * predictability_score +
            0.20 * consistency_score +
            0.15 * external_score +
            0.10 * irregularity_score +
            0.05 * conditional_vol_score
        )
        
        # 10. FORECAST ERROR SIMULATION (Simple naive forecast evaluation)
        # Simple moving average forecast
        if len(rest_data) > 100:
            rest_data['forecast_naive'] = rest_data['customer_count'].shift(48)  # Previous day same time
            rest_data['forecast_ma7'] = rest_data['customer_count'].rolling(window=336).mean()  # 7-day MA
            
            # Calculate errors where we have both actual and forecast
            valid_naive = rest_data.dropna(subset=['forecast_naive', 'customer_count'])
            valid_ma = rest_data.dropna(subset=['forecast_ma7', 'customer_count'])
            
            if len(valid_naive) > 50:
                naive_mae = mean_absolute_error(valid_naive['customer_count'], valid_naive['forecast_naive'])
                naive_rmse = np.sqrt(mean_squared_error(valid_naive['customer_count'], valid_naive['forecast_naive']))
                naive_mape = np.mean(np.abs((valid_naive['customer_count'] - valid_naive['forecast_naive']) / (valid_naive['customer_count'] + 0.01))) * 100
            else:
                naive_mae = naive_rmse = naive_mape = 0
            
            if len(valid_ma) > 50:
                ma_mae = mean_absolute_error(valid_ma['customer_count'], valid_ma['forecast_ma7'])
                ma_rmse = np.sqrt(mean_squared_error(valid_ma['customer_count'], valid_ma['forecast_ma7']))
                ma_mape = np.mean(np.abs((valid_ma['customer_count'] - valid_ma['forecast_ma7']) / (valid_ma['customer_count'] + 0.01))) * 100
            else:
                ma_mae = ma_rmse = ma_mape = 0
        else:
            naive_mae = naive_rmse = naive_mape = 0
            ma_mae = ma_rmse = ma_mape = 0
        
        # Store comprehensive analysis
        restaurant_analysis[restaurant_type] = {
            # Basic statistics
            'mean_demand': round(mean_demand, 1),
            'std_demand': round(std_demand, 1),
            'cv_demand': round(cv_demand, 3),
            'quartile_dispersion': round(quartile_dispersion, 3),
            
            # Pattern consistency
            'hourly_consistency': round(hourly_consistency, 3),
            'daily_consistency': round(daily_consistency, 3),
            'seasonal_cv': round(seasonal_cv, 3),
            
            # Predictability
            'autocorr_1period': round(autocorr_1period, 3),
            'autocorr_24h': round(autocorr_24h, 3),
            'autocorr_7d': round(autocorr_7d, 3),
            'autocorr_predictability': round(autocorr_predictability, 3),
            
            # External factors
            'avg_external_sensitivity': round(avg_external_sensitivity, 3),
            'max_external_sensitivity': round(max_external_sensitivity, 3),
            
            # Peak characteristics
            'peak_regularity': round(peak_regularity, 3),
            'avg_peak_interval_hours': round(avg_peak_interval, 1),
            'num_peaks': len(peaks),
            
            # Conditional volatility
            'weekend_volatility_ratio': round(weekend_volatility_ratio, 3),
            'peak_volatility_ratio': round(peak_volatility_ratio, 3),
            
            # Business characteristics
            'revenue_consistency': round(revenue_consistency, 3),
            'capacity_cv': round(capacity_cv, 3),
            
            # Forecast difficulty components
            'difficulty_components': {
                'volatility': round(volatility_score, 3),
                'predictability': round(predictability_score, 3),
                'consistency': round(consistency_score, 3),
                'external': round(external_score, 3),
                'irregularity': round(irregularity_score, 3),
                'conditional_vol': round(conditional_vol_score, 3)
            },
            
            # Overall scores
            'composite_difficulty': round(composite_difficulty, 3),
            'trend_strength': round(trend_strength, 3),
            
            # Forecast error estimates
            'naive_mae': round(naive_mae, 1),
            'naive_rmse': round(naive_rmse, 1),
            'naive_mape': round(naive_mape, 1),
            'ma7_mae': round(ma_mae, 1),
            'ma7_rmse': round(ma_rmse, 1),
            'ma7_mape': round(ma_mape, 1),
            
            # Sample size
            'sample_size': len(rest_data)
        }
        
        # Print summary
        print(f"   📊 Sample Size: {len(rest_data):,} records")
        print(f"   🎯 Composite Difficulty: {composite_difficulty:.3f}/1.000")
        print(f"   📈 Coefficient of Variation: {cv_demand:.3f}")
        print(f"   🔮 Autocorr Predictability: {autocorr_predictability:.3f}")
        print(f"   🌍 External Sensitivity: {avg_external_sensitivity:.3f}")
        print(f"   ⏰ Pattern Consistency: {((hourly_consistency + daily_consistency) / 2):.3f}")
        print(f"   📊 Naive Forecast MAPE: {naive_mape:.1f}%")
    
    return restaurant_analysis

def compare_restaurant_types(restaurant_analysis):
    """Compare forecast difficulty across restaurant types and identify patterns."""
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS ACROSS RESTAURANT TYPES")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_metrics = []
    for restaurant_type, analysis in restaurant_analysis.items():
        metrics = {
            'Restaurant_Type': restaurant_type,
            'Difficulty_Score': analysis['composite_difficulty'],
            'CV_Demand': analysis['cv_demand'],
            'Autocorr_Predictability': analysis['autocorr_predictability'],
            'External_Sensitivity': analysis['avg_external_sensitivity'],
            'Pattern_Consistency': (analysis['hourly_consistency'] + analysis['daily_consistency']) / 2,
            'Peak_Regularity': analysis['peak_regularity'],
            'Weekend_Volatility_Ratio': analysis['weekend_volatility_ratio'],
            'Naive_MAPE': analysis['naive_mape'],
            'MA7_MAPE': analysis['ma7_mape'],
            'Sample_Size': analysis['sample_size']
        }
        comparison_metrics.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_metrics)
    comparison_df = comparison_df.sort_values('Difficulty_Score', ascending=False)
    
    print("\n📋 FORECAST DIFFICULTY RANKING:")
    print("-" * 50)
    
    for i, row in comparison_df.iterrows():
        restaurant = row['Restaurant_Type']
        score = row['Difficulty_Score']
        
        # Classify difficulty level
        if score > 0.7:
            level = "EXTREMELY DIFFICULT"
            emoji = "🔥"
        elif score > 0.5:
            level = "DIFFICULT"
            emoji = "⚠️"
        elif score > 0.3:
            level = "MODERATE"
            emoji = "📊"
        else:
            level = "RELATIVELY EASY"
            emoji = "✅"
        
        rank = list(comparison_df['Restaurant_Type']).index(restaurant) + 1
        print(f"{rank}. {emoji} {restaurant.upper()}: {score:.3f} - {level}")
        print(f"   • CV: {row['CV_Demand']:.3f} | Predictability: {row['Autocorr_Predictability']:.3f} | "
              f"External Sens: {row['External_Sensitivity']:.3f}")
        print(f"   • Naive MAPE: {row['Naive_MAPE']:.1f}% | MA7 MAPE: {row['MA7_MAPE']:.1f}%")
    
    # Statistical analysis
    print(f"\n📈 STATISTICAL SUMMARY:")
    print("-" * 30)
    print(f"Average Difficulty Score: {comparison_df['Difficulty_Score'].mean():.3f}")
    print(f"Difficulty Range: {comparison_df['Difficulty_Score'].min():.3f} - {comparison_df['Difficulty_Score'].max():.3f}")
    print(f"Most Variable Restaurant: {comparison_df.loc[comparison_df['CV_Demand'].idxmax(), 'Restaurant_Type']}")
    print(f"Least Predictable: {comparison_df.loc[comparison_df['Autocorr_Predictability'].idxmin(), 'Restaurant_Type']}")
    print(f"Most External Sensitive: {comparison_df.loc[comparison_df['External_Sensitivity'].idxmax(), 'Restaurant_Type']}")
    
    # Identify patterns and clusters
    print(f"\n🔍 RESTAURANT TYPE PATTERNS:")
    print("-" * 40)
    
    # High-end vs Casual analysis
    high_end_types = ['fine dining', 'seafood']  # Typically higher-end
    casual_types = ['fast casual', 'casual bistro', 'family restaurant']  # More casual
    
    high_end_difficulty = comparison_df[comparison_df['Restaurant_Type'].isin(high_end_types)]['Difficulty_Score'].mean()
    casual_difficulty = comparison_df[comparison_df['Restaurant_Type'].isin(casual_types)]['Difficulty_Score'].mean()
    
    print(f"High-end restaurants avg difficulty: {high_end_difficulty:.3f}")
    print(f"Casual restaurants avg difficulty: {casual_difficulty:.3f}")
    
    if high_end_difficulty > casual_difficulty:
        print(f"→ High-end restaurants are {((high_end_difficulty/casual_difficulty - 1) * 100):.1f}% more difficult to forecast")
    else:
        print(f"→ Casual restaurants are {((casual_difficulty/high_end_difficulty - 1) * 100):.1f}% more difficult to forecast")
    
    # Correlation analysis
    print(f"\n📊 CORRELATION INSIGHTS:")
    print("-" * 30)
    
    # Correlations between difficulty factors
    correlations = {
        'CV vs External Sensitivity': comparison_df['CV_Demand'].corr(comparison_df['External_Sensitivity']),
        'Difficulty vs Predictability': comparison_df['Difficulty_Score'].corr(comparison_df['Autocorr_Predictability']),
        'Difficulty vs Pattern Consistency': comparison_df['Difficulty_Score'].corr(comparison_df['Pattern_Consistency']),
        'Weekend Volatility vs Difficulty': comparison_df['Weekend_Volatility_Ratio'].corr(comparison_df['Difficulty_Score'])
    }
    
    for relationship, corr in correlations.items():
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"• {relationship}: {corr:.3f} ({strength} {direction})")
    
    return comparison_df

def identify_restaurant_specific_challenges(restaurant_analysis):
    """Identify specific forecasting challenges for each restaurant type."""
    
    print("\n" + "="*80)
    print("RESTAURANT-SPECIFIC FORECASTING CHALLENGES")
    print("="*80)
    
    challenges = {}
    
    for restaurant_type, analysis in restaurant_analysis.items():
        print(f"\n🏪 {restaurant_type.upper()} - SPECIFIC CHALLENGES:")
        print("-" * 60)
        
        restaurant_challenges = []
        
        # Volatility challenges
        if analysis['cv_demand'] > 0.5:
            restaurant_challenges.append({
                'type': 'High Volatility',
                'severity': 'High' if analysis['cv_demand'] > 0.7 else 'Medium',
                'description': f"Demand CV of {analysis['cv_demand']:.3f} indicates highly unpredictable demand swings",
                'recommendation': "Implement buffer staffing and flexible scheduling"
            })
        
        # Predictability challenges  
        if analysis['autocorr_predictability'] < 0.3:
            restaurant_challenges.append({
                'type': 'Low Predictability',
                'severity': 'High' if analysis['autocorr_predictability'] < 0.1 else 'Medium',
                'description': f"Low autocorrelation ({analysis['autocorr_predictability']:.3f}) means past patterns don't predict future well",
                'recommendation': "Focus on real-time demand signals and short-term forecasting"
            })
        
        # External sensitivity challenges
        if analysis['avg_external_sensitivity'] > 0.4:
            restaurant_challenges.append({
                'type': 'External Factor Sensitivity',
                'severity': 'High' if analysis['avg_external_sensitivity'] > 0.6 else 'Medium',
                'description': f"High sensitivity to external factors ({analysis['avg_external_sensitivity']:.3f}) creates unpredictable spikes",
                'recommendation': "Monitor weather, events, and competitor activity closely"
            })
        
        # Pattern inconsistency challenges
        avg_consistency = (analysis['hourly_consistency'] + analysis['daily_consistency']) / 2
        if avg_consistency < 0.5:
            restaurant_challenges.append({
                'type': 'Irregular Patterns',
                'severity': 'High' if avg_consistency < 0.3 else 'Medium',
                'description': f"Low pattern consistency ({avg_consistency:.3f}) makes template-based forecasting difficult",
                'recommendation': "Use adaptive algorithms that can handle pattern changes"
            })
        
        # Peak irregularity challenges
        if analysis['peak_regularity'] < 0.4:
            restaurant_challenges.append({
                'type': 'Irregular Peak Patterns',
                'severity': 'Medium',
                'description': f"Irregular peak timing ({analysis['peak_regularity']:.3f}) makes staffing optimization difficult",
                'recommendation': "Use flexible staffing with on-call personnel for peaks"
            })
        
        # Weekend/peak volatility challenges
        if analysis['weekend_volatility_ratio'] > 1.5 or analysis['peak_volatility_ratio'] > 1.5:
            restaurant_challenges.append({
                'type': 'Conditional Volatility',
                'severity': 'Medium',
                'description': f"Higher volatility during weekends ({analysis['weekend_volatility_ratio']:.1f}x) and peaks ({analysis['peak_volatility_ratio']:.1f}x)",
                'recommendation': "Separate forecasting models for different time periods"
            })
        
        # Revenue inconsistency challenges
        if analysis['revenue_consistency'] < 0.5:
            restaurant_challenges.append({
                'type': 'Revenue Pattern Mismatch',
                'severity': 'Low',
                'description': f"Customer count and revenue patterns don't align well ({analysis['revenue_consistency']:.3f})",
                'recommendation': "Monitor both customer forecasts and revenue per customer trends"
            })
        
        # High forecast errors
        if analysis['naive_mape'] > 40:
            restaurant_challenges.append({
                'type': 'High Forecast Errors',
                'severity': 'High' if analysis['naive_mape'] > 60 else 'Medium',
                'description': f"High baseline forecast errors (Naive MAPE: {analysis['naive_mape']:.1f}%)",
                'recommendation': "Invest in advanced forecasting methods and more frequent model updates"
            })
        
        challenges[restaurant_type] = restaurant_challenges
        
        # Display challenges
        if restaurant_challenges:
            for i, challenge in enumerate(restaurant_challenges, 1):
                severity_emoji = "🔥" if challenge['severity'] == 'High' else "⚠️" if challenge['severity'] == 'Medium' else "📊"
                print(f"   {i}. {severity_emoji} {challenge['type']} ({challenge['severity']} Severity)")
                print(f"      Issue: {challenge['description']}")
                print(f"      Action: {challenge['recommendation']}")
                print()
        else:
            print("   ✅ No major forecasting challenges identified - relatively predictable demand patterns")
        
        # Overall difficulty assessment
        difficulty_score = analysis['composite_difficulty']
        if difficulty_score > 0.7:
            overall_assessment = "CRITICAL - Requires sophisticated forecasting approach"
        elif difficulty_score > 0.5:
            overall_assessment = "HIGH - Needs careful attention and adaptive methods"
        elif difficulty_score > 0.3:
            overall_assessment = "MODERATE - Standard forecasting methods with some enhancements"
        else:
            overall_assessment = "LOW - Simple forecasting methods should suffice"
        
        print(f"   🎯 OVERALL ASSESSMENT: {overall_assessment}")
    
    return challenges

def create_restaurant_difficulty_visualizations(df, restaurant_analysis, comparison_df, challenges):
    """Create comprehensive visualizations for restaurant type forecast difficulty analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Forecast Difficulty Analysis by Restaurant Type', fontsize=16, y=0.98)
    
    # Visualization 1: Difficulty Score Comparison
    restaurants = comparison_df['Restaurant_Type'].tolist()
    difficulty_scores = comparison_df['Difficulty_Score'].tolist()
    
    # Color code by difficulty level
    colors = []
    for score in difficulty_scores:
        if score > 0.7:
            colors.append('#d62728')  # Red - Very hard
        elif score > 0.5:
            colors.append('#ff7f0e')  # Orange - Hard  
        elif score > 0.3:
            colors.append('#2ca02c')  # Green - Moderate
        else:
            colors.append('#1f77b4')  # Blue - Easy
    
    bars = axes[0,0].barh(range(len(restaurants)), difficulty_scores, color=colors)
    axes[0,0].set_yticks(range(len(restaurants)))
    axes[0,0].set_yticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=11)
    axes[0,0].set_xlabel('Composite Difficulty Score', fontsize=12)
    axes[0,0].set_title('Restaurant Type Forecast Difficulty Ranking', fontsize=13, pad=15)
    axes[0,0].grid(axis='x', alpha=0.3)
    axes[0,0].set_xlim(0, 1.0)
    
    # Add score labels and difficulty thresholds
    for i, (bar, score) in enumerate(zip(bars, difficulty_scores)):
        axes[0,0].text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                      va='center', fontsize=10, weight='bold')
    
    axes[0,0].axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Moderate')
    axes[0,0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Difficult')
    axes[0,0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Very Difficult')
    axes[0,0].legend(loc='lower right', fontsize=10)
    
    # Visualization 2: Multi-dimensional difficulty analysis
    # Radar chart showing different difficulty components
    categories = ['Volatility', 'Predictability', 'Consistency', 'External\nSensitivity', 'Peak\nIrregularity']
    
    # Prepare data for radar chart
    restaurant_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Use polar subplot
    ax_radar = plt.subplot(2, 2, 2, projection='polar')
    
    for i, (restaurant, analysis) in enumerate(list(restaurant_analysis.items())[:5]):  # Top 5 restaurants
        values = [
            analysis['difficulty_components']['volatility'],
            1 - analysis['autocorr_predictability'],  # Invert for difficulty
            1 - ((analysis['hourly_consistency'] + analysis['daily_consistency']) / 2),  # Invert
            analysis['avg_external_sensitivity'],
            analysis['difficulty_components']['irregularity']
        ]
        values += values[:1]  # Complete the circle
        
        color = restaurant_colors[i % len(restaurant_colors)]
        ax_radar.plot(angles, values, color=color, linewidth=2, 
                     label=restaurant.replace(' ', '\n'), alpha=0.8)
        ax_radar.fill(angles, values, color=color, alpha=0.1)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Multi-Dimensional Difficulty Analysis', fontsize=13, pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax_radar.grid(True)
    
    # Visualization 3: Volatility vs Predictability Scatter
    volatility_scores = comparison_df['CV_Demand'].tolist()
    predictability_scores = comparison_df['Autocorr_Predictability'].tolist()
    external_sensitivity = comparison_df['External_Sensitivity'].tolist()
    difficulty_scores_norm = comparison_df['Difficulty_Score'].tolist()
    
    scatter = axes[1,0].scatter(volatility_scores, predictability_scores, 
                               s=[e*400 for e in external_sensitivity], 
                               c=difficulty_scores_norm, cmap='plasma', alpha=0.7)
    
    # Add restaurant labels
    for i, restaurant in enumerate(restaurants):
        axes[1,0].annotate(restaurant.replace(' ', '\n'), 
                          (volatility_scores[i], predictability_scores[i]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=9, ha='left')
    
    axes[1,0].set_xlabel('Demand Volatility (CV)', fontsize=12)
    axes[1,0].set_ylabel('Autocorrelation Predictability', fontsize=12)
    axes[1,0].set_title('Volatility vs Predictability by Restaurant Type', fontsize=13, pad=15)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add quadrant labels
    axes[1,0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].text(0.02, 0.98, 'Low Vol\nHigh Predict\n(EASY)', transform=axes[1,0].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                   fontsize=8, va='top')
    axes[1,0].text(0.98, 0.02, 'High Vol\nLow Predict\n(VERY HARD)', transform=axes[1,0].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                   fontsize=8, va='bottom', ha='right')
    
    # Add colorbar and size legend
    plt.colorbar(scatter, ax=axes[1,0], label='Difficulty Score')
    axes[1,0].text(0.98, 0.98, 'Bubble size =\nExternal Sensitivity', 
                   transform=axes[1,0].transAxes, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', ha='right')
    
    # Visualization 4: Forecast Error Comparison
    naive_mape = comparison_df['Naive_MAPE'].tolist()
    ma7_mape = comparison_df['MA7_MAPE'].tolist()
    
    x = np.arange(len(restaurants))
    width = 0.35
    
    bars1 = axes[1,1].bar(x - width/2, naive_mape, width, label='Naive Forecast MAPE', alpha=0.8, color='lightcoral')
    bars2 = axes[1,1].bar(x + width/2, ma7_mape, width, label='7-Day MA MAPE', alpha=0.8, color='lightblue')
    
    axes[1,1].set_xlabel('Restaurant Type', fontsize=12)
    axes[1,1].set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    axes[1,1].set_title('Forecast Error Comparison by Restaurant Type', fontsize=13, pad=15)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=10, rotation=45)
    axes[1,1].legend(fontsize=10)
    axes[1,1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1,1].annotate(f'{height:.0f}%',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('forecast_difficulty_restaurant_type_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: forecast_difficulty_restaurant_type_analysis.png")

def main():
    """Main function for forecast difficulty by restaurant type analysis."""
    print("="*80)
    print("STINT PART 1: FORECAST DIFFICULTY BY RESTAURANT TYPE ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Analyze forecast difficulty by restaurant type
    restaurant_analysis = analyze_forecast_difficulty_by_type(df)
    
    # Compare restaurant types
    comparison_df = compare_restaurant_types(restaurant_analysis)
    
    # Identify specific challenges
    challenges = identify_restaurant_specific_challenges(restaurant_analysis)
    
    # Create visualizations
    create_restaurant_difficulty_visualizations(df, restaurant_analysis, comparison_df, challenges)
    
    # Final summary and recommendations
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY & STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    # Key findings
    hardest_restaurant = comparison_df.iloc[0]
    easiest_restaurant = comparison_df.iloc[-1]
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"• Hardest to forecast: {hardest_restaurant['Restaurant_Type'].title()} (Score: {hardest_restaurant['Difficulty_Score']:.3f})")
    print(f"• Easiest to forecast: {easiest_restaurant['Restaurant_Type'].title()} (Score: {easiest_restaurant['Difficulty_Score']:.3f})")
    print(f"• Average difficulty across types: {comparison_df['Difficulty_Score'].mean():.3f}")
    print(f"• Difficulty range: {comparison_df['Difficulty_Score'].std():.3f} standard deviation")
    
    # Strategic recommendations
    print(f"\n🚀 STRATEGIC RECOMMENDATIONS:")
    print("1. TIERED FORECASTING APPROACH:")
    high_difficulty = comparison_df[comparison_df['Difficulty_Score'] > 0.5]['Restaurant_Type'].tolist()
    low_difficulty = comparison_df[comparison_df['Difficulty_Score'] <= 0.3]['Restaurant_Type'].tolist()
    
    if high_difficulty:
        print(f"   • High-difficulty types ({', '.join([r.title() for r in high_difficulty])}): Advanced ML models, real-time updates")
    if low_difficulty:
        print(f"   • Low-difficulty types ({', '.join([r.title() for r in low_difficulty])}): Simple statistical models suffice")
    
    print("2. SPECIALIZED MONITORING:")
    high_external_sens = comparison_df[comparison_df['External_Sensitivity'] > 0.4]['Restaurant_Type'].tolist()
    if high_external_sens:
        print(f"   • Weather/event monitoring critical for: {', '.join([r.title() for r in high_external_sens])}")
    
    print("3. RESOURCE ALLOCATION:")
    print(f"   • Focus data science resources on highest difficulty types")
    print(f"   • Implement automated alerts for volatile periods")
    print(f"   • Develop restaurant-type specific forecasting models")
    
    # Save comprehensive results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'restaurant_analysis': restaurant_analysis,
        'comparison_summary': comparison_df.to_dict('records'),
        'challenges': challenges,
        'key_insights': {
            'hardest_restaurant': hardest_restaurant['Restaurant_Type'],
            'hardest_score': float(hardest_restaurant['Difficulty_Score']),
            'easiest_restaurant': easiest_restaurant['Restaurant_Type'],
            'easiest_score': float(easiest_restaurant['Difficulty_Score']),
            'average_difficulty': float(comparison_df['Difficulty_Score'].mean()),
            'difficulty_range': float(comparison_df['Difficulty_Score'].std())
        },
        'strategic_recommendations': [
            "Implement tiered forecasting approach based on difficulty scores",
            "Develop restaurant-type specific models and parameters", 
            "Allocate more resources to high-difficulty restaurant types",
            "Implement specialized monitoring for external-factor sensitive types",
            "Use simpler methods for consistently predictable restaurant types"
        ]
    }
    
    with open('forecast_difficulty_restaurant_type_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("📊 Results saved to:")
    print("  • forecast_difficulty_restaurant_type_results.json")
    print("  • forecast_difficulty_restaurant_type_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()