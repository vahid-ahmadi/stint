"""
Stint Data Science Technical Task - Part 1: Hardest-to-Predict Periods Analysis
Identify the hardest-to-predict periods and explain why
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

def load_and_prepare_data():
    """Load and prepare the restaurant demand dataset."""
    print("Loading restaurant demand data for prediction difficulty analysis...")
    df = pd.read_csv('ds_task_dataset.csv')
    
    df = df.dropna(subset=['restaurant_type'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create comprehensive features for volatility analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
    
    # Calculate customer count
    df['customer_count'] = df['main_meal_count'].fillna(0) * 1.2
    
    # Create time-based features that affect predictability
    df['is_lunch_rush'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_dinner_rush'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_rush_period'] = (df['is_lunch_rush'] | df['is_dinner_rush']).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def calculate_prediction_difficulty_metrics(df):
    """Calculate various metrics that indicate prediction difficulty."""
    
    print("\n" + "="*75)
    print("PREDICTION DIFFICULTY METRICS CALCULATION")
    print("="*75)
    
    difficulty_metrics = {}
    
    # Sample for computational efficiency while maintaining statistical validity
    df_sample = df.sample(n=min(40000, len(df)), random_state=42)
    
    for restaurant_type in df_sample['restaurant_type'].unique():
        print(f"\n🏪 Analyzing {restaurant_type.upper()}...")
        
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant_type].copy()
        
        if len(rest_data) < 200:
            print(f"   ⚠️  Insufficient data ({len(rest_data)} records)")
            continue
        
        # Sort by timestamp for time series analysis
        rest_data = rest_data.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Volatility Measures
        overall_cv = rest_data['customer_count'].std() / rest_data['customer_count'].mean()
        
        # Rolling volatility (24-hour window = 48 periods)
        rest_data['rolling_mean'] = rest_data['customer_count'].rolling(window=48, min_periods=24).mean()
        rest_data['rolling_std'] = rest_data['customer_count'].rolling(window=48, min_periods=24).std()
        rest_data['rolling_cv'] = rest_data['rolling_std'] / rest_data['rolling_mean']
        
        avg_rolling_cv = rest_data['rolling_cv'].mean()
        max_rolling_cv = rest_data['rolling_cv'].max()
        
        # 2. Demand Variability by Time Periods
        hourly_volatility = rest_data.groupby('hour')['customer_count'].std().mean()
        daily_volatility = rest_data.groupby('day_of_week')['customer_count'].std().mean()
        monthly_volatility = rest_data.groupby('month')['customer_count'].std().mean()
        
        # 3. External Factor Impact Volatility
        # High external factor periods vs normal periods
        high_event_periods = rest_data[rest_data['local_event'] > rest_data['local_event'].quantile(0.8)]
        normal_periods = rest_data[rest_data['local_event'] <= rest_data['local_event'].quantile(0.5)]
        
        event_volatility = high_event_periods['customer_count'].std() if len(high_event_periods) > 10 else 0
        normal_volatility = normal_periods['customer_count'].std() if len(normal_periods) > 10 else 0
        event_volatility_ratio = event_volatility / normal_volatility if normal_volatility > 0 else 1
        
        # Weather volatility impact
        extreme_weather = rest_data[
            (rest_data['temperature'] > rest_data['temperature'].quantile(0.9)) |
            (rest_data['temperature'] < rest_data['temperature'].quantile(0.1)) |
            (rest_data['precipitation'] > rest_data['precipitation'].quantile(0.9))
        ]
        mild_weather = rest_data[
            (rest_data['temperature'] >= rest_data['temperature'].quantile(0.3)) &
            (rest_data['temperature'] <= rest_data['temperature'].quantile(0.7)) &
            (rest_data['precipitation'] <= rest_data['precipitation'].quantile(0.5))
        ]
        
        weather_volatility_ratio = (extreme_weather['customer_count'].std() / 
                                  mild_weather['customer_count'].std()) if len(mild_weather) > 10 and len(extreme_weather) > 10 else 1
        
        # 4. Autocorrelation Analysis (predictability from past values)
        # Calculate lag-1 autocorrelation
        autocorr_lag1 = rest_data['customer_count'].autocorr(lag=1)
        autocorr_lag24 = rest_data['customer_count'].autocorr(lag=48) if len(rest_data) > 48 else 0  # Same time next day
        
        # Lower autocorrelation means harder to predict
        predictability_from_past = max(abs(autocorr_lag1), abs(autocorr_lag24))
        
        # 5. Trend Changes (regime changes make prediction harder)
        # Calculate first differences to detect trend changes
        rest_data['demand_diff'] = rest_data['customer_count'].diff()
        rest_data['trend_change'] = (rest_data['demand_diff'] * rest_data['demand_diff'].shift(1) < 0).astype(int)
        trend_change_frequency = rest_data['trend_change'].mean()
        
        # 6. Peak Detection Irregularity
        demand_series = rest_data['customer_count'].values
        peaks, _ = find_peaks(demand_series, height=np.percentile(demand_series, 70))
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            peak_regularity = 1 - (np.std(peak_intervals) / np.mean(peak_intervals)) if np.mean(peak_intervals) > 0 else 0
            peak_regularity = max(0, peak_regularity)  # Ensure non-negative
        else:
            peak_regularity = 0
        
        # 7. External Factor Sensitivity
        # Correlation with external factors indicates susceptibility to external shocks
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
        
        # 8. Composite Difficulty Score
        # Higher score = harder to predict
        difficulty_components = {
            'volatility': min(overall_cv / 0.5, 1.0),  # Normalize to 0-1 scale
            'rolling_volatility': min(avg_rolling_cv / 0.5, 1.0),
            'external_sensitivity': avg_external_sensitivity,
            'trend_instability': trend_change_frequency,
            'peak_irregularity': 1 - peak_regularity,
            'low_autocorr': 1 - predictability_from_past,
            'event_impact': min((event_volatility_ratio - 1) / 2, 1.0),
            'weather_impact': min((weather_volatility_ratio - 1) / 2, 1.0)
        }
        
        # Weight the components
        weights = {
            'volatility': 0.2,
            'rolling_volatility': 0.15,
            'external_sensitivity': 0.15,
            'trend_instability': 0.15,
            'peak_irregularity': 0.1,
            'low_autocorr': 0.1,
            'event_impact': 0.075,
            'weather_impact': 0.075
        }
        
        composite_difficulty = sum(weights[comp] * score for comp, score in difficulty_components.items())
        
        # Store all metrics
        difficulty_metrics[restaurant_type] = {
            'overall_cv': round(overall_cv, 3),
            'avg_rolling_cv': round(avg_rolling_cv, 3),
            'max_rolling_cv': round(max_rolling_cv, 3),
            'autocorr_lag1': round(autocorr_lag1, 3),
            'autocorr_lag24': round(autocorr_lag24, 3),
            'trend_change_frequency': round(trend_change_frequency, 3),
            'peak_regularity': round(peak_regularity, 3),
            'external_sensitivity_avg': round(avg_external_sensitivity, 3),
            'external_sensitivity_max': round(max_external_sensitivity, 3),
            'event_volatility_ratio': round(event_volatility_ratio, 3),
            'weather_volatility_ratio': round(weather_volatility_ratio, 3),
            'difficulty_components': {k: round(v, 3) for k, v in difficulty_components.items()},
            'composite_difficulty_score': round(composite_difficulty, 3),
            'hourly_volatility': round(hourly_volatility, 1),
            'daily_volatility': round(daily_volatility, 1),
            'monthly_volatility': round(monthly_volatility, 1)
        }
        
        print(f"   📊 Composite Difficulty Score: {composite_difficulty:.3f}/1.000")
        print(f"   🎯 Overall CV: {overall_cv:.3f}")
        print(f"   📈 Autocorr (Lag-1): {autocorr_lag1:.3f}")
        print(f"   🌟 External Sensitivity: {avg_external_sensitivity:.3f}")
    
    return difficulty_metrics

def identify_hardest_predict_conditions(df, difficulty_metrics):
    """Identify specific conditions that make prediction most difficult."""
    
    print("\n" + "="*75)
    print("IDENTIFYING HARDEST-TO-PREDICT CONDITIONS")
    print("="*75)
    
    hardest_conditions = {}
    
    # Sample for analysis
    df_sample = df.sample(n=min(30000, len(df)), random_state=42)
    
    # Define condition categories that typically make prediction harder
    condition_definitions = {
        'High Local Events': df_sample['local_event'] > df_sample['local_event'].quantile(0.85),
        'Viral Social Media': df_sample['social_trend'] > df_sample['social_trend'].quantile(0.9),
        'Extreme Weather': (
            (df_sample['temperature'] > df_sample['temperature'].quantile(0.95)) |
            (df_sample['temperature'] < df_sample['temperature'].quantile(0.05)) |
            (df_sample['precipitation'] > df_sample['precipitation'].quantile(0.9))
        ),
        'High Competition': df_sample['competitor_promo'] > df_sample['competitor_promo'].quantile(0.8),
        'Economic Volatility': (
            (df_sample['economic_indicator'] > df_sample['economic_indicator'].quantile(0.9)) |
            (df_sample['economic_indicator'] < df_sample['economic_indicator'].quantile(0.1))
        ),
        'Weekend Rush': (df_sample['is_weekend'] == 1) & (df_sample['is_rush_period'] == 1),
        'Holiday Season': df_sample['is_holiday_season'] == 1,
        'Late Night': df_sample['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]),
        'Multiple Factors': (
            (df_sample['local_event'] > df_sample['local_event'].quantile(0.7)) &
            (df_sample['social_trend'] > df_sample['social_trend'].quantile(0.7)) &
            (df_sample['is_weekend'] == 1)
        )
    }
    
    # Analyze each condition
    for condition_name, condition_mask in condition_definitions.items():
        print(f"\n🔍 Analyzing: {condition_name}")
        print("-" * 50)
        
        condition_data = df_sample[condition_mask]
        normal_data = df_sample[~condition_mask]
        
        if len(condition_data) < 50:
            print(f"   ⚠️  Insufficient data for {condition_name} ({len(condition_data)} records)")
            continue
        
        # Calculate volatility under this condition vs normal
        condition_volatility = condition_data.groupby('restaurant_type')['customer_count'].std()
        normal_volatility = normal_data.groupby('restaurant_type')['customer_count'].std()
        
        # Calculate mean absolute deviation from expected
        condition_mean_demand = condition_data.groupby('restaurant_type')['customer_count'].mean()
        normal_mean_demand = normal_data.groupby('restaurant_type')['customer_count'].mean()
        
        condition_analysis = {}
        
        for restaurant_type in df_sample['restaurant_type'].unique():
            if (restaurant_type in condition_volatility.index and 
                restaurant_type in normal_volatility.index and
                normal_volatility[restaurant_type] > 0):
                
                volatility_ratio = condition_volatility[restaurant_type] / normal_volatility[restaurant_type]
                
                # Calculate demand predictability deviation
                if restaurant_type in condition_mean_demand.index and restaurant_type in normal_mean_demand.index:
                    demand_deviation = abs(condition_mean_demand[restaurant_type] - 
                                         normal_mean_demand[restaurant_type]) / normal_mean_demand[restaurant_type]
                else:
                    demand_deviation = 0
                
                # Combine metrics for overall difficulty
                condition_difficulty = (volatility_ratio + demand_deviation) / 2
                
                condition_analysis[restaurant_type] = {
                    'volatility_ratio': round(volatility_ratio, 2),
                    'demand_deviation': round(demand_deviation, 2),
                    'condition_difficulty': round(condition_difficulty, 2),
                    'sample_size': len(condition_data[condition_data['restaurant_type'] == restaurant_type])
                }
                
                print(f"   • {restaurant_type.title()}: "
                      f"Volatility {volatility_ratio:.2f}x, "
                      f"Demand Dev {demand_deviation:.1%}, "
                      f"Difficulty {condition_difficulty:.2f}")
        
        # Calculate overall condition impact
        if condition_analysis:
            avg_difficulty = np.mean([a['condition_difficulty'] for a in condition_analysis.values()])
            max_difficulty = np.max([a['condition_difficulty'] for a in condition_analysis.values()])
            
            hardest_conditions[condition_name] = {
                'restaurants': condition_analysis,
                'avg_difficulty': round(avg_difficulty, 3),
                'max_difficulty': round(max_difficulty, 3),
                'total_occurrences': len(condition_data),
                'percentage_of_data': round(len(condition_data) / len(df_sample) * 100, 1)
            }
            
            print(f"   📊 Average Difficulty: {avg_difficulty:.3f}")
            print(f"   🚨 Max Difficulty: {max_difficulty:.3f}")
            print(f"   📈 Occurrence: {len(condition_data)} records ({(len(condition_data)/len(df_sample)*100):.1f}%)")
    
    return hardest_conditions

def explain_prediction_difficulties(difficulty_metrics, hardest_conditions):
    """Provide detailed explanations for why certain periods are hard to predict."""
    
    print("\n" + "="*75)
    print("EXPLAINING PREDICTION DIFFICULTIES")
    print("="*75)
    
    explanations = {}
    
    # Rank restaurants by difficulty
    restaurant_difficulty_ranking = sorted(
        difficulty_metrics.items(), 
        key=lambda x: x[1]['composite_difficulty_score'], 
        reverse=True
    )
    
    print("\n🎯 RESTAURANT PREDICTABILITY RANKING (Hardest to Easiest):")
    print("-" * 60)
    
    for i, (restaurant, metrics) in enumerate(restaurant_difficulty_ranking, 1):
        score = metrics['composite_difficulty_score']
        
        # Determine difficulty level
        if score > 0.7:
            difficulty_level = "EXTREMELY DIFFICULT"
            emoji = "🔥"
        elif score > 0.5:
            difficulty_level = "DIFFICULT"
            emoji = "⚠️"
        elif score > 0.3:
            difficulty_level = "MODERATE"
            emoji = "📊"
        else:
            difficulty_level = "RELATIVELY EASY"
            emoji = "✅"
        
        print(f"{i}. {emoji} {restaurant.upper()}: {score:.3f} - {difficulty_level}")
        
        # Identify top contributing factors
        components = metrics['difficulty_components']
        top_factors = sorted(components.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"   Top Difficulty Factors:")
        for factor, value in top_factors:
            factor_name = factor.replace('_', ' ').title()
            print(f"     • {factor_name}: {value:.3f}")
        
        # Create explanation based on metrics
        explanation_parts = []
        
        if metrics['overall_cv'] > 0.5:
            explanation_parts.append(f"High demand volatility (CV: {metrics['overall_cv']:.2f})")
        
        if metrics['autocorr_lag1'] < 0.3:
            explanation_parts.append(f"Low autocorrelation ({metrics['autocorr_lag1']:.2f}) - past demand doesn't predict future well")
        
        if metrics['external_sensitivity_avg'] > 0.4:
            explanation_parts.append(f"High sensitivity to external factors ({metrics['external_sensitivity_avg']:.2f})")
        
        if metrics['trend_change_frequency'] > 0.3:
            explanation_parts.append(f"Frequent trend changes ({metrics['trend_change_frequency']:.2f})")
        
        if metrics['event_volatility_ratio'] > 1.5:
            explanation_parts.append(f"Events cause {metrics['event_volatility_ratio']:.1f}x more volatility")
        
        if metrics['weather_volatility_ratio'] > 1.5:
            explanation_parts.append(f"Weather creates {metrics['weather_volatility_ratio']:.1f}x more volatility")
        
        explanations[restaurant] = {
            'difficulty_score': score,
            'difficulty_level': difficulty_level,
            'main_factors': [f[0] for f in top_factors],
            'explanation_text': "; ".join(explanation_parts) if explanation_parts else "Relatively stable demand patterns",
            'metrics': metrics
        }
        
        print(f"   Why it's difficult: {explanations[restaurant]['explanation_text']}")
        print()
    
    # Analyze hardest conditions
    print("\n🔍 MOST DIFFICULT CONDITIONS TO PREDICT:")
    print("-" * 60)
    
    # Rank conditions by difficulty
    condition_ranking = sorted(
        hardest_conditions.items(),
        key=lambda x: x[1]['avg_difficulty'],
        reverse=True
    )
    
    for i, (condition, analysis) in enumerate(condition_ranking[:7], 1):  # Top 7 conditions
        print(f"{i}. {condition.upper()}")
        print(f"   Average Difficulty: {analysis['avg_difficulty']:.3f}")
        print(f"   Occurrence Rate: {analysis['percentage_of_data']:.1f}% of periods")
        print(f"   Total Impact: {analysis['total_occurrences']} periods")
        
        # Explain why this condition is difficult
        condition_explanations = {
            'High Local Events': "Unpredictable crowd sizes and timing create demand spikes",
            'Viral Social Media': "Social media buzz creates sudden, hard-to-forecast demand surges",
            'Extreme Weather': "Weather extremes cause non-linear behavioral changes in dining patterns",
            'High Competition': "Competitor promotions create unpredictable customer shifts",
            'Economic Volatility': "Economic uncertainty affects consumer spending unpredictably",
            'Weekend Rush': "Weekend rush periods combine multiple unpredictable factors",
            'Holiday Season': "Holiday periods have irregular patterns and external influences",
            'Late Night': "Late night demand depends on events, weather, and social activities",
            'Multiple Factors': "Multiple simultaneous factors create compounding unpredictability"
        }
        
        if condition in condition_explanations:
            print(f"   Why difficult: {condition_explanations[condition]}")
        
        print()
    
    return explanations

def create_prediction_difficulty_visualizations(df, difficulty_metrics, hardest_conditions, explanations):
    """Create visualizations showing prediction difficulty analysis."""
    
    df_viz = df.sample(n=min(25000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Hardest-to-Predict Periods Analysis - Restaurant Demand Forecasting', fontsize=16, y=0.98)
    
    # Visualization 1: Difficulty Score Comparison
    restaurants = list(difficulty_metrics.keys())
    difficulty_scores = [difficulty_metrics[r]['composite_difficulty_score'] for r in restaurants]
    
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
    axes[0,0].set_yticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=10)
    axes[0,0].set_xlabel('Composite Difficulty Score', fontsize=12)
    axes[0,0].set_title('Restaurant Prediction Difficulty Ranking', fontsize=13, pad=15)
    axes[0,0].grid(axis='x', alpha=0.3)
    axes[0,0].set_xlim(0, 1.0)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, difficulty_scores)):
        axes[0,0].text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                      va='center', fontsize=9, weight='bold')
    
    # Add difficulty thresholds
    axes[0,0].axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Moderate')
    axes[0,0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Difficult')
    axes[0,0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Very Difficult')
    axes[0,0].legend(loc='lower right', fontsize=9)
    
    # Visualization 2: Volatility vs Predictability Scatter
    volatility_scores = [difficulty_metrics[r]['overall_cv'] for r in restaurants]
    autocorr_scores = [difficulty_metrics[r]['autocorr_lag1'] for r in restaurants]
    external_sensitivity = [difficulty_metrics[r]['external_sensitivity_avg'] for r in restaurants]
    
    # Create scatter plot
    scatter = axes[0,1].scatter(volatility_scores, autocorr_scores, 
                               s=[e*300 for e in external_sensitivity], 
                               c=difficulty_scores, cmap='plasma', alpha=0.7)
    
    # Add restaurant labels
    for i, restaurant in enumerate(restaurants):
        axes[0,1].annotate(restaurant.replace(' ', '\n'), 
                          (volatility_scores[i], autocorr_scores[i]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=8, ha='left')
    
    axes[0,1].set_xlabel('Demand Volatility (CV)', fontsize=12)
    axes[0,1].set_ylabel('Autocorrelation (Predictability)', fontsize=12)
    axes[0,1].set_title('Volatility vs Predictability Analysis', fontsize=13, pad=15)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add colorbar and size legend
    plt.colorbar(scatter, ax=axes[0,1], label='Difficulty Score')
    axes[0,1].text(0.02, 0.98, 'Bubble size = External Sensitivity', 
                   transform=axes[0,1].transAxes, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
    
    # Visualization 3: Difficult Conditions Heatmap
    condition_names = list(hardest_conditions.keys())[:6]  # Top 6 conditions
    condition_matrix = []
    restaurant_labels = []
    
    for restaurant in restaurants:
        row = []
        for condition in condition_names:
            if (condition in hardest_conditions and 
                restaurant in hardest_conditions[condition]['restaurants']):
                difficulty = hardest_conditions[condition]['restaurants'][restaurant]['condition_difficulty']
            else:
                difficulty = 0
            row.append(difficulty)
        condition_matrix.append(row)
        restaurant_labels.append(restaurant.replace(' ', '\n'))
    
    condition_matrix = np.array(condition_matrix)
    
    im = axes[1,0].imshow(condition_matrix, cmap='Reds', aspect='auto')
    
    axes[1,0].set_xticks(range(len(condition_names)))
    axes[1,0].set_xticklabels([name.replace(' ', '\n') for name in condition_names], 
                              rotation=45, ha='right', fontsize=9)
    axes[1,0].set_yticks(range(len(restaurant_labels)))
    axes[1,0].set_yticklabels(restaurant_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(restaurant_labels)):
        for j in range(len(condition_names)):
            if condition_matrix[i, j] > 0:
                text_color = "white" if condition_matrix[i, j] > np.mean(condition_matrix[condition_matrix > 0]) else "black"
                axes[1,0].text(j, i, f'{condition_matrix[i, j]:.2f}',
                              ha="center", va="center", color=text_color, fontsize=8)
    
    axes[1,0].set_title('Difficulty by Specific Conditions', fontsize=13, pad=15)
    plt.colorbar(im, ax=axes[1,0], label='Condition Difficulty Score')
    
    # Visualization 4: Time-based Difficulty Patterns
    # Calculate hourly difficulty patterns
    hourly_difficulty = {}
    
    for hour in range(24):
        hour_data = df_viz[df_viz['hour'] == hour]
        if len(hour_data) > 100:
            # Calculate volatility for this hour across all restaurants
            hour_volatility = hour_data.groupby('restaurant_type')['customer_count'].std().mean()
            hourly_difficulty[hour] = hour_volatility
        else:
            hourly_difficulty[hour] = 0
    
    hours = list(hourly_difficulty.keys())
    difficulties = list(hourly_difficulty.values())
    
    axes[1,1].plot(hours, difficulties, marker='o', linewidth=2.5, markersize=6, color='crimson')
    axes[1,1].fill_between(hours, difficulties, alpha=0.3, color='crimson')
    
    # Highlight peak difficulty periods
    max_difficulty_hour = max(hourly_difficulty, key=hourly_difficulty.get)
    axes[1,1].axvline(x=max_difficulty_hour, color='red', linestyle='--', alpha=0.7, 
                      label=f'Peak Difficulty: {max_difficulty_hour:02d}:00')
    
    # Mark rush periods
    axes[1,1].axvspan(11, 14, alpha=0.2, color='orange', label='Lunch Rush')
    axes[1,1].axvspan(18, 22, alpha=0.2, color='purple', label='Dinner Rush')
    
    axes[1,1].set_xlabel('Hour of Day', fontsize=12)
    axes[1,1].set_ylabel('Average Volatility', fontsize=12)
    axes[1,1].set_title('Prediction Difficulty by Hour', fontsize=13, pad=15)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(loc='upper right', fontsize=9)
    axes[1,1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('hardest_predict_periods_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: hardest_predict_periods_analysis.png")

def main():
    """Main function for hardest-to-predict periods analysis."""
    print("="*75)
    print("STINT PART 1: HARDEST-TO-PREDICT PERIODS ANALYSIS")
    print("="*75)
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate prediction difficulty metrics
    difficulty_metrics = calculate_prediction_difficulty_metrics(df)
    
    # Identify hardest conditions
    hardest_conditions = identify_hardest_predict_conditions(df, difficulty_metrics)
    
    # Explain difficulties
    explanations = explain_prediction_difficulties(difficulty_metrics, hardest_conditions)
    
    # Create visualizations
    create_prediction_difficulty_visualizations(df, difficulty_metrics, hardest_conditions, explanations)
    
    # Final summary
    print("\n" + "="*75)
    print("KEY FINDINGS SUMMARY")
    print("="*75)
    
    # Find hardest restaurant to predict
    hardest_restaurant = max(difficulty_metrics.items(), key=lambda x: x[1]['composite_difficulty_score'])
    easiest_restaurant = min(difficulty_metrics.items(), key=lambda x: x[1]['composite_difficulty_score'])
    
    print(f"\n🔥 HARDEST TO PREDICT: {hardest_restaurant[0].title()}")
    print(f"   Difficulty Score: {hardest_restaurant[1]['composite_difficulty_score']:.3f}")
    print(f"   Main Issues: {explanations[hardest_restaurant[0]]['explanation_text']}")
    
    print(f"\n✅ EASIEST TO PREDICT: {easiest_restaurant[0].title()}")
    print(f"   Difficulty Score: {easiest_restaurant[1]['composite_difficulty_score']:.3f}")
    print(f"   Characteristics: {explanations[easiest_restaurant[0]]['explanation_text']}")
    
    # Top difficult conditions
    top_conditions = sorted(hardest_conditions.items(), key=lambda x: x[1]['avg_difficulty'], reverse=True)[:3]
    print(f"\n🚨 TOP 3 MOST DIFFICULT CONDITIONS:")
    for i, (condition, analysis) in enumerate(top_conditions, 1):
        print(f"   {i}. {condition}: {analysis['avg_difficulty']:.3f} avg difficulty")
    
    # Save results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'difficulty_metrics': difficulty_metrics,
        'hardest_conditions': hardest_conditions,
        'explanations': explanations,
        'summary': {
            'hardest_restaurant': hardest_restaurant[0],
            'hardest_score': hardest_restaurant[1]['composite_difficulty_score'],
            'easiest_restaurant': easiest_restaurant[0],
            'easiest_score': easiest_restaurant[1]['composite_difficulty_score'],
            'top_difficult_conditions': [cond for cond, _ in top_conditions]
        }
    }
    
    with open('hardest_predict_periods_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*75)
    print("ANALYSIS COMPLETE")
    print("="*75)
    print("📊 Results saved to:")
    print("  • hardest_predict_periods_results.json")
    print("  • hardest_predict_periods_analysis.png")
    print("="*75)

if __name__ == "__main__":
    main()