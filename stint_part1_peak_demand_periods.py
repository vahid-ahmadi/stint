"""
Stint Data Science Technical Task - Part 1: Peak Demand Periods Analysis
Derive and characterize peak demand periods from the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("rocket")

def load_and_prepare_data():
    """Load and prepare the restaurant demand dataset."""
    print("Loading restaurant demand data for peak periods analysis...")
    df = pd.read_csv('ds_task_dataset.csv')
    
    df = df.dropna(subset=['restaurant_type'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create comprehensive time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)  # Holiday season
    
    # Calculate customer count
    df['customer_count'] = df['main_meal_count'].fillna(0) * 1.2
    
    # Create time periods
    df['time_period'] = pd.cut(df['hour'], 
                               bins=[0, 6, 11, 14, 18, 21, 24],
                               labels=['Late Night', 'Morning', 'Lunch', 'Afternoon', 'Dinner', 'Evening'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def derive_peak_demand_periods(df):
    """Identify and characterize peak demand periods comprehensively."""
    
    print("\n" + "="*70)
    print("PEAK DEMAND PERIODS ANALYSIS")
    print("="*70)
    
    peak_analysis = {}
    
    # Sample for performance while maintaining statistical significance
    df_sample = df.sample(n=min(30000, len(df)), random_state=42)
    
    for restaurant_type in df_sample['restaurant_type'].unique():
        print(f"\n🏪 {restaurant_type.upper()} RESTAURANTS:")
        print("-" * 50)
        
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant_type]
        
        if len(rest_data) < 100:
            print(f"   ⚠️  Insufficient data ({len(rest_data)} records)")
            continue
        
        # 1. Define peak periods using percentile thresholds
        demand_percentiles = rest_data['customer_count'].quantile([0.8, 0.9, 0.95, 0.99])
        
        rest_data['demand_tier'] = pd.cut(rest_data['customer_count'],
                                        bins=[-np.inf, demand_percentiles[0.8], 
                                              demand_percentiles[0.9], demand_percentiles[0.95], np.inf],
                                        labels=['Normal', 'High', 'Peak', 'Extreme'])
        
        # 2. Hourly peak analysis
        hourly_stats = rest_data.groupby('hour').agg({
            'customer_count': ['mean', 'std', 'max', 'count']
        }).round(2)
        hourly_stats.columns = ['mean_demand', 'std_demand', 'max_demand', 'count']
        hourly_stats['cv'] = hourly_stats['std_demand'] / hourly_stats['mean_demand']
        
        # Identify peak hours (top 20% of average demand)
        peak_hour_threshold = hourly_stats['mean_demand'].quantile(0.8)
        peak_hours = hourly_stats[hourly_stats['mean_demand'] >= peak_hour_threshold].index.tolist()
        
        # 3. Day of week analysis
        dow_stats = rest_data.groupby('day_of_week').agg({
            'customer_count': ['mean', 'std', 'max']
        }).round(2)
        dow_stats.columns = ['mean_demand', 'std_demand', 'max_demand']
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = dow_stats.reindex([day for day in day_order if day in dow_stats.index])
        
        peak_days = dow_stats.nlargest(3, 'mean_demand').index.tolist()
        
        # 4. Seasonal patterns
        monthly_stats = rest_data.groupby('month').agg({
            'customer_count': ['mean', 'std']
        }).round(2)
        monthly_stats.columns = ['mean_demand', 'std_demand']
        peak_months = monthly_stats.nlargest(3, 'mean_demand').index.tolist()
        
        # 5. Combined peak periods (hour + day combinations)
        combined_peaks = rest_data.groupby(['day_of_week', 'hour'])['customer_count'].mean()
        top_combinations = combined_peaks.nlargest(5)
        
        # 6. Peak duration analysis
        rest_data_sorted = rest_data.sort_values('timestamp')
        rest_data_sorted['is_peak_hour'] = rest_data_sorted['hour'].isin(peak_hours)
        
        # Calculate consecutive peak periods
        rest_data_sorted['peak_group'] = (rest_data_sorted['is_peak_hour'] != 
                                        rest_data_sorted['is_peak_hour'].shift()).cumsum()
        
        peak_durations = rest_data_sorted[rest_data_sorted['is_peak_hour']].groupby('peak_group').size()
        avg_peak_duration = peak_durations.mean() * 0.5  # Convert to hours (30-min intervals)
        
        # 7. Peak intensity analysis
        normal_avg = rest_data[rest_data['demand_tier'] == 'Normal']['customer_count'].mean()
        peak_avg = rest_data[rest_data['demand_tier'] == 'Peak']['customer_count'].mean()
        extreme_avg = rest_data[rest_data['demand_tier'] == 'Extreme']['customer_count'].mean()
        
        peak_intensity = peak_avg / normal_avg if normal_avg > 0 else 0
        extreme_intensity = extreme_avg / normal_avg if normal_avg > 0 else 0
        
        # Store analysis results
        peak_analysis[restaurant_type] = {
            'peak_hours': peak_hours,
            'peak_days': peak_days,
            'peak_months': peak_months,
            'top_time_combinations': [(combo[0], combo[1], value) for combo, value in top_combinations.items()],
            'avg_peak_duration_hours': round(avg_peak_duration, 1),
            'peak_intensity_multiplier': round(peak_intensity, 2),
            'extreme_intensity_multiplier': round(extreme_intensity, 2),
            'peak_frequency_pct': round((rest_data['demand_tier'] == 'Peak').sum() / len(rest_data) * 100, 1),
            'demand_statistics': {
                'normal_avg': round(normal_avg, 1),
                'peak_avg': round(peak_avg, 1),
                'extreme_avg': round(extreme_avg, 1),
                'overall_std': round(rest_data['customer_count'].std(), 1)
            }
        }
        
        # Print detailed analysis
        print(f"   📊 PEAK HOURS: {', '.join([f'{int(h):02d}:00' for h in sorted(peak_hours)])}")
        print(f"   📅 PEAK DAYS: {', '.join(peak_days)}")
        print(f"   🗓️  PEAK MONTHS: {', '.join([str(m) for m in peak_months])}")
        print(f"   ⏱️  AVERAGE PEAK DURATION: {avg_peak_duration:.1f} hours")
        print(f"   🚀 PEAK INTENSITY: {peak_intensity:.1f}x normal demand")
        print(f"   💥 EXTREME INTENSITY: {extreme_intensity:.1f}x normal demand")
        print(f"   📈 PEAK FREQUENCY: {((rest_data['demand_tier'] == 'Peak').sum() / len(rest_data) * 100):.1f}% of time periods")
        
        print(f"\n   🔥 TOP 3 PEAK TIME COMBINATIONS:")
        for i, ((day, hour), demand) in enumerate(list(top_combinations.items())[:3], 1):
            print(f"      {i}. {day} at {int(hour):02d}:00 - {demand:.1f} avg customers")
    
    return peak_analysis

def characterize_peak_patterns(df, peak_analysis):
    """Characterize different types of peak patterns."""
    
    print("\n" + "="*70)
    print("PEAK PATTERN CHARACTERIZATION")
    print("="*70)
    
    pattern_analysis = {}
    
    for restaurant_type, analysis in peak_analysis.items():
        print(f"\n🏪 {restaurant_type.upper()} - PEAK CHARACTERISTICS:")
        print("-" * 50)
        
        rest_data = df[df['restaurant_type'] == restaurant_type].sample(n=min(5000, len(df[df['restaurant_type'] == restaurant_type])))
        
        # 1. Peak type classification
        peak_hours = analysis['peak_hours']
        lunch_peaks = [h for h in peak_hours if 11 <= h <= 14]
        dinner_peaks = [h for h in peak_hours if 18 <= h <= 22]
        unusual_peaks = [h for h in peak_hours if h not in range(11, 15) and h not in range(18, 23)]
        
        # 2. Peak consistency analysis
        hourly_cv = rest_data.groupby('hour')['customer_count'].apply(lambda x: x.std() / x.mean()).mean()
        
        # 3. Weekend vs weekday peak patterns
        weekend_peaks = rest_data[rest_data['is_weekend'] == 1]['customer_count'].mean()
        weekday_peaks = rest_data[rest_data['is_weekend'] == 0]['customer_count'].mean()
        weekend_bias = weekend_peaks / weekday_peaks if weekday_peaks > 0 else 1
        
        # 4. Seasonal peak variation
        seasonal_cv = rest_data.groupby('month')['customer_count'].mean().std() / rest_data.groupby('month')['customer_count'].mean().mean()
        
        # 5. Peak pattern classification
        if len(lunch_peaks) > 0 and len(dinner_peaks) > 0:
            pattern_type = "Dual Peak (Lunch & Dinner)"
        elif len(lunch_peaks) > 0:
            pattern_type = "Lunch-Focused"
        elif len(dinner_peaks) > 0:
            pattern_type = "Dinner-Focused"
        else:
            pattern_type = "Irregular Pattern"
        
        # 6. Peak predictability score
        consistency_score = 1 - min(hourly_cv, 1.0)  # Lower CV = higher consistency
        seasonal_stability = 1 - min(seasonal_cv, 1.0)
        predictability_score = (consistency_score + seasonal_stability) / 2
        
        pattern_analysis[restaurant_type] = {
            'pattern_type': pattern_type,
            'lunch_peak_hours': lunch_peaks,
            'dinner_peak_hours': dinner_peaks,
            'unusual_peak_hours': unusual_peaks,
            'weekend_bias': round(weekend_bias, 2),
            'hourly_consistency': round(consistency_score, 2),
            'seasonal_stability': round(seasonal_stability, 2),
            'predictability_score': round(predictability_score, 2),
            'peak_spread': len(peak_hours)  # How spread out peaks are
        }
        
        print(f"   🎯 PATTERN TYPE: {pattern_type}")
        if lunch_peaks:
            print(f"   🍽️  LUNCH PEAKS: {', '.join([f'{int(h):02d}:00' for h in lunch_peaks])}")
        if dinner_peaks:
            print(f"   🍷 DINNER PEAKS: {', '.join([f'{int(h):02d}:00' for h in dinner_peaks])}")
        if unusual_peaks:
            print(f"   ❓ UNUSUAL PEAKS: {', '.join([f'{int(h):02d}:00' for h in unusual_peaks])}")
        print(f"   📊 WEEKEND BIAS: {weekend_bias:.1f}x weekday demand")
        print(f"   🎯 PREDICTABILITY SCORE: {predictability_score:.2f}/1.00")
        print(f"   📈 CONSISTENCY: {consistency_score:.2f}/1.00")
    
    return pattern_analysis

def create_peak_demand_visualizations(df, peak_analysis, pattern_analysis):
    """Create comprehensive visualizations for peak demand analysis."""
    
    df_viz = df.sample(n=min(20000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Peak Demand Periods Analysis - Restaurant Demand Forecasting', fontsize=16, y=0.98)
    
    # Visualization 1: Hourly demand patterns by restaurant type
    hourly_patterns = df_viz.groupby(['restaurant_type', 'hour'])['customer_count'].mean().unstack(0)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, col in enumerate(hourly_patterns.columns):
        if not hourly_patterns[col].isna().all():
            axes[0,0].plot(hourly_patterns.index, hourly_patterns[col], 
                          label=col.replace(' ', '\n'), linewidth=2.5, 
                          color=colors[i % len(colors)], marker='o', markersize=4)
    
    # Highlight peak periods
    axes[0,0].axvspan(11, 14, alpha=0.2, color='gold', label='Lunch Period')
    axes[0,0].axvspan(18, 22, alpha=0.2, color='orange', label='Dinner Period')
    
    axes[0,0].set_xlabel('Hour of Day', fontsize=12)
    axes[0,0].set_ylabel('Average Customer Count', fontsize=12)
    axes[0,0].set_title('Daily Peak Demand Patterns by Restaurant Type', fontsize=13, pad=15)
    axes[0,0].legend(loc='upper left', fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(range(6, 24, 2))
    
    # Visualization 2: Peak intensity heatmap
    peak_intensity_data = []
    for rest_type, analysis in peak_analysis.items():
        peak_intensity_data.append({
            'Restaurant': rest_type.replace(' ', '\n'),
            'Peak\nIntensity': analysis['peak_intensity_multiplier'],
            'Extreme\nIntensity': analysis['extreme_intensity_multiplier'],
            'Peak\nFrequency': analysis['peak_frequency_pct']
        })
    
    intensity_df = pd.DataFrame(peak_intensity_data).set_index('Restaurant')
    
    sns.heatmap(intensity_df.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=axes[0,1], cbar_kws={'label': 'Multiplier / Percentage'})
    axes[0,1].set_title('Peak Intensity Characteristics', fontsize=13, pad=15)
    axes[0,1].set_xlabel('Restaurant Type', fontsize=12)
    
    # Visualization 3: Peak duration and frequency analysis
    duration_data = []
    frequency_data = []
    restaurant_names = []
    
    for rest_type, analysis in peak_analysis.items():
        duration_data.append(analysis['avg_peak_duration_hours'])
        frequency_data.append(analysis['peak_frequency_pct'])
        restaurant_names.append(rest_type.replace(' ', '\n'))
    
    # Create scatter plot with size representing intensity
    intensity_sizes = [peak_analysis[rest]['peak_intensity_multiplier'] * 50 for rest in peak_analysis.keys()]
    
    scatter = axes[1,0].scatter(duration_data, frequency_data, s=intensity_sizes, 
                               c=range(len(duration_data)), cmap='viridis', alpha=0.7)
    
    # Add restaurant labels
    for i, name in enumerate(restaurant_names):
        axes[1,0].annotate(name, (duration_data[i], frequency_data[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1,0].set_xlabel('Average Peak Duration (Hours)', fontsize=12)
    axes[1,0].set_ylabel('Peak Frequency (%)', fontsize=12)
    axes[1,0].set_title('Peak Duration vs Frequency Analysis', fontsize=13, pad=15)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add size legend
    axes[1,0].text(0.02, 0.98, 'Bubble size = Peak Intensity', 
                   transform=axes[1,0].transAxes, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
    
    # Visualization 4: Day of week peak patterns
    dow_patterns = df_viz.groupby(['restaurant_type', 'day_of_week'])['customer_count'].mean()
    
    # Reorganize data for better visualization
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_matrix = []
    restaurant_labels = []
    
    for rest_type in df_viz['restaurant_type'].unique():
        if rest_type in dow_patterns.index:
            row = []
            for day in day_order:
                if (rest_type, day) in dow_patterns.index:
                    row.append(dow_patterns[rest_type, day])
                else:
                    row.append(np.nan)
            dow_matrix.append(row)
            restaurant_labels.append(rest_type.replace(' ', '\n'))
    
    dow_matrix = np.array(dow_matrix)
    
    # Create heatmap
    im = axes[1,1].imshow(dow_matrix, cmap='Reds', aspect='auto')
    
    # Set labels
    axes[1,1].set_xticks(range(len(day_order)))
    axes[1,1].set_xticklabels([day[:3] for day in day_order], rotation=45)
    axes[1,1].set_yticks(range(len(restaurant_labels)))
    axes[1,1].set_yticklabels(restaurant_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(restaurant_labels)):
        for j in range(len(day_order)):
            if not np.isnan(dow_matrix[i, j]):
                text_color = "white" if dow_matrix[i, j] > np.nanmean(dow_matrix) else "black"
                axes[1,1].text(j, i, f'{dow_matrix[i, j]:.0f}',
                              ha="center", va="center", color=text_color, fontsize=9)
    
    axes[1,1].set_title('Weekly Peak Demand Patterns', fontsize=13, pad=15)
    axes[1,1].set_xlabel('Day of Week', fontsize=12)
    axes[1,1].set_ylabel('Restaurant Type', fontsize=12)
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1,1], label='Average Customer Count')
    
    plt.tight_layout()
    plt.savefig('peak_demand_periods_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: peak_demand_periods_analysis.png")

def generate_peak_insights_summary(peak_analysis, pattern_analysis):
    """Generate actionable insights from peak analysis."""
    
    print("\n" + "="*70)
    print("PEAK DEMAND INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    insights = []
    
    # Overall peak patterns
    print("\n🔍 KEY FINDINGS:")
    print("-" * 30)
    
    # Find restaurants with highest peak intensity
    highest_intensity = max(peak_analysis.items(), key=lambda x: x[1]['peak_intensity_multiplier'])
    print(f"1. Highest Peak Intensity: {highest_intensity[0].title()} ({highest_intensity[1]['peak_intensity_multiplier']:.1f}x normal)")
    
    # Find most frequent peak periods
    most_frequent_peaks = max(peak_analysis.items(), key=lambda x: x[1]['peak_frequency_pct'])
    print(f"2. Most Frequent Peaks: {most_frequent_peaks[0].title()} ({most_frequent_peaks[1]['peak_frequency_pct']:.1f}% of time)")
    
    # Find longest peak durations
    longest_peaks = max(peak_analysis.items(), key=lambda x: x[1]['avg_peak_duration_hours'])
    print(f"3. Longest Peak Duration: {longest_peaks[0].title()} ({longest_peaks[1]['avg_peak_duration_hours']:.1f} hours)")
    
    # Pattern distribution
    pattern_types = {}
    for rest_type, pattern in pattern_analysis.items():
        ptype = pattern['pattern_type']
        if ptype not in pattern_types:
            pattern_types[ptype] = []
        pattern_types[ptype].append(rest_type)
    
    print(f"\n4. Pattern Distribution:")
    for pattern, restaurants in pattern_types.items():
        print(f"   • {pattern}: {', '.join([r.title() for r in restaurants])}")
    
    # Predictability rankings
    predictability_ranking = sorted(pattern_analysis.items(), 
                                  key=lambda x: x[1]['predictability_score'], reverse=True)
    
    print(f"\n5. Predictability Ranking (Most to Least Predictable):")
    for i, (rest_type, pattern) in enumerate(predictability_ranking, 1):
        print(f"   {i}. {rest_type.title()}: {pattern['predictability_score']:.2f}/1.00")
    
    print("\n" + "="*70)
    print("STAFFING RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    for rest_type, analysis in peak_analysis.items():
        peak_hours = analysis['peak_hours']
        intensity = analysis['peak_intensity_multiplier']
        duration = analysis['avg_peak_duration_hours']
        
        # Generate specific recommendations
        if intensity > 2.0:
            rec = f"🚨 {rest_type.title()}: Critical staffing needed - {intensity:.1f}x normal demand during peaks"
            priority = "HIGH"
        elif intensity > 1.5:
            rec = f"⚠️ {rest_type.title()}: Significant staffing increase needed - {intensity:.1f}x normal demand"
            priority = "MEDIUM"
        else:
            rec = f"✅ {rest_type.title()}: Moderate staffing adjustment - {intensity:.1f}x normal demand"
            priority = "LOW"
        
        recommendations.append({
            'restaurant': rest_type,
            'recommendation': rec,
            'priority': priority,
            'peak_hours': peak_hours,
            'intensity': intensity,
            'duration': duration
        })
        
        print(f"\n{rec}")
        print(f"   Peak Hours: {', '.join([f'{int(h):02d}:00' for h in sorted(peak_hours)])}")
        print(f"   Duration: {duration:.1f} hours average")
        print(f"   Priority: {priority}")
    
    return insights, recommendations

def main():
    """Main function for peak demand periods analysis."""
    print("="*70)
    print("STINT PART 1: PEAK DEMAND PERIODS ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Derive peak demand periods
    peak_analysis = derive_peak_demand_periods(df)
    
    # Characterize peak patterns
    pattern_analysis = characterize_peak_patterns(df, peak_analysis)
    
    # Create visualizations
    create_peak_demand_visualizations(df, peak_analysis, pattern_analysis)
    
    # Generate insights and recommendations
    insights, recommendations = generate_peak_insights_summary(peak_analysis, pattern_analysis)
    
    # Save results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'peak_analysis': peak_analysis,
        'pattern_analysis': pattern_analysis,
        'recommendations': recommendations,
        'summary_statistics': {
            'total_restaurants_analyzed': len(peak_analysis),
            'avg_peak_intensity': round(np.mean([a['peak_intensity_multiplier'] for a in peak_analysis.values()]), 2),
            'avg_peak_duration': round(np.mean([a['avg_peak_duration_hours'] for a in peak_analysis.values()]), 2),
            'avg_peak_frequency': round(np.mean([a['peak_frequency_pct'] for a in peak_analysis.values()]), 2)
        }
    }
    
    with open('peak_demand_periods_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("📊 Results saved to:")
    print("  • peak_demand_periods_results.json")
    print("  • peak_demand_periods_analysis.png")
    print("="*70)

if __name__ == "__main__":
    main()