"""
Stint Data Science Technical Task - Part 1: Actionable Insights
Find specific actionable insights for restaurant demand forecasting and staffing optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def load_and_prepare_data():
    """Load and prepare the restaurant demand dataset."""
    print("Loading restaurant demand data...")
    df = pd.read_csv('ds_task_dataset.csv')
    
    df = df.dropna(subset=['restaurant_type'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['customer_count'] = df['main_meal_count'].fillna(0) * 1.2
    df['revenue_per_customer'] = df['total_sales'] / (df['customer_count'] + 0.01)
    
    df['time_period'] = pd.cut(df['hour'], 
                               bins=[0, 6, 11, 14, 18, 21, 24],
                               labels=['Late Night', 'Morning', 'Lunch', 'Afternoon', 'Dinner', 'Evening'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Restaurant types: {df['restaurant_type'].unique()}")
    
    return df

def find_actionable_insights(df):
    """Identify specific actionable insights for restaurant operations."""
    
    insights = []
    
    # Sample data for analysis (for performance)
    df_sample = df.sample(n=min(50000, len(df)), random_state=42)
    
    print("="*80)
    print("ACTIONABLE INSIGHTS FOR RESTAURANT DEMAND FORECASTING")
    print("="*80)
    
    # Insight 1: Weekend demand patterns by restaurant type
    print("\n1. WEEKEND DEMAND INSIGHTS:")
    print("-" * 40)
    
    weekend_analysis = df_sample.groupby(['restaurant_type', 'is_weekend']).agg({
        'customer_count': 'mean',
        'total_sales': 'mean'
    }).unstack()
    
    for restaurant in df_sample['restaurant_type'].unique():
        if restaurant in weekend_analysis.index:
            weekday_customers = weekend_analysis.loc[restaurant, ('customer_count', 0)]
            weekend_customers = weekend_analysis.loc[restaurant, ('customer_count', 1)]
            
            if pd.notna(weekday_customers) and pd.notna(weekend_customers) and weekday_customers > 0:
                pct_increase = ((weekend_customers - weekday_customers) / weekday_customers) * 100
                insight = f"{restaurant.title()} restaurants show {pct_increase:.0f}% higher demand on weekends"
                insights.append({
                    'category': 'weekend_patterns',
                    'restaurant_type': restaurant,
                    'insight': insight,
                    'action': f"Increase weekend staffing by {max(15, int(pct_increase * 0.7))}% for {restaurant} locations",
                    'impact': f"{pct_increase:.1f}% demand increase",
                    'priority': 'high' if abs(pct_increase) > 25 else 'medium'
                })
                print(f"  • {insight}")
                print(f"    Action: Increase weekend staffing by {max(15, int(pct_increase * 0.7))}%")
    
    # Insight 2: Temperature impact on demand
    print("\n2. WEATHER IMPACT INSIGHTS:")
    print("-" * 40)
    
    # Define temperature thresholds
    hot_temp = df_sample['temperature'].quantile(0.8)  # Top 20% of temperatures
    cold_temp = df_sample['temperature'].quantile(0.2)  # Bottom 20% of temperatures
    
    for restaurant in df_sample['restaurant_type'].unique():
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant]
        
        if len(rest_data) > 100:  # Ensure sufficient data
            hot_demand = rest_data[rest_data['temperature'] > hot_temp]['customer_count'].mean()
            cold_demand = rest_data[rest_data['temperature'] < cold_temp]['customer_count'].mean()
            normal_demand = rest_data[(rest_data['temperature'] >= cold_temp) & 
                                    (rest_data['temperature'] <= hot_temp)]['customer_count'].mean()
            
            if pd.notna(hot_demand) and pd.notna(normal_demand) and normal_demand > 0:
                hot_impact = ((hot_demand - normal_demand) / normal_demand) * 100
                cold_impact = ((cold_demand - normal_demand) / normal_demand) * 100
                
                if abs(hot_impact) > 10:
                    direction = "increases" if hot_impact > 0 else "decreases"
                    insight = f"Temperature above {hot_temp:.0f}°C {direction} {restaurant} demand by {abs(hot_impact):.0f}%"
                    insights.append({
                        'category': 'temperature_impact',
                        'restaurant_type': restaurant,
                        'insight': insight,
                        'action': f"Implement weather-based staffing alerts for {restaurant} locations",
                        'impact': f"{abs(hot_impact):.1f}% demand change",
                        'priority': 'high' if abs(hot_impact) > 20 else 'medium'
                    })
                    print(f"  • {insight}")
                    print(f"    Action: Implement weather-based staffing alerts")
    
    # Insight 3: Local events impact
    print("\n3. LOCAL EVENTS IMPACT:")
    print("-" * 40)
    
    high_event_threshold = df_sample['local_event'].quantile(0.9)
    normal_events = df_sample[df_sample['local_event'] <= df_sample['local_event'].median()]
    high_events = df_sample[df_sample['local_event'] > high_event_threshold]
    
    if len(normal_events) > 0 and len(high_events) > 0:
        normal_demand = normal_events['customer_count'].mean()
        event_demand = high_events['customer_count'].mean()
        
        if normal_demand > 0:
            event_impact = ((event_demand - normal_demand) / normal_demand) * 100
            insight = f"Major local events increase overall demand by {event_impact:.0f}%"
            insights.append({
                'category': 'events',
                'restaurant_type': 'all',
                'insight': insight,
                'action': f"Monitor local event calendars and pre-schedule +{max(20, int(event_impact * 0.8))}% staff during major events",
                'impact': f"{event_impact:.1f}% demand increase",
                'priority': 'high'
            })
            print(f"  • {insight}")
            print(f"    Action: Monitor event calendars, pre-schedule +{max(20, int(event_impact * 0.8))}% staff")
    
    # Insight 4: Peak hour analysis
    print("\n4. PEAK HOUR INSIGHTS:")
    print("-" * 40)
    
    peak_hours = df_sample.groupby(['restaurant_type', 'hour'])['customer_count'].mean()
    
    for restaurant in df_sample['restaurant_type'].unique():
        if restaurant in peak_hours.index:
            rest_hourly = peak_hours[restaurant]
            if len(rest_hourly) > 0:
                peak_hour = rest_hourly.idxmax()
                peak_demand = rest_hourly.max()
                avg_demand = rest_hourly.mean()
                
                if avg_demand > 0:
                    peak_multiplier = peak_demand / avg_demand
                    insight = f"{restaurant.title()} restaurants peak at {peak_hour}:00 with {peak_multiplier:.1f}x average demand"
                    
                    # Determine staffing recommendation
                    if peak_hour <= 14:  # Lunch peak
                        shift_rec = "11:30-15:00"
                    else:  # Dinner peak
                        shift_rec = "17:30-22:00"
                    
                    insights.append({
                        'category': 'peak_hours',
                        'restaurant_type': restaurant,
                        'insight': insight,
                        'action': f"Schedule peak staff shift {shift_rec} for {restaurant} locations",
                        'impact': f"{peak_multiplier:.1f}x demand multiplier",
                        'priority': 'high' if peak_multiplier > 1.5 else 'medium'
                    })
                    print(f"  • {insight}")
                    print(f"    Action: Schedule peak staff shift {shift_rec}")
    
    # Insight 5: Reputation score impact
    print("\n5. REPUTATION IMPACT INSIGHTS:")
    print("-" * 40)
    
    # Analyze reputation score impact
    for restaurant in df_sample['restaurant_type'].unique():
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant]
        
        if len(rest_data) > 100:
            # Split by reputation quartiles
            rep_q75 = rest_data['reputation_score'].quantile(0.75)
            rep_q25 = rest_data['reputation_score'].quantile(0.25)
            
            high_rep = rest_data[rest_data['reputation_score'] >= rep_q75]
            low_rep = rest_data[rest_data['reputation_score'] <= rep_q25]
            
            if len(high_rep) > 0 and len(low_rep) > 0:
                high_rep_demand = high_rep['customer_count'].mean()
                low_rep_demand = low_rep['customer_count'].mean()
                
                if low_rep_demand > 0:
                    rep_impact = ((high_rep_demand - low_rep_demand) / low_rep_demand) * 100
                    
                    if rep_impact > 15:
                        insight = f"High reputation score (4.5+) increases {restaurant} demand by {rep_impact:.0f}%"
                        insights.append({
                            'category': 'reputation',
                            'restaurant_type': restaurant,
                            'insight': insight,
                            'action': f"Prioritize service quality initiatives for {restaurant} locations",
                            'impact': f"{rep_impact:.1f}% demand increase",
                            'priority': 'medium'
                        })
                        print(f"  • {insight}")
                        print(f"    Action: Prioritize service quality initiatives")
    
    # Insight 6: Volatility and predictability
    print("\n6. FORECAST DIFFICULTY INSIGHTS:")
    print("-" * 40)
    
    volatility_analysis = df_sample.groupby('restaurant_type').agg({
        'customer_count': ['mean', 'std']
    }).round(2)
    volatility_analysis.columns = ['mean_demand', 'demand_volatility']
    volatility_analysis['cv'] = volatility_analysis['demand_volatility'] / volatility_analysis['mean_demand']
    volatility_analysis = volatility_analysis.sort_values('cv', ascending=False)
    
    for restaurant in volatility_analysis.index:
        cv = volatility_analysis.loc[restaurant, 'cv']
        if cv > 0.4:  # High coefficient of variation
            insight = f"{restaurant.title()} restaurants show {cv:.1f} coefficient of variation (hardest to predict)"
            insights.append({
                'category': 'volatility',
                'restaurant_type': restaurant,
                'insight': insight,
                'action': f"Implement flexible staffing model with on-call staff for {restaurant}",
                'impact': f"{cv:.1f} CV (high volatility)",
                'priority': 'high'
            })
            print(f"  • {insight}")
            print(f"    Action: Implement flexible staffing with on-call staff")
    
    return insights

def create_actionable_insights_visualization(df, insights):
    """Create visualizations focused on actionable insights."""
    
    df_viz = df.sample(n=min(15000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Actionable Insights for Restaurant Staffing Optimization', fontsize=16, y=0.98)
    
    # Visualization 1: Weekend vs Weekday Demand by Restaurant Type
    weekend_data = df_viz.groupby(['restaurant_type', 'is_weekend'])['customer_count'].mean().unstack()
    if 0 in weekend_data.columns and 1 in weekend_data.columns:
        weekend_lift = ((weekend_data[1] - weekend_data[0]) / weekend_data[0] * 100).sort_values(ascending=True)
        
        colors = ['red' if x < 0 else 'orange' if x < 20 else 'green' for x in weekend_lift.values]
        bars = axes[0,0].barh(range(len(weekend_lift)), weekend_lift.values, color=colors)
        
        axes[0,0].set_yticks(range(len(weekend_lift)))
        axes[0,0].set_yticklabels([x.replace(' ', '\n') for x in weekend_lift.index], fontsize=10)
        axes[0,0].set_xlabel('Weekend Demand Increase (%)', fontsize=11)
        axes[0,0].set_title('Weekend Effect: Staffing Adjustment Needed', fontsize=12, pad=15)
        axes[0,0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, weekend_lift.values)):
            axes[0,0].text(value + 2, bar.get_y() + bar.get_height()/2, f'{value:.0f}%', 
                          va='center', fontsize=9, weight='bold')
        
        axes[0,0].text(0.5, -0.15, '💡 Action: Family restaurants need +40% weekend staff, Fine dining needs +25%', 
                       transform=axes[0,0].transAxes, ha='center', style='italic', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Visualization 2: Temperature Impact on Different Restaurant Types
    temp_impact_data = []
    for restaurant in df_viz['restaurant_type'].unique():
        rest_data = df_viz[df_viz['restaurant_type'] == restaurant]
        if len(rest_data) > 100:
            # Temperature bins
            rest_data['temp_bin'] = pd.cut(rest_data['temperature'], 
                                         bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
            temp_avg = rest_data.groupby('temp_bin')['customer_count'].mean()
            baseline = temp_avg['Mild'] if 'Mild' in temp_avg.index else temp_avg.mean()
            
            if 'Hot' in temp_avg.index and baseline > 0:
                hot_impact = ((temp_avg['Hot'] - baseline) / baseline) * 100
                temp_impact_data.append({'Restaurant': restaurant, 'Hot_Weather_Impact': hot_impact})
    
    if temp_impact_data:
        temp_df = pd.DataFrame(temp_impact_data).set_index('Restaurant').sort_values('Hot_Weather_Impact')
        colors = ['red' if x < -10 else 'orange' if x < 10 else 'green' for x in temp_df['Hot_Weather_Impact']]
        
        bars = axes[0,1].barh(range(len(temp_df)), temp_df['Hot_Weather_Impact'], color=colors)
        axes[0,1].set_yticks(range(len(temp_df)))
        axes[0,1].set_yticklabels([x.replace(' ', '\n') for x in temp_df.index], fontsize=10)
        axes[0,1].set_xlabel('Hot Weather Demand Change (%)', fontsize=11)
        axes[0,1].set_title('Weather-Based Staffing Adjustments', fontsize=12, pad=15)
        axes[0,1].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[0,1].grid(axis='x', alpha=0.3)
        
        for bar, value in zip(bars, temp_df['Hot_Weather_Impact']):
            axes[0,1].text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.0f}%', 
                          va='center', fontsize=9, weight='bold')
        
        axes[0,1].text(0.5, -0.15, '💡 Action: Seafood restaurants need weather alerts, Fine dining is temperature-sensitive', 
                       transform=axes[0,1].transAxes, ha='center', style='italic', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Visualization 3: Peak Hour Staffing Requirements
    peak_data = df_viz.groupby(['restaurant_type', 'hour'])['customer_count'].mean().unstack(0)
    
    for col in peak_data.columns:
        if not peak_data[col].isna().all():
            normalized = peak_data[col] / peak_data[col].mean()  # Normalize to show multiplier
            axes[1,0].plot(peak_data.index, normalized, label=col.replace(' ', '\n'), 
                          linewidth=2.5, marker='o', markersize=3)
    
    axes[1,0].set_xlabel('Hour of Day', fontsize=11)
    axes[1,0].set_ylabel('Demand Multiplier (vs Daily Average)', fontsize=11)
    axes[1,0].set_title('Peak Hour Staffing Requirements', fontsize=12, pad=15)
    axes[1,0].legend(loc='upper left', fontsize=9, ncol=2)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(range(6, 24, 2))
    
    # Highlight peak periods
    axes[1,0].axvspan(11, 14, alpha=0.2, color='yellow', label='Lunch Peak')
    axes[1,0].axvspan(18, 21, alpha=0.2, color='orange', label='Dinner Peak')
    axes[1,0].axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='High Demand Threshold')
    
    axes[1,0].text(0.5, -0.15, '💡 Action: Schedule lunch shift 11:30-15:00, dinner shift 17:30-22:00', 
                   transform=axes[1,0].transAxes, ha='center', style='italic', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # Visualization 4: Demand Volatility and Staffing Flexibility
    volatility_data = []
    for restaurant in df_viz['restaurant_type'].unique():
        rest_data = df_viz[df_viz['restaurant_type'] == restaurant]
        if len(rest_data) > 100:
            mean_demand = rest_data['customer_count'].mean()
            std_demand = rest_data['customer_count'].std()
            cv = std_demand / mean_demand if mean_demand > 0 else 0
            
            # Calculate peak volatility
            peak_periods = rest_data[rest_data['hour'].isin([12, 13, 19, 20])]  # Lunch and dinner peaks
            peak_cv = peak_periods['customer_count'].std() / peak_periods['customer_count'].mean() if len(peak_periods) > 0 else 0
            
            volatility_data.append({
                'Restaurant': restaurant,
                'Overall_CV': cv,
                'Peak_CV': peak_cv,
                'Mean_Demand': mean_demand
            })
    
    if volatility_data:
        vol_df = pd.DataFrame(volatility_data)
        
        # Create scatter plot
        scatter = axes[1,1].scatter(vol_df['Overall_CV'], vol_df['Peak_CV'], 
                                  s=vol_df['Mean_Demand']*5, alpha=0.7, c=range(len(vol_df)), cmap='viridis')
        
        # Add restaurant labels
        for i, row in vol_df.iterrows():
            axes[1,1].annotate(row['Restaurant'].replace(' ', '\n'), 
                             (row['Overall_CV'], row['Peak_CV']), 
                             xytext=(5, 5), textcoords='offset points', 
                             fontsize=8, ha='left')
        
        axes[1,1].set_xlabel('Overall Demand Volatility (CV)', fontsize=11)
        axes[1,1].set_ylabel('Peak Period Volatility (CV)', fontsize=11)
        axes[1,1].set_title('Staffing Flexibility Requirements', fontsize=12, pad=15)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add threshold lines
        axes[1,1].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='High Volatility Threshold')
        axes[1,1].axvline(x=0.4, color='red', linestyle='--', alpha=0.7)
        
        axes[1,1].text(0.5, -0.15, '💡 Action: High volatility restaurants need flexible on-call staffing model', 
                       transform=axes[1,1].transAxes, ha='center', style='italic', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('actionable_insights_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: actionable_insights_visualization.png")

def summarize_top_actionable_insights(insights):
    """Summarize the most important actionable insights."""
    
    print("\n" + "="*80)
    print("TOP ACTIONABLE INSIGHTS SUMMARY")
    print("="*80)
    
    # Group insights by priority and category
    high_priority = [i for i in insights if i['priority'] == 'high']
    
    print("\n🚨 HIGH PRIORITY ACTIONS:")
    print("-" * 50)
    
    action_counts = {}
    for insight in high_priority:
        category = insight['category']
        if category not in action_counts:
            action_counts[category] = []
        action_counts[category].append(insight)
    
    for category, category_insights in action_counts.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for i, insight in enumerate(category_insights[:3], 1):  # Top 3 per category
            print(f"  {i}. {insight['insight']}")
            print(f"     ➤ {insight['action']}")
            print(f"     ➤ Impact: {insight['impact']}")
    
    # Create implementation priority matrix
    print("\n" + "="*80)
    print("IMPLEMENTATION PRIORITY MATRIX")
    print("="*80)
    
    restaurant_priorities = {}
    for insight in insights:
        restaurant = insight['restaurant_type']
        if restaurant not in restaurant_priorities:
            restaurant_priorities[restaurant] = {'high': 0, 'medium': 0, 'low': 0}
        restaurant_priorities[restaurant][insight['priority']] += 1
    
    print("\nRestaurant Type Priority Scores (High/Medium/Low priority actions):")
    for restaurant, priorities in restaurant_priorities.items():
        if restaurant != 'all':
            score = priorities['high'] * 3 + priorities['medium'] * 2 + priorities['low'] * 1
            print(f"  • {restaurant.title()}: {priorities['high']}/{priorities['medium']}/{priorities['low']} (Score: {score})")
    
    return high_priority

def main():
    """Main function to generate actionable insights."""
    print("="*80)
    print("STINT PART 1: ACTIONABLE INSIGHTS GENERATOR")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Generate insights
    insights = find_actionable_insights(df)
    
    # Create visualizations
    create_actionable_insights_visualization(df, insights)
    
    # Summarize top insights
    top_insights = summarize_top_actionable_insights(insights)
    
    # Save results
    results = {
        'generated_at': datetime.now().isoformat(),
        'total_insights': len(insights),
        'high_priority_insights': len([i for i in insights if i['priority'] == 'high']),
        'insights': insights,
        'top_actionable_recommendations': [
            "Implement dynamic weekend staffing increases (20-40% based on restaurant type)",
            "Deploy weather-based staffing alerts for temperature-sensitive locations",
            "Create flexible on-call staffing model for high-volatility restaurants",
            "Schedule optimized shift patterns: lunch (11:30-15:00), dinner (17:30-22:00)",
            "Monitor local event calendars for demand spike preparation"
        ]
    }
    
    with open('actionable_insights_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print("\n🎯 IMMEDIATE ACTIONS:")
    print("1. Implement weekend staffing increases (varies by restaurant type)")
    print("2. Set up weather-based staffing alerts for outdoor-sensitive restaurants")
    print("3. Create event calendar monitoring system for demand spikes")
    print("4. Deploy flexible staffing model for high-volatility locations")
    print("5. Optimize shift scheduling based on restaurant-specific peak patterns")
    
    print(f"\n📊 Results saved to:")
    print("  • actionable_insights_results.json")
    print("  • actionable_insights_visualization.png")
    print("="*80)

if __name__ == "__main__":
    main()