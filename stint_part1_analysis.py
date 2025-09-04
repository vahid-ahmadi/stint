import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RestaurantDemandAnalysis:
    def __init__(self, csv_path):
        """Initialize the analysis with restaurant demand data"""
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Clean and prepare the dataset for analysis"""
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Calculate total customer count as main demand metric
        self.df['total_customers'] = (
            self.df['main_meal_count'].fillna(0) + 
            self.df['starter_count'].fillna(0) + 
            self.df['soft_drink_count'].fillna(0) + 
            self.df['complex_drink_count'].fillna(0) + 
            self.df['dessert_count'].fillna(0) + 
            self.df['child_meal_count'].fillna(0)
        )
        
        # Create time-based features
        self.df['hour_block'] = self.df['hour'].map({
            8: 'Breakfast', 9: 'Breakfast', 10: 'Breakfast',
            11: 'Lunch', 12: 'Lunch', 13: 'Lunch', 14: 'Lunch',
            15: 'Afternoon', 16: 'Afternoon', 17: 'Afternoon',
            18: 'Dinner', 19: 'Dinner', 20: 'Dinner', 21: 'Dinner'
        })
        
        # Weekend flag
        self.df['is_weekend'] = self.df['day_of_week'].isin(['Saturday', 'Sunday'])
        
        # Temperature categories
        self.df['temp_category'] = pd.cut(self.df['temperature'], 
                                        bins=[0, 15, 25, 35, 50], 
                                        labels=['Cold', 'Mild', 'Warm', 'Hot'])
        
        # Peak demand periods (top 20%)
        demand_threshold = self.df['total_customers'].quantile(0.8)
        self.df['is_peak'] = self.df['total_customers'] >= demand_threshold
        
        print(f"Data loaded: {len(self.df)} records from {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Restaurant types: {self.df['restaurant_type'].unique()}")
        
    def analyze_demand_drivers(self):
        """Identify primary drivers of demand for each restaurant type"""
        results = {}
        
        for restaurant_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == restaurant_type]
            
            # Calculate correlations with demand
            correlations = {}
            for factor in ['temperature', 'precipitation', 'economic_indicator', 
                          'competitor_promo', 'social_trend', 'local_event', 'reputation_score']:
                if factor in type_data.columns:
                    corr = type_data['total_customers'].corr(type_data[factor])
                    if not pd.isna(corr):
                        correlations[factor] = corr
            
            # Weekend vs weekday impact
            weekend_avg = type_data[type_data['is_weekend']]['total_customers'].mean()
            weekday_avg = type_data[~type_data['is_weekend']]['total_customers'].mean()
            weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0
            
            results[restaurant_type] = {
                'correlations': correlations,
                'weekend_lift': weekend_lift,
                'avg_customers': type_data['total_customers'].mean()
            }
            
        return results
    
    def analyze_external_factors(self):
        """Quantify how external factors impact demand with detailed business insights"""
        factor_impacts = {}
        
        # Temperature impact analysis with precise thresholds
        temp_impact = []
        baseline_temp = self.df[self.df['temperature'] <= 15]['total_customers'].mean()
        
        for temp_cat in ['Cold', 'Mild', 'Warm', 'Hot']:
            if temp_cat in self.df['temp_category'].values:
                category_data = self.df[self.df['temp_category'] == temp_cat]
                avg_demand = category_data['total_customers'].mean()
                pct_change = ((avg_demand - baseline_temp) / baseline_temp * 100) if baseline_temp > 0 else 0
                temp_impact.append({
                    'category': temp_cat, 
                    'avg_demand': avg_demand,
                    'pct_change_from_baseline': pct_change,
                    'sample_size': len(category_data)
                })
        
        factor_impacts['temperature'] = temp_impact
        
        # Detailed competitor promotion analysis
        high_promo = self.df[self.df['competitor_promo'] > 0.7]['total_customers'].mean()
        med_promo = self.df[(self.df['competitor_promo'] > 0.3) & (self.df['competitor_promo'] <= 0.7)]['total_customers'].mean()
        low_promo = self.df[self.df['competitor_promo'] <= 0.3]['total_customers'].mean()
        
        promo_impact = ((high_promo - low_promo) / low_promo * 100) if low_promo > 0 else 0
        
        factor_impacts['competitor_promo_impact'] = promo_impact
        factor_impacts['competitor_promo_detailed'] = {
            'high_promo_avg': high_promo,
            'medium_promo_avg': med_promo,
            'low_promo_avg': low_promo
        }
        
        # Local events comprehensive analysis
        event_quartiles = self.df['local_event'].quantile([0.25, 0.5, 0.75, 1.0])
        no_event = self.df[self.df['local_event'] == 0]['total_customers'].mean()
        low_event = self.df[(self.df['local_event'] > 0) & (self.df['local_event'] <= event_quartiles[0.5])]['total_customers'].mean()
        high_event = self.df[self.df['local_event'] > event_quartiles[0.75]]['total_customers'].mean()
        
        event_lift = ((high_event - no_event) / no_event * 100) if no_event > 0 else 0
        
        factor_impacts['local_event_lift'] = event_lift
        factor_impacts['event_analysis'] = {
            'no_event_avg': no_event,
            'low_event_avg': low_event,
            'high_event_avg': high_event,
            'max_event_impact': self.df[self.df['local_event'] == self.df['local_event'].max()]['total_customers'].mean()
        }
        
        # Economic indicator impact
        econ_high = self.df[self.df['economic_indicator'] > self.df['economic_indicator'].quantile(0.75)]['total_customers'].mean()
        econ_low = self.df[self.df['economic_indicator'] < self.df['economic_indicator'].quantile(0.25)]['total_customers'].mean()
        econ_impact = ((econ_high - econ_low) / econ_low * 100) if econ_low > 0 else 0
        
        factor_impacts['economic_impact'] = econ_impact
        
        # Weather combination effects
        sunny_warm = self.df[(self.df['temperature'] > 20) & (self.df['precipitation'] < 1)]['total_customers'].mean()
        rainy_cold = self.df[(self.df['temperature'] < 15) & (self.df['precipitation'] > 5)]['total_customers'].mean()
        weather_combo_effect = ((sunny_warm - rainy_cold) / rainy_cold * 100) if rainy_cold > 0 else 0
        
        factor_impacts['weather_combination_effect'] = weather_combo_effect
        
        return factor_impacts
    
    def identify_peak_periods(self):
        """Derive and characterize peak demand periods"""
        peak_analysis = {}
        
        # Peak periods by time of day
        hourly_avg = self.df.groupby('hour')['total_customers'].agg(['mean', 'std']).reset_index()
        peak_hours = hourly_avg.nlargest(3, 'mean')['hour'].tolist()
        
        # Peak periods by day of week
        daily_avg = self.df.groupby('day_of_week')['total_customers'].agg(['mean', 'std']).reset_index()
        
        # Peak periods by restaurant type
        type_peaks = {}
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            peak_data = type_data[type_data['is_peak']]
            
            peak_hours_type = peak_data['hour'].value_counts().head(3).index.tolist()
            peak_days_type = peak_data['day_of_week'].value_counts().head(3).index.tolist()
            
            type_peaks[rest_type] = {
                'peak_hours': peak_hours_type,
                'peak_days': peak_days_type,
                'peak_volatility': peak_data['total_customers'].std(),
                'avg_peak_demand': peak_data['total_customers'].mean()
            }
        
        peak_analysis['overall_peak_hours'] = peak_hours
        peak_analysis['by_restaurant_type'] = type_peaks
        peak_analysis['daily_patterns'] = daily_avg.to_dict('records')
        
        return peak_analysis
    
    def analyze_forecast_difficulty(self):
        """Identify hardest-to-predict periods and analyze difficulty by restaurant type"""
        difficulty_analysis = {}
        
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            
            # Calculate coefficient of variation as a measure of predictability
            hourly_cv = type_data.groupby('hour')['total_customers'].agg(['mean', 'std'])
            hourly_cv['cv'] = hourly_cv['std'] / hourly_cv['mean']
            
            # Most difficult hours to predict
            difficult_hours = hourly_cv.nlargest(3, 'cv').index.tolist()
            
            # Overall variability metrics
            peak_periods = type_data[type_data['is_peak']]
            non_peak_periods = type_data[~type_data['is_peak']]
            
            difficulty_analysis[rest_type] = {
                'difficult_hours': difficult_hours,
                'peak_volatility': peak_periods['total_customers'].std(),
                'non_peak_volatility': non_peak_periods['total_customers'].std(),
                'overall_cv': type_data['total_customers'].std() / type_data['total_customers'].mean()
            }
        
        return difficulty_analysis
    
    def create_visualizations(self):
        """Generate 4 key visualizations for business insights"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Restaurant Demand Analysis - Key Business Insights', fontsize=16, fontweight='bold')
        
        # Visualization 1: Demand patterns by restaurant type and time
        ax1 = axes[0, 0]
        pivot_data = self.df.groupby(['restaurant_type', 'hour'])['total_customers'].mean().unstack(level=0)
        pivot_data.plot(ax=ax1, kind='line', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Hourly Demand Patterns by Restaurant Type', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Customers')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Visualization 2: External factor impact on demand
        ax2 = axes[0, 1]
        temp_demand = self.df.groupby('temp_category')['total_customers'].mean()
        temp_demand.plot(ax=ax2, kind='bar', color='skyblue', alpha=0.8)
        ax2.set_title('Temperature Impact on Customer Demand', fontweight='bold')
        ax2.set_xlabel('Temperature Category')
        ax2.set_ylabel('Average Customers')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        baseline = temp_demand.min()
        for i, v in enumerate(temp_demand):
            pct_change = ((v - baseline) / baseline * 100) if baseline > 0 else 0
            ax2.text(i, v + 1, f'+{pct_change:.1f}%' if pct_change > 0 else f'{pct_change:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Visualization 3: Weekend vs Weekday demand by restaurant type
        ax3 = axes[1, 0]
        weekend_comparison = []
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            weekend_avg = type_data[type_data['is_weekend']]['total_customers'].mean()
            weekday_avg = type_data[~type_data['is_weekend']]['total_customers'].mean()
            weekend_comparison.append({
                'Restaurant Type': rest_type,
                'Weekday': weekday_avg,
                'Weekend': weekend_avg
            })
        
        comparison_df = pd.DataFrame(weekend_comparison)
        x_pos = np.arange(len(comparison_df))
        width = 0.35
        
        ax3.bar(x_pos - width/2, comparison_df['Weekday'], width, label='Weekday', alpha=0.8, color='lightcoral')
        ax3.bar(x_pos + width/2, comparison_df['Weekend'], width, label='Weekend', alpha=0.8, color='lightblue')
        
        ax3.set_title('Weekend vs Weekday Demand by Restaurant Type', fontweight='bold')
        ax3.set_xlabel('Restaurant Type')
        ax3.set_ylabel('Average Customers')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(comparison_df['Restaurant Type'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Visualization 4: Peak period volatility and business impact
        ax4 = axes[1, 1]
        volatility_data = []
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            peak_data = type_data[type_data['is_peak']]
            non_peak_data = type_data[~type_data['is_peak']]
            
            volatility_data.append({
                'Restaurant Type': rest_type,
                'Peak Volatility': peak_data['total_customers'].std(),
                'Non-Peak Volatility': non_peak_data['total_customers'].std(),
                'Peak Avg': peak_data['total_customers'].mean()
            })
        
        vol_df = pd.DataFrame(volatility_data)
        
        # Create bubble chart
        scatter = ax4.scatter(vol_df['Peak Volatility'], vol_df['Non-Peak Volatility'], 
                            s=vol_df['Peak Avg']*5, alpha=0.6, c=range(len(vol_df)), cmap='viridis')
        
        for i, row in vol_df.iterrows():
            ax4.annotate(row['Restaurant Type'], (row['Peak Volatility'], row['Non-Peak Volatility']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_title('Demand Volatility: Peak vs Non-Peak Periods', fontweight='bold')
        ax4.set_xlabel('Peak Period Volatility (Std Dev)')
        ax4.set_ylabel('Non-Peak Period Volatility (Std Dev)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/janansadeqian/Downloads/stint_task/restaurant_analysis_insights.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_business_insights(self):
        """Generate comprehensive, actionable business insights per PDF requirements"""
        demand_drivers = self.analyze_demand_drivers()
        factor_impacts = self.analyze_external_factors()
        peak_analysis = self.identify_peak_periods()
        difficulty_analysis = self.analyze_forecast_difficulty()
        
        insights = {
            'key_findings': [],
            'actionable_recommendations': [],
            'forecasting_challenges': [],
            'quantified_impacts': {},
            'peak_insights': {},
            'volatility_analysis': {}
        }
        
        # Comprehensive key findings with precise quantification
        insights['key_findings'].append(f"Dataset contains {len(self.df):,} records across {len(self.df['restaurant_type'].unique())} restaurant types")
        insights['key_findings'].append(f"Average customers per period: {self.df['total_customers'].mean():.1f}")
        insights['key_findings'].append(f"Peak demand varies from {self.df['total_customers'].min():.0f} to {self.df['total_customers'].max():.0f} customers")
        
        # Weekend impact analysis
        max_weekend_lift = 0
        max_weekend_type = ""
        for rest_type, data in demand_drivers.items():
            if pd.notna(data['weekend_lift']) and data['weekend_lift'] > max_weekend_lift:
                max_weekend_lift = data['weekend_lift']
                max_weekend_type = rest_type
            
            if data['weekend_lift'] > 20:
                insights['key_findings'].append(
                    f"{rest_type.title()} restaurants show {data['weekend_lift']:.1f}% higher demand on weekends"
                )
        
        # Temperature impact with specific thresholds
        if 'temperature' in factor_impacts:
            temp_data = factor_impacts['temperature']
            warm_data = next((x for x in temp_data if x['category'] == 'Warm'), None)
            cold_data = next((x for x in temp_data if x['category'] == 'Cold'), None)
            
            if warm_data and cold_data:
                insights['key_findings'].append(
                    f"Warm weather (15-25°C) increases demand by {warm_data['pct_change_from_baseline']:.1f}% compared to cold weather (<15°C)"
                )
        
        # External factor quantification
        insights['quantified_impacts'] = {
            'temperature_effect': factor_impacts.get('weather_combination_effect', 0),
            'competitor_promo_effect': factor_impacts.get('competitor_promo_impact', 0),
            'local_event_effect': factor_impacts.get('local_event_lift', 0),
            'economic_effect': factor_impacts.get('economic_impact', 0)
        }
        
        # Peak period business insights
        peak_revenue_impact = 0
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            peak_revenue = type_data[type_data['is_peak']]['total_sales'].mean()
            non_peak_revenue = type_data[~type_data['is_peak']]['total_sales'].mean()
            if pd.notna(peak_revenue) and pd.notna(non_peak_revenue) and non_peak_revenue > 0:
                revenue_lift = ((peak_revenue - non_peak_revenue) / non_peak_revenue * 100)
                peak_revenue_impact = max(peak_revenue_impact, revenue_lift)
        
        insights['peak_insights'] = {
            'revenue_multiplier': peak_revenue_impact,
            'peak_hours': peak_analysis['overall_peak_hours'],
            'capacity_utilization': self.calculate_capacity_utilization()
        }
        
        # Enhanced actionable recommendations
        insights['actionable_recommendations'] = [
            f"Focus staffing on peak hours ({', '.join([f'{h}:00' for h in peak_analysis['overall_peak_hours']])}) - revenue is {peak_revenue_impact:.1f}% higher",
            f"Implement weather-based staffing: increase staff by {abs(factor_impacts.get('weather_combination_effect', 0)):.0f}% on sunny warm days",
            "Deploy dynamic pricing during high-demand periods to optimize revenue per customer",
            f"Monitor competitor promotions closely - they impact demand by {abs(factor_impacts.get('competitor_promo_impact', 0)):.1f}%",
            f"Weekend staffing for {max_weekend_type} should be {max_weekend_lift:.0f}% higher than weekdays",
            "Implement event-driven staffing alerts - local events boost demand significantly",
            "Focus inventory management on peak volatility periods to avoid stockouts"
        ]
        
        # Advanced forecasting challenges
        volatility_ranking = sorted(difficulty_analysis.items(), key=lambda x: x[1]['overall_cv'], reverse=True)
        most_volatile = volatility_ranking[0]
        
        insights['forecasting_challenges'] = [
            f"High variability during peak periods requires robust models (CV: {most_volatile[1]['overall_cv']:.2f})",
            "External factors have complex interactions requiring ensemble methods",
            f"Restaurant type significantly affects demand patterns - {most_volatile[0]} is most unpredictable",
            "Asymmetric cost of understaffing vs overstaffing requires custom loss functions",
            "Capacity constraints create non-linear demand relationships during peak periods"
        ]
        
        # Volatility business impact analysis
        insights['volatility_analysis'] = {rest_type: {
            'staffing_challenge_score': data['overall_cv'],
            'peak_unpredictability': data['peak_volatility'],
            'operational_difficulty': 'High' if data['overall_cv'] > 0.5 else 'Medium' if data['overall_cv'] > 0.3 else 'Low'
        } for rest_type, data in difficulty_analysis.items() if pd.notna(data['overall_cv'])}
        
        return insights
        
    def calculate_capacity_utilization(self):
        """Calculate capacity utilization metrics"""
        utilization = {}
        for rest_type in self.df['restaurant_type'].unique():
            type_data = self.df[self.df['restaurant_type'] == rest_type]
            if 'capacity_available' in type_data.columns:
                # Assume capacity_available represents remaining capacity, so total capacity = customers + available
                total_capacity = type_data['total_customers'] + type_data['capacity_available']
                avg_utilization = (type_data['total_customers'] / total_capacity).mean()
                peak_utilization = (type_data[type_data['is_peak']]['total_customers'] / 
                                  (type_data[type_data['is_peak']]['total_customers'] + type_data[type_data['is_peak']]['capacity_available'])).mean()
                utilization[rest_type] = {
                    'average': avg_utilization,
                    'peak': peak_utilization
                }
        return utilization
    
    def export_results(self):
        """Export analysis results for dashboard integration"""
        results = {
            'demand_drivers': self.analyze_demand_drivers(),
            'external_factors': self.analyze_external_factors(),
            'peak_analysis': self.identify_peak_periods(),
            'difficulty_analysis': self.analyze_forecast_difficulty(),
            'insights': self.generate_business_insights(),
            'summary_stats': {
                'total_records': len(self.df),
                'date_range': f"{self.df['timestamp'].min()} to {self.df['timestamp'].max()}",
                'restaurant_types': self.df['restaurant_type'].unique().tolist(),
                'avg_daily_customers': self.df.groupby(self.df['timestamp'].dt.date)['total_customers'].sum().mean()
            }
        }
        
        # Save to JSON for dashboard
        import json
        with open('/Users/janansadeqian/Downloads/stint_task/analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

def main():
    """Main execution function"""
    print("🍽️  Starting Restaurant Demand Analysis - Part 1")
    print("=" * 50)
    
    # Initialize analysis
    analyzer = RestaurantDemandAnalysis('/Users/janansadeqian/Downloads/stint_task/ds_task_dataset.csv')
    
    # Generate visualizations
    print("\n📊 Generating key business insight visualizations...")
    analyzer.create_visualizations()
    
    # Generate and display insights
    print("\n💡 Key Business Insights:")
    insights = analyzer.generate_business_insights()
    
    print("\n🔍 KEY FINDINGS:")
    for finding in insights['key_findings']:
        print(f"  • {finding}")
    
    print("\n🎯 ACTIONABLE RECOMMENDATIONS:")
    for rec in insights['actionable_recommendations']:
        print(f"  • {rec}")
    
    print("\n⚠️  FORECASTING CHALLENGES:")
    for challenge in insights['forecasting_challenges']:
        print(f"  • {challenge}")
    
    # Export results
    print("\n💾 Exporting results for dashboard integration...")
    results = analyzer.export_results()
    
    print(f"\n✅ Analysis complete! Generated:")
    print(f"  • Visualization: restaurant_analysis_insights.png")
    print(f"  • Data export: analysis_results.json")
    print(f"  • Ready for React dashboard integration")
    
    return results

if __name__ == "__main__":
    main()