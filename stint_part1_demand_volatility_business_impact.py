"""
Stint Data Science Technical Task - Part 1: Business Impact of Demand Volatility Analysis
Quantify the business impact of demand volatility during peak periods
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
sns.set_palette("coolwarm")

def load_and_prepare_data():
    """Load and prepare the restaurant demand dataset."""
    print("Loading restaurant demand data for business impact analysis...")
    df = pd.read_csv('ds_task_dataset.csv')
    
    df = df.dropna(subset=['restaurant_type'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create comprehensive features for business impact analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
    
    # Calculate customer count and business metrics
    df['customer_count'] = df['main_meal_count'].fillna(0) * 1.2
    df['revenue_per_customer'] = df['total_sales'] / (df['customer_count'] + 0.01)
    
    # Define peak periods (multiple definitions for comprehensive analysis)
    df['is_lunch_peak'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_dinner_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_any_peak'] = (df['is_lunch_peak'] | df['is_dinner_peak']).astype(int)
    
    # Calculate capacity utilization
    df['capacity_utilization'] = np.minimum(df['customer_count'] / (df['capacity_available'] + 1), 1.5)  # Cap at 150%
    
    # Business cost proxies
    # Understaffing cost proxy: when demand exceeds expected levels
    # Overstaffing cost proxy: when demand is much lower than expected
    
    print(f"Dataset shape: {df.shape}")
    print(f"Restaurant types: {df['restaurant_type'].unique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def define_peak_periods_and_volatility(df):
    """Define peak periods and calculate volatility metrics for business impact analysis."""
    
    print("\n" + "="*85)
    print("DEFINING PEAK PERIODS AND VOLATILITY FOR BUSINESS IMPACT")
    print("="*85)
    
    peak_volatility_analysis = {}
    
    # Sample for computational efficiency
    df_sample = df.sample(n=min(60000, len(df)), random_state=42)
    
    for restaurant_type in df_sample['restaurant_type'].unique():
        print(f"\n🏪 Analyzing {restaurant_type.upper()}...")
        
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant_type].copy()
        
        if len(rest_data) < 1000:
            print(f"   ⚠️  Insufficient data ({len(rest_data)} records)")
            continue
        
        # 1. Define peak periods using percentile-based approach
        demand_threshold_80 = rest_data['customer_count'].quantile(0.80)
        demand_threshold_90 = rest_data['customer_count'].quantile(0.90)
        demand_threshold_95 = rest_data['customer_count'].quantile(0.95)
        
        rest_data['is_high_demand'] = (rest_data['customer_count'] >= demand_threshold_80).astype(int)
        rest_data['is_peak_demand'] = (rest_data['customer_count'] >= demand_threshold_90).astype(int)
        rest_data['is_extreme_demand'] = (rest_data['customer_count'] >= demand_threshold_95).astype(int)
        
        # 2. Calculate volatility metrics for different periods
        # Overall volatility
        overall_mean = rest_data['customer_count'].mean()
        overall_std = rest_data['customer_count'].std()
        overall_cv = overall_std / overall_mean if overall_mean > 0 else 0
        
        # Peak period volatility
        peak_periods = rest_data[rest_data['is_any_peak'] == 1]
        off_peak_periods = rest_data[rest_data['is_any_peak'] == 0]
        
        peak_mean = peak_periods['customer_count'].mean() if len(peak_periods) > 0 else 0
        peak_std = peak_periods['customer_count'].std() if len(peak_periods) > 0 else 0
        peak_cv = peak_std / peak_mean if peak_mean > 0 else 0
        
        off_peak_mean = off_peak_periods['customer_count'].mean() if len(off_peak_periods) > 0 else 0
        off_peak_std = off_peak_periods['customer_count'].std() if len(off_peak_periods) > 0 else 0
        off_peak_cv = off_peak_std / off_peak_mean if off_peak_mean > 0 else 0
        
        # High demand period volatility
        high_demand_periods = rest_data[rest_data['is_high_demand'] == 1]
        normal_demand_periods = rest_data[rest_data['is_high_demand'] == 0]
        
        high_demand_std = high_demand_periods['customer_count'].std() if len(high_demand_periods) > 0 else 0
        normal_demand_std = normal_demand_periods['customer_count'].std() if len(normal_demand_periods) > 0 else 0
        
        # 3. Weekend peak analysis
        weekend_peaks = rest_data[(rest_data['is_weekend'] == 1) & (rest_data['is_any_peak'] == 1)]
        weekday_peaks = rest_data[(rest_data['is_weekend'] == 0) & (rest_data['is_any_peak'] == 1)]
        
        weekend_peak_cv = (weekend_peaks['customer_count'].std() / weekend_peaks['customer_count'].mean()) if len(weekend_peaks) > 10 and weekend_peaks['customer_count'].mean() > 0 else 0
        weekday_peak_cv = (weekday_peaks['customer_count'].std() / weekday_peaks['customer_count'].mean()) if len(weekday_peaks) > 10 and weekday_peaks['customer_count'].mean() > 0 else 0
        
        # 4. Revenue volatility during peaks
        peak_revenue_cv = (peak_periods['total_sales'].std() / peak_periods['total_sales'].mean()) if len(peak_periods) > 10 and peak_periods['total_sales'].mean() > 0 else 0
        off_peak_revenue_cv = (off_peak_periods['total_sales'].std() / off_peak_periods['total_sales'].mean()) if len(off_peak_periods) > 10 and off_peak_periods['total_sales'].mean() > 0 else 0
        
        # 5. Capacity utilization volatility
        peak_capacity_util = peak_periods['capacity_utilization'].mean() if len(peak_periods) > 0 else 0
        peak_capacity_std = peak_periods['capacity_utilization'].std() if len(peak_periods) > 0 else 0
        
        peak_volatility_analysis[restaurant_type] = {
            # Basic metrics
            'overall_mean_demand': round(overall_mean, 1),
            'overall_std_demand': round(overall_std, 1),
            'overall_cv': round(overall_cv, 3),
            
            # Peak vs off-peak
            'peak_mean_demand': round(peak_mean, 1),
            'peak_std_demand': round(peak_std, 1),
            'peak_cv': round(peak_cv, 3),
            'off_peak_mean_demand': round(off_peak_mean, 1),
            'off_peak_cv': round(off_peak_cv, 3),
            'peak_volatility_ratio': round(peak_cv / off_peak_cv if off_peak_cv > 0 else 1, 2),
            
            # High demand periods
            'high_demand_std': round(high_demand_std, 1),
            'normal_demand_std': round(normal_demand_std, 1),
            'high_demand_volatility_ratio': round(high_demand_std / normal_demand_std if normal_demand_std > 0 else 1, 2),
            
            # Weekend analysis
            'weekend_peak_cv': round(weekend_peak_cv, 3),
            'weekday_peak_cv': round(weekday_peak_cv, 3),
            'weekend_volatility_premium': round(weekend_peak_cv / weekday_peak_cv if weekday_peak_cv > 0 else 1, 2),
            
            # Revenue volatility
            'peak_revenue_cv': round(peak_revenue_cv, 3),
            'off_peak_revenue_cv': round(off_peak_revenue_cv, 3),
            'revenue_volatility_ratio': round(peak_revenue_cv / off_peak_revenue_cv if off_peak_revenue_cv > 0 else 1, 2),
            
            # Capacity metrics
            'peak_capacity_utilization': round(peak_capacity_util, 3),
            'peak_capacity_volatility': round(peak_capacity_std, 3),
            
            # Sample sizes
            'total_periods': len(rest_data),
            'peak_periods': len(peak_periods),
            'high_demand_periods': len(high_demand_periods),
            'weekend_peak_periods': len(weekend_peaks)
        }
        
        print(f"   📊 Overall CV: {overall_cv:.3f}")
        print(f"   🚀 Peak CV: {peak_cv:.3f} | Off-peak CV: {off_peak_cv:.3f}")
        print(f"   📈 Peak volatility is {(peak_cv / off_peak_cv):.1f}x higher than off-peak")
        print(f"   🎯 Peak periods: {len(peak_periods)} ({(len(peak_periods)/len(rest_data)*100):.1f}% of time)")
    
    return peak_volatility_analysis

def quantify_business_costs_impact(df, peak_volatility_analysis):
    """Quantify the business impact of demand volatility in terms of operational costs."""
    
    print("\n" + "="*85)
    print("QUANTIFYING BUSINESS IMPACT OF DEMAND VOLATILITY")
    print("="*85)
    
    business_impact_analysis = {}
    
    # Define cost assumptions (these would be calibrated with real business data)
    COST_ASSUMPTIONS = {
        'hourly_staff_cost': 15.0,  # USD per hour per staff member
        'understaffing_penalty_multiplier': 2.5,  # Lost revenue multiplier when understaffed
        'overstaffing_cost_multiplier': 1.2,  # Additional cost when overstaffed
        'customer_service_degradation_cost': 5.0,  # USD per affected customer
        'rush_hiring_premium': 1.5,  # Premium for last-minute staffing
        'avg_revenue_per_customer': 25.0  # Average revenue per customer
    }
    
    df_sample = df.sample(n=min(40000, len(df)), random_state=42)
    
    for restaurant_type, volatility_data in peak_volatility_analysis.items():
        print(f"\n💰 Business Impact Analysis: {restaurant_type.upper()}")
        print("-" * 70)
        
        rest_data = df_sample[df_sample['restaurant_type'] == restaurant_type].copy()
        
        if len(rest_data) < 500:
            continue
        
        # 1. STAFFING MISMATCH COSTS
        # Assume staffing is planned based on average demand, calculate costs of volatility
        
        # Define expected staffing levels (simplified model)
        rest_data['expected_demand'] = rest_data.groupby('hour')['customer_count'].transform('mean')
        rest_data['demand_deviation'] = rest_data['customer_count'] - rest_data['expected_demand']
        rest_data['abs_demand_deviation'] = rest_data['demand_deviation'].abs()
        
        # Staffing model: 1 staff per 10 customers (minimum 2 staff)
        rest_data['expected_staff'] = np.maximum(2, np.ceil(rest_data['expected_demand'] / 10))
        rest_data['actual_demand_staff_needed'] = np.maximum(2, np.ceil(rest_data['customer_count'] / 10))
        rest_data['staffing_gap'] = rest_data['actual_demand_staff_needed'] - rest_data['expected_staff']
        
        # Understaffing situations (negative gap)
        understaffed_periods = rest_data[rest_data['staffing_gap'] < 0]
        overstaffed_periods = rest_data[rest_data['staffing_gap'] > 0]
        
        # Calculate costs
        # Understaffing costs: lost revenue + service degradation
        if len(understaffed_periods) > 0:
            avg_understaffing = abs(understaffed_periods['staffing_gap'].mean())
            understaffing_frequency = len(understaffed_periods) / len(rest_data)
            
            # Lost revenue due to poor service/long waits
            lost_customers_per_period = understaffed_periods['staffing_gap'].abs() * 2  # 2 customers lost per missing staff
            total_lost_revenue = (lost_customers_per_period * COST_ASSUMPTIONS['avg_revenue_per_customer']).sum()
            avg_lost_revenue_per_period = total_lost_revenue / len(rest_data) * 0.5  # Per 30-min period
            
            # Service degradation costs
            service_degradation_cost = (understaffed_periods['customer_count'] * 
                                      COST_ASSUMPTIONS['customer_service_degradation_cost']).sum() / len(rest_data) * 0.5
        else:
            avg_understaffing = understaffing_frequency = 0
            avg_lost_revenue_per_period = service_degradation_cost = 0
        
        # Overstaffing costs: additional labor costs
        if len(overstaffed_periods) > 0:
            avg_overstaffing = overstaffed_periods['staffing_gap'].mean()
            overstaffing_frequency = len(overstaffed_periods) / len(rest_data)
            
            excess_labor_cost = (overstaffed_periods['staffing_gap'] * 
                               COST_ASSUMPTIONS['hourly_staff_cost'] * 0.5).sum() / len(rest_data) * 0.5
        else:
            avg_overstaffing = overstaffing_frequency = 0
            excess_labor_cost = 0
        
        # 2. PEAK PERIOD SPECIFIC COSTS
        peak_periods = rest_data[rest_data['is_any_peak'] == 1]
        
        if len(peak_periods) > 0:
            # Peak period volatility costs
            peak_staffing_variance = peak_periods['staffing_gap'].var()
            peak_demand_variance = peak_periods['customer_count'].var()
            
            # Rush staffing costs (when peaks are higher than expected)
            high_peak_periods = peak_periods[peak_periods['customer_count'] > peak_periods['customer_count'].quantile(0.8)]
            if len(high_peak_periods) > 0:
                rush_staffing_instances = len(high_peak_periods) / len(rest_data)
                rush_staffing_cost = (high_peak_periods['staffing_gap'].clip(lower=0) * 
                                    COST_ASSUMPTIONS['hourly_staff_cost'] * 
                                    COST_ASSUMPTIONS['rush_hiring_premium'] * 0.5).sum() / len(rest_data) * 0.5
            else:
                rush_staffing_instances = rush_staffing_cost = 0
            
            # Capacity constraint costs (when demand exceeds capacity)
            over_capacity_periods = peak_periods[peak_periods['capacity_utilization'] > 1.0]
            if len(over_capacity_periods) > 0:
                capacity_constraint_frequency = len(over_capacity_periods) / len(rest_data)
                # Lost customers due to capacity constraints
                excess_demand = over_capacity_periods['customer_count'] - over_capacity_periods['capacity_available']
                capacity_lost_revenue = (excess_demand * COST_ASSUMPTIONS['avg_revenue_per_customer']).sum() / len(rest_data) * 0.5
            else:
                capacity_constraint_frequency = capacity_lost_revenue = 0
        else:
            peak_staffing_variance = peak_demand_variance = 0
            rush_staffing_instances = rush_staffing_cost = 0
            capacity_constraint_frequency = capacity_lost_revenue = 0
        
        # 3. INVENTORY AND WASTE COSTS
        # Food waste when overstaffed for expected demand that doesn't materialize
        # Stockout costs when understaffed and can't serve customers
        
        # Simplified model: waste cost proportional to overstaffing, stockout cost to understaffing
        food_waste_cost = max(0, avg_overstaffing) * 8.0 * overstaffing_frequency  # $8 waste per excess staff per period
        stockout_cost = max(0, abs(avg_understaffing)) * 15.0 * understaffing_frequency  # $15 stockout cost per missing staff
        
        # 4. REPUTATION AND CUSTOMER SATISFACTION IMPACT
        # Based on service quality degradation during high volatility periods
        
        high_volatility_periods = rest_data[rest_data['abs_demand_deviation'] > rest_data['abs_demand_deviation'].quantile(0.8)]
        reputation_impact_frequency = len(high_volatility_periods) / len(rest_data)
        
        # Customer satisfaction degradation cost
        satisfaction_cost = len(high_volatility_periods) * 3.0 / len(rest_data) * 0.5  # $3 per period with high volatility
        
        # 5. TOTAL BUSINESS IMPACT CALCULATION
        total_volatility_cost_per_period = (
            avg_lost_revenue_per_period +
            service_degradation_cost +
            excess_labor_cost +
            rush_staffing_cost +
            capacity_lost_revenue +
            food_waste_cost +
            stockout_cost +
            satisfaction_cost
        )
        
        # Scale to daily and monthly estimates
        daily_volatility_cost = total_volatility_cost_per_period * 48  # 48 periods per day
        monthly_volatility_cost = daily_volatility_cost * 30
        annual_volatility_cost = monthly_volatility_cost * 12
        
        # Calculate as percentage of revenue
        avg_revenue_per_period = rest_data['total_sales'].mean()
        daily_revenue = avg_revenue_per_period * 48
        volatility_cost_percentage = (daily_volatility_cost / daily_revenue * 100) if daily_revenue > 0 else 0
        
        business_impact_analysis[restaurant_type] = {
            # Staffing mismatch metrics
            'avg_understaffing_staff': round(avg_understaffing, 2),
            'understaffing_frequency_pct': round(understaffing_frequency * 100, 1),
            'avg_overstaffing_staff': round(avg_overstaffing, 2),
            'overstaffing_frequency_pct': round(overstaffing_frequency * 100, 1),
            
            # Cost breakdown (per 30-min period)
            'lost_revenue_cost': round(avg_lost_revenue_per_period, 2),
            'service_degradation_cost': round(service_degradation_cost, 2),
            'excess_labor_cost': round(excess_labor_cost, 2),
            'rush_staffing_cost': round(rush_staffing_cost, 2),
            'capacity_lost_revenue': round(capacity_lost_revenue, 2),
            'food_waste_cost': round(food_waste_cost, 2),
            'stockout_cost': round(stockout_cost, 2),
            'satisfaction_cost': round(satisfaction_cost, 2),
            
            # Total impact
            'total_volatility_cost_per_period': round(total_volatility_cost_per_period, 2),
            'daily_volatility_cost': round(daily_volatility_cost, 2),
            'monthly_volatility_cost': round(monthly_volatility_cost, 2),
            'annual_volatility_cost': round(annual_volatility_cost, 2),
            
            # Revenue impact
            'avg_daily_revenue': round(daily_revenue, 2),
            'volatility_cost_as_pct_revenue': round(volatility_cost_percentage, 2),
            
            # Frequency metrics
            'rush_staffing_frequency_pct': round(rush_staffing_instances * 100, 1),
            'capacity_constraint_frequency_pct': round(capacity_constraint_frequency * 100, 1),
            'high_volatility_frequency_pct': round(reputation_impact_frequency * 100, 1),
            
            # Peak-specific metrics
            'peak_staffing_variance': round(peak_staffing_variance, 2),
            'peak_demand_variance': round(peak_demand_variance, 2)
        }
        
        print(f"   💸 Total Volatility Cost per Period: ${total_volatility_cost_per_period:.2f}")
        print(f"   📅 Estimated Daily Cost: ${daily_volatility_cost:.2f}")
        print(f"   📊 As % of Revenue: {volatility_cost_percentage:.2f}%")
        print(f"   ⚠️  Understaffing: {understaffing_frequency*100:.1f}% of periods")
        print(f"   💰 Overstaffing: {overstaffing_frequency*100:.1f}% of periods")
        print(f"   🚀 Rush Staffing Needed: {rush_staffing_instances*100:.1f}% of periods")
    
    return business_impact_analysis

def analyze_volatility_mitigation_opportunities(df, peak_volatility_analysis, business_impact_analysis):
    """Analyze opportunities to mitigate business impact through better forecasting and planning."""
    
    print("\n" + "="*85)
    print("VOLATILITY MITIGATION OPPORTUNITIES ANALYSIS")
    print("="*85)
    
    mitigation_analysis = {}
    
    for restaurant_type in business_impact_analysis.keys():
        print(f"\n🛡️  Mitigation Opportunities: {restaurant_type.upper()}")
        print("-" * 70)
        
        volatility_data = peak_volatility_analysis[restaurant_type]
        impact_data = business_impact_analysis[restaurant_type]
        
        # Current situation assessment
        current_cost = impact_data['daily_volatility_cost']
        current_cost_pct = impact_data['volatility_cost_as_pct_revenue']
        
        # 1. FORECASTING IMPROVEMENT POTENTIAL
        # Assume better forecasting can reduce volatility by 20-40% depending on current volatility level
        current_cv = volatility_data['overall_cv']
        
        if current_cv > 0.6:
            forecasting_improvement_potential = 0.4  # High volatility = high improvement potential
            improvement_difficulty = "High investment needed"
        elif current_cv > 0.4:
            forecasting_improvement_potential = 0.3
            improvement_difficulty = "Moderate investment"
        elif current_cv > 0.2:
            forecasting_improvement_potential = 0.2
            improvement_difficulty = "Low investment"
        else:
            forecasting_improvement_potential = 0.1
            improvement_difficulty = "Minimal gains possible"
        
        forecasting_savings = current_cost * forecasting_improvement_potential
        
        # 2. FLEXIBLE STAFFING POTENTIAL
        understaffing_freq = impact_data['understaffing_frequency_pct']
        overstaffing_freq = impact_data['overstaffing_frequency_pct']
        
        # Flexible staffing can reduce both under and overstaffing
        if understaffing_freq > 30 or overstaffing_freq > 30:
            flexible_staffing_improvement = 0.5  # Can reduce mismatch by 50%
            staffing_implementation_cost = "High"
        elif understaffing_freq > 15 or overstaffing_freq > 15:
            flexible_staffing_improvement = 0.3
            staffing_implementation_cost = "Medium"
        else:
            flexible_staffing_improvement = 0.15
            staffing_implementation_cost = "Low"
        
        staffing_related_costs = (impact_data['excess_labor_cost'] + 
                                impact_data['rush_staffing_cost'] + 
                                impact_data['lost_revenue_cost']) * 48  # Daily
        flexible_staffing_savings = staffing_related_costs * flexible_staffing_improvement
        
        # 3. CAPACITY OPTIMIZATION POTENTIAL
        capacity_constraint_freq = impact_data['capacity_constraint_frequency_pct']
        capacity_lost_revenue = impact_data['capacity_lost_revenue'] * 48  # Daily
        
        if capacity_constraint_freq > 10:
            capacity_optimization_potential = 0.6  # Can recover 60% of lost revenue
            capacity_investment_needed = "Significant"
        elif capacity_constraint_freq > 5:
            capacity_optimization_potential = 0.4
            capacity_investment_needed = "Moderate"
        else:
            capacity_optimization_potential = 0.2
            capacity_investment_needed = "Minor"
        
        capacity_savings = capacity_lost_revenue * capacity_optimization_potential
        
        # 4. INVENTORY/WASTE MANAGEMENT POTENTIAL
        waste_and_stockout_costs = (impact_data['food_waste_cost'] + 
                                  impact_data['stockout_cost']) * 48  # Daily
        
        # Better demand forecasting can significantly reduce waste and stockouts
        inventory_management_improvement = 0.6  # 60% reduction possible
        inventory_savings = waste_and_stockout_costs * inventory_management_improvement
        
        # 5. CUSTOMER SATISFACTION IMPROVEMENTS
        satisfaction_costs = impact_data['satisfaction_cost'] * 48  # Daily
        service_degradation_costs = impact_data['service_degradation_cost'] * 48
        
        # Better planning can improve service quality
        service_improvement_potential = 0.5  # 50% improvement
        service_savings = (satisfaction_costs + service_degradation_costs) * service_improvement_potential
        
        # 6. TOTAL MITIGATION POTENTIAL
        total_potential_savings = (forecasting_savings * 48 +  # Daily
                                 flexible_staffing_savings +
                                 capacity_savings +
                                 inventory_savings +
                                 service_savings)
        
        improvement_percentage = (total_potential_savings / current_cost) * 100 if current_cost > 0 else 0
        monthly_savings = total_potential_savings * 30
        annual_savings = monthly_savings * 12
        
        # 7. IMPLEMENTATION PRIORITIES
        priorities = []
        
        # Priority scoring based on impact and implementation difficulty
        if forecasting_savings * 48 > current_cost * 0.15:  # >15% of current cost
            priorities.append({
                'initiative': 'Advanced Forecasting System',
                'potential_daily_savings': forecasting_savings * 48,
                'implementation': improvement_difficulty,
                'timeframe': '3-6 months',
                'priority_score': 9
            })
        
        if flexible_staffing_savings > current_cost * 0.1:  # >10% of current cost
            priorities.append({
                'initiative': 'Flexible Staffing Model',
                'potential_daily_savings': flexible_staffing_savings,
                'implementation': staffing_implementation_cost,
                'timeframe': '2-4 months',
                'priority_score': 8
            })
        
        if capacity_savings > current_cost * 0.08:  # >8% of current cost
            priorities.append({
                'initiative': 'Dynamic Capacity Management',
                'potential_daily_savings': capacity_savings,
                'implementation': capacity_investment_needed,
                'timeframe': '6-12 months',
                'priority_score': 7
            })
        
        if inventory_savings > current_cost * 0.05:  # >5% of current cost
            priorities.append({
                'initiative': 'Smart Inventory Management',
                'potential_daily_savings': inventory_savings,
                'implementation': 'Medium',
                'timeframe': '1-3 months',
                'priority_score': 6
            })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        mitigation_analysis[restaurant_type] = {
            # Current situation
            'current_daily_cost': round(current_cost, 2),
            'current_cost_pct_revenue': round(current_cost_pct, 2),
            
            # Improvement potentials
            'forecasting_improvement_potential_pct': round(forecasting_improvement_potential * 100, 1),
            'forecasting_daily_savings': round(forecasting_savings * 48, 2),
            'flexible_staffing_daily_savings': round(flexible_staffing_savings, 2),
            'capacity_optimization_daily_savings': round(capacity_savings, 2),
            'inventory_management_daily_savings': round(inventory_savings, 2),
            'service_improvement_daily_savings': round(service_savings, 2),
            
            # Total potential
            'total_potential_daily_savings': round(total_potential_savings, 2),
            'total_improvement_percentage': round(improvement_percentage, 1),
            'potential_monthly_savings': round(monthly_savings, 2),
            'potential_annual_savings': round(annual_savings, 2),
            
            # Implementation priorities
            'top_priorities': priorities,
            'implementation_difficulty_assessment': improvement_difficulty
        }
        
        print(f"   💰 Current Daily Volatility Cost: ${current_cost:.2f}")
        print(f"   🎯 Total Potential Daily Savings: ${total_potential_savings:.2f} ({improvement_percentage:.1f}%)")
        print(f"   📈 Potential Annual Savings: ${annual_savings:,.2f}")
        print(f"   🏆 Top Priority: {priorities[0]['initiative'] if priorities else 'No major initiatives needed'}")
        
        if priorities:
            print(f"   📋 Implementation Roadmap:")
            for i, priority in enumerate(priorities[:3], 1):  # Top 3
                print(f"      {i}. {priority['initiative']}: ${priority['potential_daily_savings']:.2f}/day "
                      f"({priority['timeframe']})")
    
    return mitigation_analysis

def create_business_impact_visualizations(df, peak_volatility_analysis, business_impact_analysis, mitigation_analysis):
    """Create comprehensive visualizations for business impact analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Business Impact of Demand Volatility During Peak Periods', fontsize=16, y=0.98)
    
    # Visualization 1: Volatility Cost by Restaurant Type
    restaurants = list(business_impact_analysis.keys())
    daily_costs = [business_impact_analysis[r]['daily_volatility_cost'] for r in restaurants]
    cost_percentages = [business_impact_analysis[r]['volatility_cost_as_pct_revenue'] for r in restaurants]
    
    # Color code by cost level
    colors = []
    for cost in daily_costs:
        if cost > 200:
            colors.append('#d62728')  # Red - Very high cost
        elif cost > 100:
            colors.append('#ff7f0e')  # Orange - High cost
        elif cost > 50:
            colors.append('#2ca02c')  # Green - Moderate cost
        else:
            colors.append('#1f77b4')  # Blue - Low cost
    
    bars = axes[0,0].barh(range(len(restaurants)), daily_costs, color=colors)
    axes[0,0].set_yticks(range(len(restaurants)))
    axes[0,0].set_yticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=11)
    axes[0,0].set_xlabel('Daily Volatility Cost ($)', fontsize=12)
    axes[0,0].set_title('Daily Business Impact of Demand Volatility', fontsize=13, pad=15)
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Add cost labels and percentage
    for i, (bar, cost, pct) in enumerate(zip(bars, daily_costs, cost_percentages)):
        axes[0,0].text(cost + 5, bar.get_y() + bar.get_height()/2, f'${cost:.0f}\n({pct:.1f}%)', 
                      va='center', fontsize=9, weight='bold')
    
    # Visualization 2: Cost Breakdown by Component
    cost_components = ['lost_revenue_cost', 'excess_labor_cost', 'rush_staffing_cost', 
                      'capacity_lost_revenue', 'food_waste_cost', 'service_degradation_cost']
    component_labels = ['Lost Revenue', 'Excess Labor', 'Rush Staffing', 
                       'Capacity Constraints', 'Food Waste', 'Service Issues']
    
    # Create stacked bar chart
    bottom = np.zeros(len(restaurants))
    colors_comp = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71', '#95a5a6']
    
    for i, (comp, label, color) in enumerate(zip(cost_components, component_labels, colors_comp)):
        values = [business_impact_analysis[r].get(comp, 0) * 48 for r in restaurants]  # Convert to daily
        axes[0,1].bar(range(len(restaurants)), values, bottom=bottom, 
                     label=label, color=color, alpha=0.8)
        bottom += values
    
    axes[0,1].set_xticks(range(len(restaurants)))
    axes[0,1].set_xticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=10, rotation=45)
    axes[0,1].set_ylabel('Daily Cost ($)', fontsize=12)
    axes[0,1].set_title('Volatility Cost Breakdown by Component', fontsize=13, pad=15)
    axes[0,1].legend(loc='upper right', fontsize=9)
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Visualization 3: Mitigation Potential Analysis
    current_costs = [mitigation_analysis[r]['current_daily_cost'] for r in restaurants]
    potential_savings = [mitigation_analysis[r]['total_potential_daily_savings'] for r in restaurants]
    improvement_pcts = [mitigation_analysis[r]['total_improvement_percentage'] for r in restaurants]
    
    x = np.arange(len(restaurants))
    width = 0.35
    
    bars1 = axes[1,0].bar(x - width/2, current_costs, width, label='Current Daily Cost', 
                         color='lightcoral', alpha=0.8)
    bars2 = axes[1,0].bar(x + width/2, potential_savings, width, label='Potential Daily Savings', 
                         color='lightgreen', alpha=0.8)
    
    axes[1,0].set_xlabel('Restaurant Type', fontsize=12)
    axes[1,0].set_ylabel('Daily Cost/Savings ($)', fontsize=12)
    axes[1,0].set_title('Volatility Mitigation Potential', fontsize=13, pad=15)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([r.replace(' ', '\n') for r in restaurants], fontsize=10, rotation=45)
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Add percentage improvement labels
    for i, (bar1, bar2, pct) in enumerate(zip(bars1, bars2, improvement_pcts)):
        height = max(bar1.get_height(), bar2.get_height())
        axes[1,0].text(i, height + 10, f'{pct:.0f}%\nimprovement', 
                      ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Visualization 4: ROI Analysis - Annual Savings Potential
    annual_savings = [mitigation_analysis[r]['potential_annual_savings'] for r in restaurants]
    current_annual_costs = [mitigation_analysis[r]['current_daily_cost'] * 365 for r in restaurants]
    
    # Create scatter plot with bubble size representing improvement percentage
    scatter = axes[1,1].scatter(current_annual_costs, annual_savings, 
                               s=[pct*10 for pct in improvement_pcts], 
                               c=improvement_pcts, cmap='viridis', alpha=0.7)
    
    # Add restaurant labels
    for i, restaurant in enumerate(restaurants):
        axes[1,1].annotate(restaurant.replace(' ', '\n'), 
                          (current_annual_costs[i], annual_savings[i]), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=9, ha='left')
    
    # Add diagonal line showing break-even
    max_val = max(max(current_annual_costs), max(annual_savings))
    axes[1,1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Break-even')
    
    axes[1,1].set_xlabel('Current Annual Volatility Cost ($)', fontsize=12)
    axes[1,1].set_ylabel('Potential Annual Savings ($)', fontsize=12)
    axes[1,1].set_title('Annual ROI Potential from Volatility Reduction', fontsize=13, pad=15)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(fontsize=10)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[1,1], label='Improvement Potential (%)')
    
    # Add text box with key insights
    axes[1,1].text(0.02, 0.98, 'Bubble size = Improvement %\nAbove red line = Positive ROI', 
                   transform=axes[1,1].transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('demand_volatility_business_impact_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: demand_volatility_business_impact_analysis.png")

def main():
    """Main function for business impact of demand volatility analysis."""
    print("="*85)
    print("STINT PART 1: BUSINESS IMPACT OF DEMAND VOLATILITY ANALYSIS")
    print("="*85)
    
    # Load data
    df = load_and_prepare_data()
    
    # Define peak periods and calculate volatility
    peak_volatility_analysis = define_peak_periods_and_volatility(df)
    
    # Quantify business costs impact
    business_impact_analysis = quantify_business_costs_impact(df, peak_volatility_analysis)
    
    # Analyze mitigation opportunities
    mitigation_analysis = analyze_volatility_mitigation_opportunities(df, peak_volatility_analysis, business_impact_analysis)
    
    # Create visualizations
    create_business_impact_visualizations(df, peak_volatility_analysis, business_impact_analysis, mitigation_analysis)
    
    # Executive Summary
    print("\n" + "="*85)
    print("EXECUTIVE SUMMARY - BUSINESS IMPACT OF DEMAND VOLATILITY")
    print("="*85)
    
    # Calculate overall statistics
    total_restaurants = len(business_impact_analysis)
    avg_daily_cost = np.mean([analysis['daily_volatility_cost'] for analysis in business_impact_analysis.values()])
    avg_cost_percentage = np.mean([analysis['volatility_cost_as_pct_revenue'] for analysis in business_impact_analysis.values()])
    total_annual_impact = sum([analysis['annual_volatility_cost'] for analysis in business_impact_analysis.values()])
    total_potential_annual_savings = sum([analysis['potential_annual_savings'] for analysis in mitigation_analysis.values()])
    
    # Find highest impact restaurant
    highest_impact_restaurant = max(business_impact_analysis.items(), key=lambda x: x[1]['daily_volatility_cost'])
    lowest_impact_restaurant = min(business_impact_analysis.items(), key=lambda x: x[1]['daily_volatility_cost'])
    
    print(f"\n📊 OVERALL BUSINESS IMPACT:")
    print(f"• Average Daily Cost per Restaurant: ${avg_daily_cost:.2f}")
    print(f"• Average Cost as % of Revenue: {avg_cost_percentage:.2f}%")
    print(f"• Total Annual Impact Across All Types: ${total_annual_impact:,.2f}")
    print(f"• Highest Impact: {highest_impact_restaurant[0].title()} (${highest_impact_restaurant[1]['daily_volatility_cost']:.2f}/day)")
    print(f"• Lowest Impact: {lowest_impact_restaurant[0].title()} (${lowest_impact_restaurant[1]['daily_volatility_cost']:.2f}/day)")
    
    print(f"\n💰 MITIGATION POTENTIAL:")
    print(f"• Total Potential Annual Savings: ${total_potential_annual_savings:,.2f}")
    print(f"• Average Improvement Potential: {np.mean([analysis['total_improvement_percentage'] for analysis in mitigation_analysis.values()]):.1f}%")
    
    # Top cost drivers across all restaurants
    print(f"\n🔍 PRIMARY COST DRIVERS:")
    cost_drivers = ['lost_revenue_cost', 'excess_labor_cost', 'rush_staffing_cost', 
                   'capacity_lost_revenue', 'food_waste_cost', 'service_degradation_cost']
    driver_totals = {}
    
    for driver in cost_drivers:
        total = sum([analysis.get(driver, 0) * 48 * 365 for analysis in business_impact_analysis.values()])  # Annual
        driver_totals[driver] = total
    
    sorted_drivers = sorted(driver_totals.items(), key=lambda x: x[1], reverse=True)
    driver_names = {
        'lost_revenue_cost': 'Lost Revenue from Understaffing',
        'excess_labor_cost': 'Excess Labor Costs',
        'rush_staffing_cost': 'Rush Staffing Premiums',
        'capacity_lost_revenue': 'Capacity Constraint Losses',
        'food_waste_cost': 'Food Waste from Overstaffing',
        'service_degradation_cost': 'Service Quality Degradation'
    }
    
    for i, (driver, total) in enumerate(sorted_drivers[:3], 1):
        percentage = (total / sum(driver_totals.values())) * 100
        print(f"{i}. {driver_names[driver]}: ${total:,.2f} annually ({percentage:.1f}%)")
    
    # Strategic recommendations
    print(f"\n🚀 STRATEGIC RECOMMENDATIONS:")
    print("1. IMMEDIATE ACTIONS (0-3 months):")
    print("   • Implement basic demand forecasting improvements")
    print("   • Establish flexible staffing protocols for peak periods")
    print("   • Set up real-time demand monitoring during high-volatility periods")
    
    print("2. MEDIUM-TERM INITIATIVES (3-12 months):")
    print("   • Deploy advanced forecasting models with external factor integration")
    print("   • Implement dynamic staffing optimization systems")
    print("   • Develop capacity management strategies for constraint periods")
    
    print("3. LONG-TERM INVESTMENTS (12+ months):")
    print("   • Build predictive analytics platform for demand volatility management")
    print("   • Establish cross-restaurant resource sharing for peak periods")
    print("   • Integrate customer behavior analytics for demand pattern recognition")
    
    # Save comprehensive results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'executive_summary': {
            'total_restaurants_analyzed': total_restaurants,
            'avg_daily_cost_per_restaurant': round(avg_daily_cost, 2),
            'avg_cost_percentage_of_revenue': round(avg_cost_percentage, 2),
            'total_annual_impact': round(total_annual_impact, 2),
            'total_potential_annual_savings': round(total_potential_annual_savings, 2),
            'highest_impact_restaurant': highest_impact_restaurant[0],
            'lowest_impact_restaurant': lowest_impact_restaurant[0]
        },
        'peak_volatility_analysis': peak_volatility_analysis,
        'business_impact_analysis': business_impact_analysis,
        'mitigation_analysis': mitigation_analysis,
        'primary_cost_drivers': [{'driver': driver_names[driver], 'annual_cost': total, 
                                'percentage': (total / sum(driver_totals.values())) * 100} 
                               for driver, total in sorted_drivers],
        'strategic_recommendations': [
            "Implement tiered forecasting approach based on volatility levels",
            "Establish flexible staffing models with on-call personnel",
            "Deploy real-time demand monitoring and alert systems",
            "Develop capacity management strategies for peak periods",
            "Create cross-restaurant resource sharing capabilities",
            "Integrate external factor monitoring for proactive planning"
        ]
    }
    
    with open('demand_volatility_business_impact_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*85)
    print("ANALYSIS COMPLETE")
    print("="*85)
    print("📊 Results saved to:")
    print("  • demand_volatility_business_impact_results.json")
    print("  • demand_volatility_business_impact_analysis.png")
    print("="*85)

if __name__ == "__main__":
    main()