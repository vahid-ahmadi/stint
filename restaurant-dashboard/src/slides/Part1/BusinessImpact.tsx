import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import ClickableImage from '../../components/ClickableImage';
import InfoTooltip from '../../components/InfoTooltip';

const Container = styled.div`
  width: 100%;
  max-width: 100vw;
  padding: 1rem 2rem;
  box-sizing: border-box;
  overflow-x: hidden;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: #333;
  margin-bottom: 1rem;
`;

const Question = styled.div`
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #fff0f5 0%, #ffe0e8 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
`;

const MethodTitle = styled.h3`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
`;

const MethodList = styled.ul`
  list-style-type: none;
  padding: 0;
`;

const MethodItem = styled.li`
  padding: 0.5rem 0;
  padding-left: 1.5rem;
  position: relative;
  color: #555;
  
  &:before {
    content: "▶";
    position: absolute;
    left: 0;
    color: #f5576c;
  }
`;


const SummarySection = styled.div`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
`;

const SummaryTitle = styled.h3`
  color: #333;
  margin-bottom: 1.5rem;
  font-size: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ExecutiveSummaryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const SummaryCard = styled.div`
  text-align: center;
  padding: 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px;
`;

const CardIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
`;

const CardTitle = styled.h4`
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
`;

const CardValue = styled.div`
  font-size: 1.8rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
`;

const CardDescription = styled.div`
  font-size: 0.9rem;
  opacity: 0.9;
`;

const RestaurantGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const RestaurantCard = styled.div<{ impactLevel: 'high' | 'medium' | 'low' }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => 
    props.impactLevel === 'high' ? '#ff4444' :
    props.impactLevel === 'medium' ? '#ffa500' : 
    '#44aa44'};
`;

const RestaurantName = styled.h4`
  font-size: 1.3rem;
  color: #333;
  margin-bottom: 1rem;
  text-transform: capitalize;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ImpactLevel = styled.span<{ impactLevel: 'high' | 'medium' | 'low' }>`
  font-size: 0.9rem;
  font-weight: bold;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  color: white;
  background: ${props => 
    props.impactLevel === 'high' ? '#ff4444' :
    props.impactLevel === 'medium' ? '#ffa500' : 
    '#44aa44'};
`;

const CostBreakdown = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin: 1rem 0;
`;

const CostItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
  font-size: 0.85rem;
  border-bottom: 1px solid #f0f0f0;
  
  &:last-child {
    border-bottom: none;
  }
`;

const CostLabel = styled.span`
  color: #666;
`;

const CostValue = styled.span`
  font-weight: bold;
  color: #333;
`;

const AnnualCostHighlight = styled.div`
  background: #fff5f5;
  padding: 1rem;
  border-radius: 8px;
  margin: 1rem 0;
  border-left: 4px solid #ff4444;
`;

const AnnualCostValue = styled.div`
  font-size: 1.5rem;
  font-weight: bold;
  color: #ff4444;
  margin-bottom: 0.5rem;
`;

const SavingsSection = styled.div`
  background: #f0fff4;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  border-left: 4px solid #22c55e;
`;

const SavingsTitle = styled.div`
  font-weight: bold;
  color: #16a34a;
  margin-bottom: 0.5rem;
`;

const SavingsValue = styled.div`
  font-size: 1.2rem;
  font-weight: bold;
  color: #16a34a;
`;

const MitigationStrategies = styled.div`
  display: grid;
  gap: 0.5rem;
  margin-top: 1rem;
`;

const StrategyItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 4px;
  font-size: 0.9rem;
`;

const StrategyName = styled.span`
  color: #555;
`;

const StrategyValue = styled.span`
  font-weight: bold;
  color: #16a34a;
`;

const ComparisonTable = styled.div`
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const TableHeader = styled.div`
  background: #f5576c;
  color: white;
  padding: 1rem;
  font-weight: bold;
  display: grid;
  grid-template-columns: 2fr repeat(5, 1fr);
  gap: 1rem;
  font-size: 0.9rem;
`;

const TableRow = styled.div<{ impactLevel: 'high' | 'medium' | 'low' }>`
  display: grid;
  grid-template-columns: 2fr repeat(5, 1fr);
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid #eee;
  background: ${props => 
    props.impactLevel === 'high' ? '#fff5f5' :
    props.impactLevel === 'medium' ? '#fff8f0' :
    '#f9fff9'};
  font-size: 0.85rem;
  
  &:last-child {
    border-bottom: none;
  }
`;

const TableCell = styled.div`
  display: flex;
  align-items: center;
  color: #333;
`;

interface BusinessImpactData {
  analysis_timestamp: string;
  executive_summary: {
    total_restaurants_analyzed: number;
    avg_daily_cost_per_restaurant: number;
    avg_cost_percentage_of_revenue: number;
    total_annual_impact: number;
    total_potential_annual_savings: number;
    highest_impact_restaurant: string;
    lowest_impact_restaurant: string;
  };
  business_impact_analysis: Record<string, {
    avg_understaffing_staff: number;
    understaffing_frequency_pct: number;
    avg_overstaffing_staff: number;
    overstaffing_frequency_pct: number;
    lost_revenue_cost: number;
    service_degradation_cost: number;
    excess_labor_cost: number;
    rush_staffing_cost: number;
    capacity_lost_revenue: number;
    food_waste_cost: number;
    stockout_cost: number;
    satisfaction_cost: number;
    total_volatility_cost_per_period: number;
    daily_volatility_cost: number;
    monthly_volatility_cost: number;
    annual_volatility_cost: number;
    avg_daily_revenue: number;
    volatility_cost_as_pct_revenue: number;
    rush_staffing_frequency_pct: number;
    capacity_constraint_frequency_pct: number;
    high_volatility_frequency_pct: number;
    peak_staffing_variance: number;
    peak_demand_variance: number;
  }>;
  mitigation_analysis: Record<string, {
    current_daily_cost: number;
    current_cost_pct_revenue: number;
    forecasting_improvement_potential_pct: number;
    forecasting_daily_savings: number;
    flexible_staffing_daily_savings: number;
    capacity_optimization_daily_savings: number;
    inventory_management_daily_savings: number;
    service_improvement_daily_savings: number;
    total_potential_daily_savings: number;
    potential_monthly_savings: number;
    potential_annual_savings: number;
  }>;
}

const getImpactLevel = (costPercentage: number): 'high' | 'medium' | 'low' => {
  if (costPercentage > 6) return 'high';
  if (costPercentage > 4) return 'medium';
  return 'low';
};

const formatCurrency = (value: number | undefined): string => {
  if (value === undefined || value === null || isNaN(value)) return '$0';
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
};

const BusinessImpact: React.FC = () => {
  const [data, setData] = useState<BusinessImpactData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/demand_volatility_business_impact_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded business impact data:', data);
        console.log('Executive summary:', data.executive_summary);
        console.log('Business impact analysis keys:', Object.keys(data.business_impact_analysis || {}));
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading business impact analysis...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  // Sort restaurants by impact level
  const sortedRestaurants = Object.entries(data.business_impact_analysis)
    .sort(([,a], [,b]) => b.volatility_cost_as_pct_revenue - a.volatility_cost_as_pct_revenue);

  return (
    <Container>
      <Title>Business Impact Analysis</Title>
      
      <Question>
        Q: Quantify the financial impact of demand volatility and identify cost reduction opportunities
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Cost Quantification: Calculated direct costs from demand volatility (understaffing, overstaffing, lost revenue)</MethodItem>
          <MethodItem>Impact Analysis: Measured frequency and financial impact of staffing mismatches</MethodItem>
          <MethodItem>Opportunity Assessment: Identified potential savings from better demand management</MethodItem>
          <MethodItem>ROI Calculation: Estimated return on investment for improvement strategies</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/demand_volatility_business_impact_analysis.png`}
        alt="Business Impact Analysis of Demand Volatility - Financial costs, staffing challenges, and improvement opportunities"
      />

      <SummarySection>
        <SummaryTitle>💡 Strategic Recommendations</SummaryTitle>
        <div style={{ display: 'grid', gap: '1rem' }}>
          <div style={{ 
            padding: '1rem', 
            background: '#fff5f5', 
            borderRadius: '8px',
            borderLeft: '4px solid #ef4444'
          }}>
            <strong>Highest Impact Restaurant:</strong> {data.executive_summary.highest_impact_restaurant} shows the highest volatility costs at {data.business_impact_analysis[data.executive_summary.highest_impact_restaurant]?.volatility_cost_as_pct_revenue?.toFixed(1) || '0'}% of revenue, requiring immediate attention.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#fffbeb', 
            borderRadius: '8px',
            borderLeft: '4px solid #f59e0b'
          }}>
            <strong>Major Cost Drivers:</strong> Service degradation and lost revenue account for the largest volatility costs, suggesting customer impact is the primary concern.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#f0fff4', 
            borderRadius: '8px',
            borderLeft: '4px solid #22c55e'
          }}>
            <strong>Improvement Opportunity:</strong> {formatCurrency(data.executive_summary.total_potential_annual_savings)} in potential annual savings through better forecasting, flexible staffing, and operational optimization.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#f0f9ff', 
            borderRadius: '8px',
            borderLeft: '4px solid #3b82f6'
          }}>
            <strong>ROI Focus:</strong> Invest in demand forecasting technology first - it offers the highest return across all restaurant types with 30%+ improvement potential.
          </div>
        </div>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>🎯 Executive Summary</SummaryTitle>
        <ExecutiveSummaryGrid>
          <SummaryCard>
            <CardIcon>💰</CardIcon>
            <CardTitle>Total Annual Impact</CardTitle>
            <CardValue>{formatCurrency(data.executive_summary.total_annual_impact)}</CardValue>
            <CardDescription>Across all restaurant types</CardDescription>
          </SummaryCard>
          <SummaryCard>
            <CardIcon>📊</CardIcon>
            <CardTitle>Avg Daily Cost</CardTitle>
            <CardValue>{formatCurrency(data.executive_summary.avg_daily_cost_per_restaurant)}</CardValue>
            <CardDescription>Per restaurant per day</CardDescription>
          </SummaryCard>
          <SummaryCard>
            <CardIcon>📈</CardIcon>
            <CardTitle>Revenue Impact</CardTitle>
            <CardValue>{data.executive_summary.avg_cost_percentage_of_revenue.toFixed(1)}%</CardValue>
            <CardDescription>Of daily revenue</CardDescription>
          </SummaryCard>
          <SummaryCard>
            <CardIcon>💡</CardIcon>
            <CardTitle>Savings Potential</CardTitle>
            <CardValue>{formatCurrency(data.executive_summary.total_potential_annual_savings)}</CardValue>
            <CardDescription>Annual improvement opportunity</CardDescription>
          </SummaryCard>
        </ExecutiveSummaryGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>🏪 Restaurant-Specific Impact Analysis</SummaryTitle>
        <RestaurantGrid>
          {Object.entries(data.business_impact_analysis).map(([restaurantType, analysis]) => {
            const impactLevel = getImpactLevel(analysis.volatility_cost_as_pct_revenue);
            const mitigation = data.mitigation_analysis[restaurantType];
            
            return (
              <RestaurantCard key={restaurantType} impactLevel={impactLevel}>
                <RestaurantName>
                  {restaurantType}
                  <ImpactLevel impactLevel={impactLevel}>
                    {impactLevel.toUpperCase()} IMPACT
                  </ImpactLevel>
                </RestaurantName>
                
                <AnnualCostHighlight>
                  <div style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.25rem', display: 'flex', alignItems: 'center' }}>
                    Annual Volatility Cost
                    <InfoTooltip text="Total yearly cost of demand unpredictability, including understaffing losses, overstaffing waste, rush expenses, and service degradation. Calculated by aggregating daily volatility costs across all operational periods." />
                  </div>
                  <AnnualCostValue>{formatCurrency(analysis.annual_volatility_cost)}</AnnualCostValue>
                  <div style={{ fontSize: '0.85rem', color: '#666' }}>
                    {analysis.volatility_cost_as_pct_revenue.toFixed(1)}% of revenue
                  </div>
                </AnnualCostHighlight>
                
                <CostBreakdown>
                  <CostItem>
                    <CostLabel style={{ display: 'flex', alignItems: 'center' }}>
                      🏃 Lost Revenue
                      <InfoTooltip text="Revenue lost due to understaffing during peak demand periods. Calculated based on missed customer opportunities when demand exceeds available service capacity." />
                    </CostLabel>
                    <CostValue>{formatCurrency(analysis.lost_revenue_cost)}</CostValue>
                  </CostItem>
                  <CostItem>
                    <CostLabel style={{ display: 'flex', alignItems: 'center' }}>
                      ⚠️ Service Issues
                      <InfoTooltip text="Cost of service quality degradation during volatile periods, including customer dissatisfaction, longer wait times, and potential reputation impact measured in lost future revenue." />
                    </CostLabel>
                    <CostValue>{formatCurrency(analysis.service_degradation_cost)}</CostValue>
                  </CostItem>
                  <CostItem>
                    <CostLabel style={{ display: 'flex', alignItems: 'center' }}>
                      👥 Excess Labor
                      <InfoTooltip text="Cost of overstaffing during low-demand periods, including unnecessary wages, benefits, and productivity losses when staff capacity exceeds customer demand." />
                    </CostLabel>
                    <CostValue>{formatCurrency(analysis.excess_labor_cost)}</CostValue>
                  </CostItem>
                  <CostItem>
                    <CostLabel style={{ display: 'flex', alignItems: 'center' }}>
                      ⚡ Rush Staffing
                      <InfoTooltip text="Premium costs for emergency staffing solutions during unexpected demand spikes, including overtime pay, last-minute shift coverage, and temporary worker premiums." />
                    </CostLabel>
                    <CostValue>{formatCurrency(analysis.rush_staffing_cost)}</CostValue>
                  </CostItem>
                </CostBreakdown>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', margin: '1rem 0', fontSize: '0.85rem' }}>
                  <div>
                    <strong>Understaffing:</strong><br/>
                    {analysis.understaffing_frequency_pct.toFixed(1)}% of periods<br/>
                    Avg {analysis.avg_understaffing_staff.toFixed(1)} staff short
                  </div>
                  <div>
                    <strong>Overstaffing:</strong><br/>
                    {analysis.overstaffing_frequency_pct.toFixed(1)}% of periods<br/>
                    Avg {analysis.avg_overstaffing_staff.toFixed(1)} staff excess
                  </div>
                </div>
                
                {mitigation && (
                  <SavingsSection>
                    <SavingsTitle style={{ display: 'flex', alignItems: 'center' }}>
                      💡 Improvement Opportunities
                      <InfoTooltip text="Estimated annual cost savings achievable through operational improvements. Based on reducing volatility costs through better forecasting, flexible staffing models, capacity optimization, and service quality enhancements." />
                    </SavingsTitle>
                    <SavingsValue>
                      {formatCurrency(mitigation.potential_annual_savings)}/year potential savings
                    </SavingsValue>
                    <MitigationStrategies>
                      <StrategyItem>
                        <StrategyName style={{ display: 'flex', alignItems: 'center' }}>
                          🔮 Better Forecasting
                          <InfoTooltip text="Savings from advanced demand prediction systems that reduce forecasting errors by 30%. Includes ML models, real-time data integration, and predictive analytics to optimize staffing decisions." />
                        </StrategyName>
                        <StrategyValue>{formatCurrency(mitigation.forecasting_daily_savings * 365)}/year</StrategyValue>
                      </StrategyItem>
                      <StrategyItem>
                        <StrategyName style={{ display: 'flex', alignItems: 'center' }}>
                          👥 Flexible Staffing
                          <InfoTooltip text="Cost reductions through on-demand staffing models, cross-training programs, and flexible work arrangements that allow rapid adjustment to demand fluctuations." />
                        </StrategyName>
                        <StrategyValue>{formatCurrency(mitigation.flexible_staffing_daily_savings * 365)}/year</StrategyValue>
                      </StrategyItem>
                      <StrategyItem>
                        <StrategyName style={{ display: 'flex', alignItems: 'center' }}>
                          📈 Capacity Optimization
                          <InfoTooltip text="Savings from optimizing seating arrangements, kitchen workflows, and service processes to handle demand variations more efficiently without additional staff." />
                        </StrategyName>
                        <StrategyValue>{formatCurrency(mitigation.capacity_optimization_daily_savings * 365)}/year</StrategyValue>
                      </StrategyItem>
                      <StrategyItem>
                        <StrategyName style={{ display: 'flex', alignItems: 'center' }}>
                          🍽️ Service Improvement
                          <InfoTooltip text="Cost reductions through service quality enhancements that reduce customer wait times, improve satisfaction, and minimize volatility-related service failures." />
                        </StrategyName>
                        <StrategyValue>{formatCurrency(mitigation.service_improvement_daily_savings * 365)}/year</StrategyValue>
                      </StrategyItem>
                    </MitigationStrategies>
                  </SavingsSection>
                )}
              </RestaurantCard>
            );
          })}
        </RestaurantGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>📊 Impact Comparison</SummaryTitle>
        <ComparisonTable>
          <TableHeader>
            <div>Restaurant Type</div>
            <div>Annual Cost</div>
            <div>% Revenue</div>
            <div>Daily Cost</div>
            <div>Understaffing</div>
            <div>Savings Potential</div>
          </TableHeader>
          {sortedRestaurants.map(([restaurantType, analysis]) => {
            const impactLevel = getImpactLevel(analysis.volatility_cost_as_pct_revenue);
            const mitigation = data.mitigation_analysis[restaurantType];
            
            return (
              <TableRow key={restaurantType} impactLevel={impactLevel}>
                <TableCell style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                  {restaurantType}
                </TableCell>
                <TableCell>{formatCurrency(analysis.annual_volatility_cost)}</TableCell>
                <TableCell>{analysis.volatility_cost_as_pct_revenue.toFixed(1)}%</TableCell>
                <TableCell>{formatCurrency(analysis.daily_volatility_cost)}</TableCell>
                <TableCell>{analysis.understaffing_frequency_pct.toFixed(1)}%</TableCell>
                <TableCell>
                  {mitigation ? formatCurrency(mitigation.potential_annual_savings) : 'N/A'}
                </TableCell>
              </TableRow>
            );
          })}
        </ComparisonTable>
      </SummarySection>

    </Container>
  );
};

export default BusinessImpact;