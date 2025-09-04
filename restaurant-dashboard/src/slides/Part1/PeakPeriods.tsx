import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import ClickableImage from '../../components/ClickableImage';

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
  background: linear-gradient(135deg, #8884d8 0%, #667eea 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
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
    color: #667eea;
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

const RestaurantGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const RestaurantCard = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #667eea;
`;

const RestaurantName = styled.h4`
  font-size: 1.2rem;
  color: #333;
  margin-bottom: 1rem;
  text-transform: capitalize;
`;

const PeakInfo = styled.div`
  margin-bottom: 1rem;
`;

const InfoRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f0f0f0;
  
  &:last-child {
    border-bottom: none;
  }
`;

const InfoLabel = styled.span`
  color: #666;
  font-size: 0.9rem;
`;

const InfoValue = styled.span`
  color: #333;
  font-weight: bold;
`;

const TopCombinations = styled.div`
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
`;

const ComboTitle = styled.h5`
  margin-bottom: 0.5rem;
  color: #333;
  font-size: 0.95rem;
`;

const ComboList = styled.ol`
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.85rem;
`;

const ComboItem = styled.li`
  margin-bottom: 0.25rem;
  color: #555;
`;

const KeyInsights = styled.div`
  background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
`;

const InsightTitle = styled.h3`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
`;

const InsightGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
`;

const InsightCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
`;

const InsightIcon = styled.div`
  font-size: 2rem;
  margin-bottom: 0.5rem;
`;

const InsightText = styled.div`
  font-size: 0.9rem;
  color: #333;
  font-weight: 500;
`;

interface PeakPeriodsData {
  analysis_timestamp: string;
  peak_analysis: Record<string, {
    peak_hours: number[];
    peak_days: string[];
    peak_months: number[];
    top_time_combinations: [string, number, number][];
    avg_peak_duration_hours: number;
    peak_intensity_multiplier: number;
    extreme_intensity_multiplier: number;
    peak_frequency_pct: number;
    demand_statistics: {
      normal_avg: number;
      peak_avg: number;
      extreme_avg: number;
      overall_std: number;
    };
  }>;
  pattern_analysis: Record<string, any>;
  recommendations: Array<{
    restaurant: string;
    recommendation: string;
    peak_hours: number[];
    intensity: number;
    duration: number;
    priority: string;
  }>;
  summary_statistics: {
    total_restaurants_analyzed: number;
    avg_peak_intensity: number;
    avg_peak_duration: number;
    avg_peak_frequency: number;
  };
}

const PeakPeriods: React.FC = () => {
  const [data, setData] = useState<PeakPeriodsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/peak_demand_periods_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded peak periods data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading peak periods analysis...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  const formatHour = (hour: number): string => `${Math.floor(hour).toString().padStart(2, '0')}:00`;
  const formatMonth = (month: number): string => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return months[Math.floor(month) - 1] || `Month ${month}`;
  };

  return (
    <Container>
      <Title>Peak Demand Periods Analysis</Title>
      
      <Question>
        Q: When are peak demand periods and how intense are they compared to normal periods?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Peak Classification: Used 80th, 90th, and 95th percentiles to define demand intensity levels</MethodItem>
          <MethodItem>Intensity Analysis: Calculated how much higher peak demand is compared to normal periods</MethodItem>
          <MethodItem>Temporal Patterns: Identified recurring peak hours, days, and seasonal patterns</MethodItem>
          <MethodItem>Duration & Frequency: Measured typical peak duration and how often they occur</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/peak_demand_periods_analysis.png`}
        alt="Peak Demand Periods Analysis - Intensity patterns, staffing requirements, and peak duration analysis across restaurant types"
      />

      <KeyInsights>
        <InsightTitle>🎯 Key Peak Period Insights</InsightTitle>
        <InsightGrid>
          <InsightCard>
            <InsightIcon>🍽️</InsightIcon>
            <InsightText>Universal peak hours: 1pm, 6-8pm across all restaurant types</InsightText>
          </InsightCard>
          <InsightCard>
            <InsightIcon>📅</InsightIcon>
            <InsightText>Weekend effect: Fri-Sun consistently show highest demand</InsightText>
          </InsightCard>
          <InsightCard>
            <InsightIcon>🚀</InsightIcon>
            <InsightText>Peak intensity: 1.8-2.1x normal demand during peaks</InsightText>
          </InsightCard>
          <InsightCard>
            <InsightIcon>⏱️</InsightIcon>
            <InsightText>Average peak duration: 0.7 hours (42 minutes)</InsightText>
          </InsightCard>
        </InsightGrid>
      </KeyInsights>

      <SummaryTitle>📊 Restaurant-Specific Peak Patterns</SummaryTitle>
      <RestaurantGrid>
        {Object.entries(data.peak_analysis).map(([restaurantType, analysis]) => (
          <RestaurantCard key={restaurantType}>
            <RestaurantName>{restaurantType}</RestaurantName>
            <PeakInfo>
              <InfoRow>
                <InfoLabel>🕐 Peak Hours</InfoLabel>
                <InfoValue>{analysis.peak_hours.map(formatHour).join(', ')}</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>📅 Peak Days</InfoLabel>
                <InfoValue>{analysis.peak_days.join(', ')}</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>📆 Peak Months</InfoLabel>
                <InfoValue>{analysis.peak_months.map(formatMonth).join(', ')}</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>🚀 Peak Intensity</InfoLabel>
                <InfoValue>{analysis.peak_intensity_multiplier}x normal</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>💥 Extreme Intensity</InfoLabel>
                <InfoValue>{analysis.extreme_intensity_multiplier}x normal</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>⏱️ Peak Duration</InfoLabel>
                <InfoValue>{analysis.avg_peak_duration_hours}h average</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>📈 Peak Frequency</InfoLabel>
                <InfoValue>{analysis.peak_frequency_pct}% of periods</InfoValue>
              </InfoRow>
            </PeakInfo>
            <TopCombinations>
              <ComboTitle>🔥 Top Peak Combinations:</ComboTitle>
              <ComboList>
                {analysis.top_time_combinations.slice(0, 3).map(([day, hour, demand], index) => (
                  <ComboItem key={index}>
                    {day} at {formatHour(hour)} - {Math.round(demand)} customers
                  </ComboItem>
                ))}
              </ComboList>
            </TopCombinations>
          </RestaurantCard>
        ))}
      </RestaurantGrid>

      <SummarySection>
        <SummaryTitle>🚨 Staffing Recommendations</SummaryTitle>
        <RestaurantGrid>
          {data.recommendations && data.recommendations.map((rec, index) => (
            <RestaurantCard key={index}>
              <RestaurantName>
                {rec.restaurant}
                <span style={{ 
                  marginLeft: '1rem', 
                  fontSize: '0.8rem', 
                  padding: '0.25rem 0.5rem',
                  borderRadius: '12px',
                  background: rec.priority === 'HIGH' ? '#ff4444' : '#ffbb28',
                  color: 'white'
                }}>
                  {rec.priority}
                </span>
              </RestaurantName>
              <div style={{ marginBottom: '1rem', color: '#555', fontSize: '0.95rem' }}>
                {rec.recommendation}
              </div>
              <InfoRow>
                <InfoLabel>📍 Peak Hours</InfoLabel>
                <InfoValue>{rec.peak_hours.map(formatHour).join(', ')}</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>🚀 Peak Intensity</InfoLabel>
                <InfoValue>{rec.intensity}x normal demand</InfoValue>
              </InfoRow>
              <InfoRow>
                <InfoLabel>⏱️ Duration</InfoLabel>
                <InfoValue>{rec.duration}h average</InfoValue>
              </InfoRow>
            </RestaurantCard>
          ))}
        </RestaurantGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>💡 Key Findings</SummaryTitle>
        <div style={{ display: 'grid', gap: '0.5rem' }}>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            All restaurant types show dual-peak patterns with lunch (1pm) and dinner (6-8pm) rushes
          </div>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            Weekend effect dominates: Friday-Sunday consistently generate highest demand across all categories
          </div>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            Peak intensity ranges from 1.8x (Fine Dining) to 2.1x (Casual Bistro, Fast Casual, Seafood) normal demand
          </div>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            Average peak duration is only {data.summary_statistics.avg_peak_duration} hours, requiring rapid staffing adjustments
          </div>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            Peaks occur in {data.summary_statistics.avg_peak_frequency.toFixed(1)}% of all time periods, making them predictable but intense
          </div>
          <div style={{ 
            padding: '0.75rem 1rem',
            background: '#f8f9fa',
            borderRadius: '6px',
            borderLeft: '3px solid #667eea',
            fontSize: '0.95rem',
            color: '#555'
          }}>
            High-intensity restaurants (Casual Bistro, Fast Casual, Seafood) need immediate staffing protocols for {data.summary_statistics.avg_peak_intensity.toFixed(1)}x demand spikes
          </div>
        </div>
      </SummarySection>
    </Container>
  );
};

export default PeakPeriods;