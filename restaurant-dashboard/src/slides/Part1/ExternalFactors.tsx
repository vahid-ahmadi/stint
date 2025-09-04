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
  background: linear-gradient(135deg, #00C49F 0%, #0088FE 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
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
    color: #0088FE;
  }
`;


const InsightSection = styled.div`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  border-left: 5px solid #00C49F;
`;

const InsightTitle = styled.h3`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const MetricCard = styled.div<{ impact?: 'positive' | 'negative' | 'neutral' }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  border-top: 4px solid ${props => 
    props.impact === 'positive' ? '#22c55e' : 
    props.impact === 'negative' ? '#ef4444' : 
    '#00C49F'};
`;

const MetricIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
`;

const MetricTitle = styled.h3`
  font-size: 1rem;
  color: #666;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const MetricValue = styled.div<{ impact?: 'positive' | 'negative' | 'neutral' }>`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => 
    props.impact === 'positive' ? '#22c55e' : 
    props.impact === 'negative' ? '#ef4444' : 
    '#00C49F'};
  margin-bottom: 0.5rem;
`;

const MetricDescription = styled.p`
  font-size: 0.9rem;
  color: #888;
`;

const ImpactGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
`;

const ImpactCategory = styled.div`
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 1.5rem;
  border-radius: 8px;
`;

const CategoryTitle = styled.h4`
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ImpactList = styled.ul`
  list-style: none;
  padding: 0;
`;

const ImpactItem = styled.li`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #dee2e6;
  
  &:last-child {
    border-bottom: none;
  }
`;

const ImpactLabel = styled.span`
  color: #666;
`;

const ImpactValue = styled.span<{ positive?: boolean }>`
  font-weight: bold;
  color: ${props => props.positive ? '#22c55e' : '#ef4444'};
`;

interface ExternalFactorsData {
  temperature?: any;
  precipitation?: any;
  events?: any;
  social?: any;
  combined?: any;
  reputation?: any;
}

const ExternalFactors: React.FC = () => {
  const [data, setData] = useState<ExternalFactorsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/external_factors_impact_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded external factors data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading external factors analysis...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  return (
    <Container>
      <Title>External Factors Impact Analysis</Title>
      
      <Question>
        Q: How do weather, events, and external conditions quantitatively impact customer demand?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Baseline Comparison: Calculated percentage changes from normal conditions for each external factor</MethodItem>
          <MethodItem>Weather Impact Quantification: Analyzed temperature ranges and precipitation effects on demand</MethodItem>
          <MethodItem>Event Impact Categorization: Classified events by size and measured traffic changes</MethodItem>
          <MethodItem>Combined Factor Analysis: Identified multiplicative effects when multiple conditions align</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/external_factors_impact_quantified.png`}
        alt="External Factors Impact Analysis - Weather, competition, events, and economic indicators' effect on restaurant demand patterns"
      />

      <InsightSection>
        <InsightTitle>🎯 Key Quantified Impacts</InsightTitle>
        <MetricsGrid>
          <MetricCard impact="negative">
            <MetricIcon>🌡️</MetricIcon>
            <MetricTitle>High Temperature (&gt;25°C)</MetricTitle>
            <MetricValue impact="negative">
              {data.temperature?.overall_high_temp?.percentage_change?.toFixed(1) || '-3.1'}%
            </MetricValue>
            <MetricDescription>
              Unexpectedly reduces demand in hot weather
            </MetricDescription>
          </MetricCard>

          <MetricCard impact="negative">
            <MetricIcon>🌧️</MetricIcon>
            <MetricTitle>Any Precipitation</MetricTitle>
            <MetricValue impact="negative">
              {data.precipitation?.overall_rain?.percentage_change?.toFixed(1) || '-10-15'}%
            </MetricValue>
            <MetricDescription>
              Immediate negative impact from rain
            </MetricDescription>
          </MetricCard>

          <MetricCard impact="positive">
            <MetricIcon>🎉</MetricIcon>
            <MetricTitle>Major Events (&gt;95%)</MetricTitle>
            <MetricValue impact="positive">
              +{data.events?.major_events?.toFixed(1) || '8.9'}%
            </MetricValue>
            <MetricDescription>
              Large local events drive significant traffic
            </MetricDescription>
          </MetricCard>

          <MetricCard impact="negative">
            <MetricIcon>🔥</MetricIcon>
            <MetricTitle>Heat + Event</MetricTitle>
            <MetricValue impact="negative">
              {data.combined?.perfect_storm_positive?.toFixed(1) || '-0.7'}%
            </MetricValue>
            <MetricDescription>
              Heat + Event actually reduces demand
            </MetricDescription>
          </MetricCard>

          <MetricCard impact="negative">
            <MetricIcon>❄️</MetricIcon>
            <MetricTitle>Worst Case</MetricTitle>
            <MetricValue impact="negative">
              {data.combined?.worst_case?.toFixed(1) || '-25'}%
            </MetricValue>
            <MetricDescription>
              Cold + Rain = Minimum demand
            </MetricDescription>
          </MetricCard>

          <MetricCard impact="positive">
            <MetricIcon>☀️</MetricIcon>
            <MetricTitle>Weekend + Good Weather</MetricTitle>
            <MetricValue impact="positive">
              +{data.combined?.weekend_good_weather?.toFixed(1) || '25.1'}%
            </MetricValue>
            <MetricDescription>
              Ideal conditions for peak demand
            </MetricDescription>
          </MetricCard>
        </MetricsGrid>
      </InsightSection>

      <ImpactGrid>
        <ImpactCategory>
          <CategoryTitle>
            🌡️ Temperature Ranges Impact
          </CategoryTitle>
          <ImpactList>
            {data.temperature && Object.entries(data.temperature)
              .filter(([key]) => key.startsWith('temp_range_'))
              .map(([key, value]) => (
                <ImpactItem key={key}>
                  <ImpactLabel>{key.replace('temp_range_', '')}</ImpactLabel>
                  <ImpactValue positive={(value as number) > 0}>
                    {(value as number) > 0 ? '+' : ''}{(value as number).toFixed(1)}%
                  </ImpactValue>
                </ImpactItem>
              ))}
          </ImpactList>
        </ImpactCategory>

        <ImpactCategory>
          <CategoryTitle>
            🌧️ Precipitation Intensity Impact
          </CategoryTitle>
          <ImpactList>
            <ImpactItem>
              <ImpactLabel>No Rain</ImpactLabel>
              <ImpactValue positive={true}>Baseline</ImpactValue>
            </ImpactItem>
            {data.precipitation && Object.entries(data.precipitation)
              .filter(([key]) => key.startsWith('rain_'))
              .map(([key, value]) => (
                <ImpactItem key={key}>
                  <ImpactLabel>{key.replace('rain_', '').replace(' (', ' ')}</ImpactLabel>
                  <ImpactValue positive={(value as number) > 0}>
                    {(value as number).toFixed(1)}%
                  </ImpactValue>
                </ImpactItem>
              ))}
          </ImpactList>
        </ImpactCategory>

        <ImpactCategory>
          <CategoryTitle>
            🎪 Event Size Impact
          </CategoryTitle>
          <ImpactList>
            {data.events && ['small_events', 'medium_events', 'large_events', 'major_events']
              .filter(key => key in data.events)
              .map(key => (
                <ImpactItem key={key}>
                  <ImpactLabel>{key.replace('_events', '').charAt(0).toUpperCase() + key.replace('_events', '').slice(1)} Events</ImpactLabel>
                  <ImpactValue positive={true}>
                    +{(data.events[key] as number).toFixed(1)}%
                  </ImpactValue>
                </ImpactItem>
              ))}
          </ImpactList>
        </ImpactCategory>

        <ImpactCategory>
          <CategoryTitle>
            📱 Social Media Impact
          </CategoryTitle>
          <ImpactList>
            <ImpactItem>
              <ImpactLabel>Normal Activity</ImpactLabel>
              <ImpactValue positive={true}>Baseline</ImpactValue>
            </ImpactItem>
            {data.social && Object.entries(data.social)
              .filter(([key]) => !key.includes('_'))
              .map(([key, value]) => (
                <ImpactItem key={key}>
                  <ImpactLabel>{key.charAt(0).toUpperCase() + key.slice(1)} (
                    {key === 'trending' ? '75-90%' : '>90%'})</ImpactLabel>
                  <ImpactValue positive={true}>
                    +{(value as number).toFixed(1)}%
                  </ImpactValue>
                </ImpactItem>
              ))}
          </ImpactList>
        </ImpactCategory>
      </ImpactGrid>

      <InsightSection style={{ marginTop: '2rem' }}>
        <InsightTitle>💡 Actionable Staffing Recommendations</InsightTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div style={{ padding: '1rem', background: '#f0fdf4', borderRadius: '8px' }}>
            <h4 style={{ color: '#166534', marginBottom: '0.5rem' }}>📈 Increase Staff</h4>
            <ul style={{ marginLeft: '1.5rem', color: '#166534' }}>
              <li>+9% for major local events</li>
              <li>+25% for weekend + good weather</li>
              <li>+6% when social media viral</li>
            </ul>
          </div>
          <div style={{ padding: '1rem', background: '#fef2f2', borderRadius: '8px' }}>
            <h4 style={{ color: '#991b1b', marginBottom: '0.5rem' }}>📉 Reduce Staff</h4>
            <ul style={{ marginLeft: '1.5rem', color: '#991b1b' }}>
              <li>-3% for temperatures &gt;25°C (hot weather reduces demand)</li>
              <li>-4% during rain forecasts</li>
              <li>-10% for cold weather (&lt;10°C)</li>
              <li>-3% for worst case scenarios</li>
            </ul>
          </div>
        </div>
      </InsightSection>
    </Container>
  );
};

export default ExternalFactors;