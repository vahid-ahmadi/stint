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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const Card = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #667eea;
`;

const RestaurantName = styled.h3`
  font-size: 1.2rem;
  font-weight: bold;
  color: #333;
  margin-bottom: 1rem;
  text-transform: capitalize;
`;

const DriverList = styled.ul`
  list-style: none;
  padding: 0;
`;

const Driver = styled.li`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f0f0f0;
  
  &:last-child {
    border-bottom: none;
  }
`;

const DriverName = styled.span`
  color: #666;
  text-transform: capitalize;
`;

const DriverValue = styled.span`
  font-weight: bold;
  color: #667eea;
`;

const ChartContainer = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
`;

interface AnalysisData {
  drivers: Record<string, Record<string, number>>;
  impacts: Record<string, string>;
  peak_analysis: Record<string, any>;
  high_volatility_periods: string[];
}

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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

const InsightSection = styled.div`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  border-left: 5px solid #667eea;
`;

const InsightTitle = styled.h3`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
`;

const InsightGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 1.5rem;
`;

const InsightCard = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1.2rem;
  border-radius: 8px;
  
  h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
  }
  
  p {
    margin: 0;
    font-size: 0.95rem;
    opacity: 0.95;
  }
`;


const DemandDrivers: React.FC = () => {
  const [data, setData] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/part1_analysis_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading analysis data...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  return (
    <Container>
      <Title>Primary Demand Drivers Analysis</Title>
      
      <Question>
        Q: What are the key factors driving customer demand across different restaurant types?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Correlation Analysis: Measured statistical relationships between external factors and customer demand</MethodItem>
          <MethodItem>Impact Quantification: Calculated percentage changes in demand for key drivers (weekends, weather, events)</MethodItem>
          <MethodItem>Restaurant-Specific Analysis: Identified unique demand patterns for each restaurant type</MethodItem>
          <MethodItem>Time-based Patterns: Analyzed demand variations across different time periods</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/restaurant_demand_insights_part1.png`}
        alt="Restaurant Demand Analysis Visualizations - Weekend patterns, daily trends, and seasonal variations across all restaurant types"
      />

      <InsightSection>
        <InsightTitle>🎯 Key Demand Drivers</InsightTitle>
        <InsightGrid>
          <InsightCard>
            <h4>🌟 Weekend Boost</h4>
            <p>Strongest driver: +44% for Seafood, +36% for Casual Bistro. Plan weekend staffing accordingly.</p>
          </InsightCard>
          <InsightCard>
            <h4>🌡️ Weather Sensitivity</h4>
            <p>Hot weather ({'>'}25°C) reduces demand by 3.1%. Rain has stronger negative impact (-4.0%).</p>
          </InsightCard>
          <InsightCard>
            <h4>⏰ Time Patterns</h4>
            <p>Universal dinner peak 6-8pm across all types. Lunch peak varies by restaurant type.</p>
          </InsightCard>
          <InsightCard>
            <h4>🎪 Events Impact</h4>
            <p>Local events provide minimal boost (+1.0%). Focus resources on weather and weekend planning.</p>
          </InsightCard>
        </InsightGrid>
      </InsightSection>

      {data.drivers && (
        <Grid>
          {Object.entries(data.drivers).map(([type, drivers]) => {
            const topDrivers = Object.entries(drivers)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 5);
            
            return (
              <Card key={type}>
                <RestaurantName>{type}</RestaurantName>
                <DriverList>
                  {topDrivers.map(([driver, correlation]) => (
                    <Driver key={driver}>
                      <DriverName>{driver.replace(/_/g, ' ')}</DriverName>
                      <DriverValue style={{ color: correlation > 0 ? '#22c55e' : '#ef4444' }}>
                        {correlation > 0 ? '+' : ''}{(correlation * 100).toFixed(1)}%
                      </DriverValue>
                    </Driver>
                  ))}
                </DriverList>
              </Card>
            );
          })}
        </Grid>
      )}

      {data.impacts && (
        <ChartContainer>
          <h3 style={{ marginBottom: '1rem', color: '#333' }}>External Factor Impacts</h3>
          <Grid>
            {Object.entries(data.impacts).map(([factor, impact]) => (
              <Card key={factor} style={{ borderLeftColor: impact.includes('+') || impact.includes('increase') ? '#22c55e' : '#ef4444' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>{factor}</h4>
                <p style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#667eea' }}>{impact}</p>
              </Card>
            ))}
          </Grid>
        </ChartContainer>
      )}
    </Container>
  );
};

export default DemandDrivers;