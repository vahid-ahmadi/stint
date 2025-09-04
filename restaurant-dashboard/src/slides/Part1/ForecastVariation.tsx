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

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
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

const RestaurantCard = styled.div<{ difficulty?: number }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#ff4444' :
    props.difficulty && props.difficulty > 0.6 ? '#ff8844' : 
    '#44aa44'};
`;

const RestaurantName = styled.h4`
  font-size: 1.2rem;
  color: #333;
  margin-bottom: 1rem;
  text-transform: capitalize;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const DifficultyScore = styled.span<{ difficulty?: number }>`
  font-size: 1.4rem;
  font-weight: bold;
  color: ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#ff4444' :
    props.difficulty && props.difficulty > 0.6 ? '#ff8844' : 
    '#44aa44'};
`;

const DifficultyLevel = styled.div<{ difficulty?: number }>`
  font-size: 0.9rem;
  font-weight: bold;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  text-align: center;
  margin: 0.5rem 0;
  background: ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#ff4444' :
    props.difficulty && props.difficulty > 0.6 ? '#ff8844' : 
    '#44aa44'};
  color: white;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin: 1rem 0;
`;

const MetricRow = styled.div`
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

const MetricLabel = styled.span`
  color: #666;
`;

const MetricValue = styled.span`
  font-weight: bold;
  color: #333;
`;

const ChallengesSection = styled.div`
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #eee;
`;

const ChallengeItem = styled.div<{ severity: string }>`
  margin: 0.5rem 0;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  background: ${props => 
    props.severity === 'High' ? '#fff5f5' :
    props.severity === 'Medium' ? '#fff8dc' :
    '#f0fff4'};
  border-left: 3px solid ${props => 
    props.severity === 'High' ? '#ff4444' :
    props.severity === 'Medium' ? '#ffa500' :
    '#44aa44'};
`;

const ChallengeName = styled.div`
  font-weight: bold;
  color: #333;
  margin-bottom: 0.25rem;
`;

const ChallengeDescription = styled.div`
  color: #666;
  margin-bottom: 0.25rem;
`;

const ChallengeRecommendation = styled.div`
  color: #555;
  font-style: italic;
`;

const ComparisonTable = styled.div`
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const TableHeader = styled.div`
  background: #667eea;
  color: white;
  padding: 1rem;
  font-weight: bold;
  display: grid;
  grid-template-columns: 2fr repeat(6, 1fr);
  gap: 1rem;
  font-size: 0.9rem;
`;

const TableRow = styled.div<{ difficulty?: number }>`
  display: grid;
  grid-template-columns: 2fr repeat(6, 1fr);
  gap: 1rem;
  padding: 1rem;
  border-bottom: 1px solid #eee;
  background: ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#fff5f5' :
    props.difficulty && props.difficulty > 0.6 ? '#fff8f0' :
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

interface ForecastDifficultyData {
  analysis_timestamp: string;
  restaurant_analysis: Record<string, {
    mean_demand: number;
    std_demand: number;
    cv_demand: number;
    autocorr_predictability: number;
    avg_external_sensitivity: number;
    peak_regularity: number;
    composite_difficulty: number;
    naive_mape: number;
    ma7_mape: number;
    weekend_volatility_ratio: number;
  }>;
  comparison_summary: Array<{
    Restaurant_Type: string;
    Difficulty_Score: number;
    CV_Demand: number;
    Autocorr_Predictability: number;
    External_Sensitivity: number;
    Pattern_Consistency: number;
    Peak_Regularity: number;
    Weekend_Volatility_Ratio: number;
    Naive_MAPE: number;
    MA7_MAPE: number;
  }>;
  challenges: Record<string, Array<{
    type: string;
    severity: string;
    description: string;
    recommendation: string;
  }>>;
}

const getDifficultyLevel = (score: number): string => {
  if (score > 0.7) return 'VERY DIFFICULT';
  if (score > 0.6) return 'DIFFICULT';
  if (score > 0.5) return 'MODERATE';
  return 'MANAGEABLE';
};

const ForecastVariation: React.FC = () => {
  const [data, setData] = useState<ForecastDifficultyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/forecast_difficulty_restaurant_type_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded forecast difficulty data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading forecast variation analysis...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  // Sort restaurants by difficulty score
  const sortedRestaurants = data.comparison_summary.sort((a, b) => b.Difficulty_Score - a.Difficulty_Score);

  return (
    <Container>
      <Title>Forecast Variation Analysis</Title>
      
      <Question>
        Q: How do forecast accuracy and demand patterns vary between restaurant types?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Forecast Accuracy Testing: Measured baseline forecast errors using naive and moving average methods</MethodItem>
          <MethodItem>Pattern Consistency Analysis: Evaluated how consistent demand patterns are across different time periods</MethodItem>
          <MethodItem>Volatility Comparison: Compared demand variation levels between restaurant types</MethodItem>
          <MethodItem>Challenge Identification: Identified specific forecasting challenges for each restaurant type</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/forecast_difficulty_restaurant_type_analysis.png`}
        alt="Forecast Difficulty Analysis by Restaurant Type - Accuracy metrics, pattern consistency, and forecasting challenges"
      />

      <SummarySection>
        <SummaryTitle>💡 Strategic Insights</SummaryTitle>
        <div style={{ display: 'grid', gap: '1rem' }}>
          <div style={{ 
            padding: '1rem', 
            background: '#fff5f5', 
            borderRadius: '8px',
            borderLeft: '4px solid #ef4444'
          }}>
            <strong>Highest Difficulty:</strong> {sortedRestaurants[0]?.Restaurant_Type} shows the highest forecast difficulty (score: {sortedRestaurants[0]?.Difficulty_Score.toFixed(3)}) due to high volatility and low predictability patterns.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#fffbeb', 
            borderRadius: '8px',
            borderLeft: '4px solid #f59e0b'
          }}>
            <strong>Volatility Leaders:</strong> Restaurants with CV {'>'}50% require specialized forecasting approaches with wider prediction intervals and flexible capacity management.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#f0fff4', 
            borderRadius: '8px',
            borderLeft: '4px solid #22c55e'
          }}>
            <strong>Best Performers:</strong> {sortedRestaurants[sortedRestaurants.length - 1]?.Restaurant_Type} shows the most predictable patterns with lower forecast errors, suitable for standard forecasting methods.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#f0f9ff', 
            borderRadius: '8px',
            borderLeft: '4px solid #3b82f6'
          }}>
            <strong>Forecast Error Patterns:</strong> All restaurant types show high MAPE values ({'>'}18K%), indicating the need for sophisticated forecasting models beyond simple baseline methods.
          </div>
        </div>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>🎯 Forecast Difficulty Ranking</SummaryTitle>
        <RestaurantGrid>
          {Object.entries(data.restaurant_analysis).map(([restaurantType, analysis]) => (
            <RestaurantCard key={restaurantType} difficulty={analysis.composite_difficulty}>
              <RestaurantName>
                {restaurantType}
                <DifficultyScore difficulty={analysis.composite_difficulty}>
                  {analysis.composite_difficulty.toFixed(3)}
                </DifficultyScore>
              </RestaurantName>
              <DifficultyLevel difficulty={analysis.composite_difficulty}>
                {getDifficultyLevel(analysis.composite_difficulty)}
              </DifficultyLevel>
              
              <MetricsGrid>
                <MetricRow>
                  <MetricLabel>📊 Demand Volatility (CV)</MetricLabel>
                  <MetricValue>{(analysis.cv_demand * 100).toFixed(1)}%</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>🔮 Predictability</MetricLabel>
                  <MetricValue>{analysis.autocorr_predictability.toFixed(3)}</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>🌍 External Sensitivity</MetricLabel>
                  <MetricValue>{analysis.avg_external_sensitivity.toFixed(3)}</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>⚡ Peak Regularity</MetricLabel>
                  <MetricValue>{analysis.peak_regularity.toFixed(3)}</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>📈 Naive MAPE</MetricLabel>
                  <MetricValue>{(analysis.naive_mape / 1000).toFixed(1)}K%</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>📊 MA7 MAPE</MetricLabel>
                  <MetricValue>{(analysis.ma7_mape / 1000).toFixed(1)}K%</MetricValue>
                </MetricRow>
              </MetricsGrid>
              
              {data.challenges[restaurantType] && (
                <ChallengesSection>
                  <strong style={{ fontSize: '0.9rem', color: '#333' }}>Key Challenges:</strong>
                  {data.challenges[restaurantType].slice(0, 3).map((challenge, index) => (
                    <ChallengeItem key={index} severity={challenge.severity}>
                      <ChallengeName>{challenge.type} ({challenge.severity})</ChallengeName>
                      <ChallengeDescription>{challenge.description}</ChallengeDescription>
                      <ChallengeRecommendation>💡 {challenge.recommendation}</ChallengeRecommendation>
                    </ChallengeItem>
                  ))}
                </ChallengesSection>
              )}
            </RestaurantCard>
          ))}
        </RestaurantGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>📊 Comparative Analysis</SummaryTitle>
        <ComparisonTable>
          <TableHeader>
            <div>Restaurant Type</div>
            <div>Difficulty</div>
            <div>Volatility</div>
            <div>Predictability</div>
            <div>External</div>
            <div>Regularity</div>
            <div>Forecast Error</div>
          </TableHeader>
          {sortedRestaurants.map((restaurant) => (
            <TableRow key={restaurant.Restaurant_Type} difficulty={restaurant.Difficulty_Score}>
              <TableCell style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                {restaurant.Restaurant_Type}
              </TableCell>
              <TableCell>{restaurant.Difficulty_Score.toFixed(3)}</TableCell>
              <TableCell>{(restaurant.CV_Demand * 100).toFixed(1)}%</TableCell>
              <TableCell>{restaurant.Autocorr_Predictability.toFixed(3)}</TableCell>
              <TableCell>{restaurant.External_Sensitivity.toFixed(3)}</TableCell>
              <TableCell>{restaurant.Peak_Regularity.toFixed(3)}</TableCell>
              <TableCell>{(restaurant.Naive_MAPE / 1000).toFixed(1)}K%</TableCell>
            </TableRow>
          ))}
        </ComparisonTable>
      </SummarySection>

    </Container>
  );
};

export default ForecastVariation;