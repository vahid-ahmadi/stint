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
  background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
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
    color: #FF6B6B;
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

const DifficultyGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const DifficultyCard = styled.div<{ difficulty?: number }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#FF4444' :
    props.difficulty && props.difficulty > 0.6 ? '#FF8E53' : 
    '#FFBB28'};
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
  font-size: 1.8rem;
  font-weight: bold;
  color: ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#FF4444' :
    props.difficulty && props.difficulty > 0.6 ? '#FF8E53' : 
    '#FFBB28'};
`;

const DifficultyLevel = styled.div<{ difficulty?: number }>`
  font-size: 0.9rem;
  font-weight: bold;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  text-align: center;
  margin: 0.5rem 0;
  background: ${props => 
    props.difficulty && props.difficulty > 0.7 ? '#FF4444' :
    props.difficulty && props.difficulty > 0.6 ? '#FF8E53' : 
    '#FFBB28'};
  color: white;
`;

const FactorList = styled.div`
  margin-top: 1rem;
`;

const FactorItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f0f0f0;
  font-size: 0.9rem;
  
  &:last-child {
    border-bottom: none;
  }
`;

const FactorLabel = styled.span`
  color: #666;
`;

const FactorValue = styled.span`
  font-weight: bold;
  color: #333;
`;

const ExplanationText = styled.p`
  color: #666;
  font-size: 0.9rem;
  line-height: 1.4;
  margin-top: 1rem;
  font-style: italic;
`;

const ConditionsSection = styled.div`
  background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
`;

const ConditionGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 1.5rem;
`;

const ConditionCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  padding: 1.5rem;
  border-radius: 8px;
  text-align: center;
`;

const ConditionTitle = styled.h4`
  color: #333;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
`;

const ConditionDifficulty = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: #e17055;
  margin: 0.5rem 0;
`;

const ConditionOccurrence = styled.div`
  font-size: 0.9rem;
  color: #636e72;
  margin-bottom: 1rem;
`;

const ConditionExplanation = styled.div`
  font-size: 0.85rem;
  color: #2d3436;
  font-style: italic;
`;

interface HardestPredictData {
  analysis_timestamp: string;
  difficulty_metrics: Record<string, {
    overall_cv: number;
    autocorr_lag1: number;
    external_sensitivity_avg: number;
    composite_difficulty_score: number;
    difficulty_components: Record<string, number>;
  }>;
  hardest_conditions: Record<string, {
    restaurants: Record<string, {
      volatility_ratio: number;
      demand_deviation: number;
      condition_difficulty: number;
      sample_size: number;
    }>;
    avg_difficulty: number;
    max_difficulty: number;
    total_occurrences: number;
    percentage_of_data: number;
  }>;
  explanations: Record<string, {
    difficulty_score: number;
    difficulty_level: string;
    main_factors: string[];
    explanation_text: string;
    metrics: {
      overall_cv: number;
      autocorr_lag1: number;
      external_sensitivity_avg: number;
      composite_difficulty_score: number;
    };
  }>;
  summary: {
    hardest_restaurant: string;
    hardest_score: number;
    easiest_restaurant: string;
    easiest_score: number;
    top_difficult_conditions: string[];
  };
}


const PredictionDifficulty: React.FC = () => {
  const [data, setData] = useState<HardestPredictData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Updated data structure to match JSON

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/hardest_predict_periods_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded hardest predict periods data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading prediction difficulty analysis...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  return (
    <Container>
      <Title>Prediction Difficulty Analysis</Title>
      
      <Question>
        Q: Which periods and restaurant types are hardest to predict, and why?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Volatility Scoring: Used coefficient of variation to measure demand unpredictability</MethodItem>
          <MethodItem>Predictability Testing: Analyzed how well past patterns predict future demand (autocorrelation)</MethodItem>
          <MethodItem>External Factor Sensitivity: Measured how external conditions affect predictability</MethodItem>
          <MethodItem>Composite Difficulty Score: Combined multiple metrics to rank restaurant types by forecast difficulty</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/hardest_predict_periods_analysis.png`}
        alt="Hardest to Predict Periods Analysis - Difficulty by Restaurant Type and Conditions"
      />

      <SummarySection>
        <SummaryTitle>🎯 Key Findings</SummaryTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#fff5f5', borderRadius: '8px' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>🔥</div>
            <h4 style={{ color: '#FF4444', marginBottom: '0.5rem' }}>Hardest to Predict</h4>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#333', textTransform: 'capitalize' }}>{data.summary.hardest_restaurant}</div>
            <div style={{ color: '#666', fontSize: '0.9rem' }}>Score: {data.summary.hardest_score.toFixed(3)}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#f0fff4', borderRadius: '8px' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>✅</div>
            <h4 style={{ color: '#22c55e', marginBottom: '0.5rem' }}>Easiest to Predict</h4>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#333', textTransform: 'capitalize' }}>{data.summary.easiest_restaurant}</div>
            <div style={{ color: '#666', fontSize: '0.9rem' }}>Score: {data.summary.easiest_score.toFixed(3)}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#fffbeb', borderRadius: '8px' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>⚠️</div>
            <h4 style={{ color: '#f59e0b', marginBottom: '0.5rem' }}>Top Difficult Conditions</h4>
            <div style={{ fontSize: '1rem', fontWeight: 'bold', color: '#333' }}>{data.summary.top_difficult_conditions.join(', ')}</div>
          </div>
        </div>
      </SummarySection>

      <SummaryTitle>📊 Restaurant-Specific Difficulty Ranking</SummaryTitle>
      <DifficultyGrid>
        {Object.entries(data.explanations).map(([restaurantType, restaurant]) => (
          <DifficultyCard key={restaurantType} difficulty={restaurant.difficulty_score}>
            <RestaurantName>
              {restaurantType}
              <DifficultyScore difficulty={restaurant.difficulty_score}>
                {restaurant.difficulty_score.toFixed(3)}
              </DifficultyScore>
            </RestaurantName>
            <DifficultyLevel difficulty={restaurant.difficulty_score}>
              {restaurant.difficulty_level}
            </DifficultyLevel>
            
            <FactorList>
              <FactorItem>
                <FactorLabel>📊 Demand Volatility (CV)</FactorLabel>
                <FactorValue>{restaurant.metrics.overall_cv.toFixed(3)}</FactorValue>
              </FactorItem>
              <FactorItem>
                <FactorLabel>🔮 Predictability (Autocorr)</FactorLabel>
                <FactorValue>{restaurant.metrics.autocorr_lag1.toFixed(3)}</FactorValue>
              </FactorItem>
              <FactorItem>
                <FactorLabel>🌍 External Sensitivity</FactorLabel>
                <FactorValue>{restaurant.metrics.external_sensitivity_avg.toFixed(3)}</FactorValue>
              </FactorItem>
            </FactorList>
            
            <ExplanationText>{restaurant.explanation_text}</ExplanationText>
          </DifficultyCard>
        ))}
      </DifficultyGrid>

      <ConditionsSection>
        <SummaryTitle style={{ color: '#2d3436', margin: 0 }}>⚠️ Most Difficult Conditions to Predict</SummaryTitle>
        <ConditionGrid>
          {Object.entries(data.hardest_conditions).slice(0, 6).map(([conditionName, condition]) => (
            <ConditionCard key={conditionName}>
              <ConditionTitle>{conditionName.replace('_', ' ').toUpperCase()}</ConditionTitle>
              <ConditionDifficulty>{condition.avg_difficulty.toFixed(3)}</ConditionDifficulty>
              <ConditionOccurrence>{condition.percentage_of_data.toFixed(1)}% of periods ({condition.total_occurrences.toLocaleString()} instances)</ConditionOccurrence>
              <ConditionExplanation>Most challenging for {Object.entries(condition.restaurants).reduce((max, [name, data]) => data.condition_difficulty > condition.restaurants[max]?.condition_difficulty ? name : max, Object.keys(condition.restaurants)[0])}</ConditionExplanation>
            </ConditionCard>
          ))}
        </ConditionGrid>
      </ConditionsSection>

      <SummarySection>
        <SummaryTitle>💡 Strategic Implications</SummaryTitle>
        <div style={{ display: 'grid', gap: '1rem' }}>
          <div style={{ 
            padding: '1rem', 
            background: '#fef2f2', 
            borderRadius: '8px',
            borderLeft: '4px solid #ef4444'
          }}>
            <strong>High-Difficulty Restaurants:</strong> {Object.entries(data.explanations).filter(([name, restaurant]) => restaurant.difficulty_score > 0.6).map(([name]) => name).join(', ')} require advanced forecasting models with real-time updates and flexible staffing approaches.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#fffbeb', 
            borderRadius: '8px',
            borderLeft: '4px solid #f59e0b'
          }}>
            <strong>Challenging Conditions:</strong> Late night, weekend rush, and economic volatility periods need specialized prediction approaches with higher uncertainty buffers.
          </div>
          <div style={{ 
            padding: '1rem', 
            background: '#f0fff4', 
            borderRadius: '8px',
            borderLeft: '4px solid #22c55e'
          }}>
            <strong>Investment Priority:</strong> Focus data science resources on the most difficult restaurant types and implement automated alerts during high-volatility periods.
          </div>
        </div>
      </SummarySection>
    </Container>
  );
};

export default PredictionDifficulty;