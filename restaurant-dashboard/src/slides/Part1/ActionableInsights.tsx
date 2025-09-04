import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
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
  background: linear-gradient(135deg, #FFBB28 0%, #FF8042 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
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
    color: #FF8042;
  }
`;


const InsightGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
`;

const InsightCard = styled(motion.div)<{ priority?: string }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => 
    props.priority === 'high' ? '#FF4444' : 
    props.priority === 'medium' ? '#FFBB28' : '#44AA44'};
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
  }
`;

const PriorityBadge = styled.div<{ priority?: string }>`
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: bold;
  text-transform: uppercase;
  margin-bottom: 1rem;
  background: ${props => 
    props.priority === 'high' ? '#FF4444' : 
    props.priority === 'medium' ? '#FFBB28' : '#44AA44'};
  color: white;
`;

const InsightText = styled.h4`
  font-size: 1.1rem;
  color: #333;
  margin-bottom: 0.75rem;
  line-height: 1.4;
`;

const ActionText = styled.p`
  font-size: 0.95rem;
  color: #666;
  margin-bottom: 0.5rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 6px;
  border-left: 3px solid #FFBB28;
`;

const ImpactText = styled.div`
  font-size: 0.9rem;
  color: #888;
  font-weight: 500;
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

const RecommendationList = styled.ul`
  list-style: none;
  padding: 0;
`;

const RecommendationItem = styled.li`
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1rem;
  margin-bottom: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #FFBB28;
`;

const RecommendationNumber = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #FFBB28 0%, #FF8042 100%);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  flex-shrink: 0;
  font-size: 0.9rem;
`;

const CategorySection = styled.div`
  margin-bottom: 2rem;
`;

const CategoryTitle = styled.h4`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.2rem;
  text-transform: capitalize;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

interface ActionableInsightsData {
  generated_at: string;
  total_insights: number;
  high_priority_insights: number;
  insights: Array<{
    category: string;
    restaurant_type: string;
    insight: string;
    action: string;
    impact: string;
    priority: string;
  }>;
  top_actionable_recommendations: string[];
}

const getCategoryIcon = (category: string): string => {
  const icons: Record<string, string> = {
    'weekend_patterns': '📅',
    'temperature_impact': '🌡️',
    'events': '🎉',
    'peak_hours': '⏰',
    'reputation': '⭐',
    'volatility': '📊'
  };
  return icons[category] || '💡';
};

const ActionableInsights: React.FC = () => {
  const [data, setData] = useState<ActionableInsightsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/actionable_insights_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded actionable insights data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading actionable insights...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  // Group insights by category
  const groupedInsights = data.insights.reduce((acc, insight) => {
    if (!acc[insight.category]) {
      acc[insight.category] = [];
    }
    acc[insight.category].push(insight);
    return acc;
  }, {} as Record<string, typeof data.insights>);

  // Get high priority insights for featured section
  const highPriorityInsights = data.insights.filter(insight => insight.priority === 'high').slice(0, 6);

  return (
    <Container>
      <Title>Actionable Business Insights</Title>
      
      <Question>
        Q: What specific actions should restaurants take based on demand patterns?
      </Question>

      <MethodologySection>
        <MethodTitle>📊 Analysis Methodology</MethodTitle>
        <MethodList>
          <MethodItem>Pattern Analysis: Identified significant demand drivers from all previous analyses</MethodItem>
          <MethodItem>Impact Quantification: Measured business impact and opportunity size for each pattern</MethodItem>
          <MethodItem>Actionability Assessment: Prioritized insights based on implementation feasibility and impact</MethodItem>
          <MethodItem>Recommendation Generation: Converted findings into specific, actionable business steps</MethodItem>
        </MethodList>
      </MethodologySection>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/actionable_insights_visualization.png`}
        alt="Actionable Insights Visualizations - Weekend Effects, Weather Impact, Peak Hours, and Volatility Analysis"
      />

      <SummarySection>
        <SummaryTitle>
          🎯 Top Actionable Recommendations
        </SummaryTitle>
        <RecommendationList>
          {data.top_actionable_recommendations.map((recommendation, index) => (
            <RecommendationItem key={index}>
              <RecommendationNumber>{index + 1}</RecommendationNumber>
              <div>{recommendation}</div>
            </RecommendationItem>
          ))}
        </RecommendationList>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>
          🚨 High Priority Actions ({data.high_priority_insights} of {data.total_insights} insights)
        </SummaryTitle>
        <InsightGrid>
          {highPriorityInsights.map((insight, index) => (
            <InsightCard
              key={index}
              priority={insight.priority}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <PriorityBadge priority={insight.priority}>{insight.priority} Priority</PriorityBadge>
              <InsightText>{insight.insight}</InsightText>
              <ActionText>➤ {insight.action}</ActionText>
              <ImpactText>Impact: {insight.impact}</ImpactText>
            </InsightCard>
          ))}
        </InsightGrid>
      </SummarySection>

      {Object.entries(groupedInsights).map(([category, categoryInsights]) => (
        <CategorySection key={category}>
          <CategoryTitle>
            {getCategoryIcon(category)} {category.replace('_', ' ').toUpperCase()}
          </CategoryTitle>
          <InsightGrid>
            {categoryInsights.map((insight, index) => (
              <InsightCard
                key={`${category}-${index}`}
                priority={insight.priority}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
                whileHover={{ scale: 1.02 }}
              >
                <PriorityBadge priority={insight.priority}>{insight.priority}</PriorityBadge>
                <InsightText>{insight.insight}</InsightText>
                <ActionText>➤ {insight.action}</ActionText>
                <ImpactText>Impact: {insight.impact} | Restaurant: {insight.restaurant_type}</ImpactText>
              </InsightCard>
            ))}
          </InsightGrid>
        </CategorySection>
      ))}

    </Container>
  );
};

export default ActionableInsights;