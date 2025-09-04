import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const SlideContent = styled.div`
  max-width: 1200px;
  width: 100%;
  padding: 2rem;
`;

const Title = styled.h1`
  font-size: 3rem;
  font-weight: bold;
  margin-bottom: 3rem;
  text-align: center;
  color: #333;
`;

const RecommendationsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
  gap: 3rem;
`;

const Section = styled(motion.div)`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const SectionTitle = styled.h2`
  font-size: 1.8rem;
  font-weight: bold;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const Icon = styled.span`
  font-size: 1.5rem;
`;

const RecommendationList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const RecommendationItem = styled(motion.li)`
  display: flex;
  align-items: flex-start;
  margin-bottom: 1.2rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 3px solid #667eea;
`;

const Bullet = styled.span`
  color: #667eea;
  font-weight: bold;
  margin-right: 1rem;
  font-size: 1.2rem;
`;

const Text = styled.span`
  flex: 1;
  color: #444;
  line-height: 1.6;
`;

const Recommendations: React.FC = () => {
  const immediateActions = [
    "Implement dynamic staffing schedules based on peak hours (18:00-20:00)",
    "Adjust weekend staffing for seafood restaurants (+52.2% demand)",
    "Create temperature-responsive promotions for mild-warm weather days",
    "Optimize capacity during peak utilization periods (53-78%)"
  ];

  const strategicInitiatives = [
    "Develop predictive models for fine dining volatility management",
    "Create event partnership programs for 25% demand boost opportunities",
    "Implement dynamic pricing strategies for peak revenue periods",
    "Build real-time dashboard for monitoring demand patterns"
  ];

  return (
    <SlideContent>
      <Title>Strategic Recommendations</Title>
      <RecommendationsContainer>
        <Section
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle>
            <Icon>🚀</Icon>
            Immediate Actions (0-30 days)
          </SectionTitle>
          <RecommendationList>
            {immediateActions.map((action, index) => (
              <RecommendationItem
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <Bullet>→</Bullet>
                <Text>{action}</Text>
              </RecommendationItem>
            ))}
          </RecommendationList>
        </Section>

        <Section
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle>
            <Icon>🎯</Icon>
            Strategic Initiatives (30-90 days)
          </SectionTitle>
          <RecommendationList>
            {strategicInitiatives.map((initiative, index) => (
              <RecommendationItem
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <Bullet>→</Bullet>
                <Text>{initiative}</Text>
              </RecommendationItem>
            ))}
          </RecommendationList>
        </Section>
      </RecommendationsContainer>
    </SlideContent>
  );
};

export default Recommendations;