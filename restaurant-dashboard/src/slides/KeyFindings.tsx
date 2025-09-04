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

const FindingsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
`;

const FindingCard = styled(motion.div)`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #667eea;
`;

const FindingIcon = styled.div`
  font-size: 2rem;
  margin-bottom: 1rem;
`;

const FindingTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: bold;
  color: #333;
  margin-bottom: 1rem;
`;

const FindingText = styled.p`
  font-size: 1rem;
  color: #666;
  line-height: 1.6;
`;

const Highlight = styled.span`
  color: #667eea;
  font-weight: bold;
`;

const KeyFindings: React.FC = () => {
  const findings = [
    {
      icon: "📈",
      title: "Peak Hour Revenue",
      text: "Peak hours (18:00-20:00) generate 42.4% higher revenue compared to average periods",
      highlight: "42.4% higher"
    },
    {
      icon: "🌊",
      title: "Weekend Surge",
      text: "Seafood restaurants show the highest weekend lift at +52.2% compared to weekdays",
      highlight: "+52.2%"
    },
    {
      icon: "🌡️",
      title: "Weather Impact",
      text: "Mild-warm weather (21-25°C) increases demand by 9.4% across all restaurant types",
      highlight: "9.4% increase"
    },
    {
      icon: "🎯",
      title: "Capacity Utilization",
      text: "Average capacity utilization ranges from 53-78% during peak hours",
      highlight: "53-78%"
    },
    {
      icon: "⚡",
      title: "Volatility Patterns",
      text: "Fine dining shows highest demand volatility (CV: 54.2%), requiring adaptive staffing",
      highlight: "54.2% volatility"
    },
    {
      icon: "🏆",
      title: "Event Boost",
      text: "Local events can increase demand by up to 25% for nearby restaurants",
      highlight: "25% boost"
    }
  ];

  return (
    <SlideContent>
      <Title>Key Business Findings</Title>
      <FindingsGrid>
        {findings.map((finding, index) => (
          <FindingCard
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ scale: 1.03 }}
          >
            <FindingIcon>{finding.icon}</FindingIcon>
            <FindingTitle>{finding.title}</FindingTitle>
            <FindingText>
              {finding.text.replace(finding.highlight, '')}
              <Highlight>{finding.highlight}</Highlight>
              {finding.text.substring(finding.text.indexOf(finding.highlight) + finding.highlight.length)}
            </FindingText>
          </FindingCard>
        ))}
      </FindingsGrid>
    </SlideContent>
  );
};

export default KeyFindings;