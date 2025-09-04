import React, { useState } from 'react';
import styled from 'styled-components';
import DemandDrivers from './Part1/DemandDrivers';
import ExternalFactors from './Part1/ExternalFactors';
import ActionableInsights from './Part1/ActionableInsights';
import PeakPeriods from './Part1/PeakPeriods';
import PredictionDifficulty from './Part1/PredictionDifficulty';
import ForecastVariation from './Part1/ForecastVariation';
import BusinessImpact from './Part1/BusinessImpact';

const Container = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const SubNavigation = styled.div`
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 0.5rem;
  padding: 1rem;
  background: white;
  border-bottom: 2px solid #f0f0f0;
  
  @media (max-width: 1200px) {
    grid-template-columns: repeat(4, 1fr);
  }
  
  @media (max-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
  
  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

const SubTab = styled.button<{ isActive: boolean }>`
  padding: 1rem 1.5rem;
  border: none;
  background: ${props => props.isActive ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f8f9fa'};
  color: ${props => props.isActive ? 'white' : '#666'};
  border-radius: 12px;
  font-size: 1rem;
  font-weight: ${props => props.isActive ? 'bold' : '600'};
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: ${props => props.isActive ? '0 4px 12px rgba(102, 126, 234, 0.4)' : '0 2px 4px rgba(0, 0, 0, 0.1)'};
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: ${props => props.isActive ? '0 6px 16px rgba(102, 126, 234, 0.5)' : '0 4px 12px rgba(0, 0, 0, 0.15)'};
    background: ${props => props.isActive ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#e9ecef'};
  }
  
  &:active {
    transform: translateY(-1px);
  }
`;

const ContentArea = styled.div`
  flex: 1;
  overflow-y: auto;
  background: #f8f9fa;
`;

const subTabs = [
  { id: 'drivers', label: 'Demand Drivers', component: DemandDrivers },
  { id: 'external', label: 'External Factors', component: ExternalFactors },
  { id: 'insights', label: 'Actionable Insights', component: ActionableInsights },
  { id: 'peak', label: 'Peak Periods', component: PeakPeriods },
  { id: 'difficulty', label: 'Prediction Difficulty', component: PredictionDifficulty },
  { id: 'variation', label: 'Forecast Variation', component: ForecastVariation },
  { id: 'impact', label: 'Business Impact', component: BusinessImpact }
];

const Part1: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState('drivers');
  
  const ActiveComponent = subTabs.find(tab => tab.id === activeSubTab)?.component || DemandDrivers;

  return (
    <Container>
      <SubNavigation>
        {subTabs.map(tab => (
          <SubTab
            key={tab.id}
            isActive={activeSubTab === tab.id}
            onClick={() => setActiveSubTab(tab.id)}
          >
            {tab.label}
          </SubTab>
        ))}
      </SubNavigation>
      <ContentArea>
        <ActiveComponent />
      </ContentArea>
    </Container>
  );
};

export default Part1;