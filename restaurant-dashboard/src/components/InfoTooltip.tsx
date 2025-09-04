import React, { useState } from 'react';
import styled from 'styled-components';

const TooltipContainer = styled.div`
  position: relative;
  display: inline-block;
  margin-left: 0.5rem;
`;

const InfoIcon = styled.div`
  width: 18px;
  height: 18px;
  background: #667eea;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  cursor: help;
  transition: all 0.2s ease;
  
  &:hover {
    background: #5a67d8;
    transform: scale(1.1);
  }
`;

const TooltipText = styled.div<{ show: boolean }>`
  visibility: ${props => props.show ? 'visible' : 'hidden'};
  opacity: ${props => props.show ? '1' : '0'};
  background: rgba(0, 0, 0, 0.9);
  color: white;
  text-align: left;
  border-radius: 8px;
  padding: 0.75rem;
  position: absolute;
  z-index: 9999;
  bottom: 130%;
  right: 0;
  width: 320px;
  font-size: 0.85rem;
  line-height: 1.4;
  transition: opacity 0.3s;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  max-height: 200px;
  overflow-y: auto;
  
  &::after {
    content: "";
    position: absolute;
    top: 100%;
    right: 20px;
    border-width: 8px;
    border-style: solid;
    border-color: rgba(0, 0, 0, 0.9) transparent transparent transparent;
  }
`;

interface InfoTooltipProps {
  text: string;
}

const InfoTooltip: React.FC<InfoTooltipProps> = ({ text }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <TooltipContainer
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <InfoIcon>i</InfoIcon>
      <TooltipText show={showTooltip}>
        {text}
      </TooltipText>
    </TooltipContainer>
  );
};

export default InfoTooltip;