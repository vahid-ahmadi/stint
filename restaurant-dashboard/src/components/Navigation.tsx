import React from 'react';
import styled from 'styled-components';

interface Slide {
  path: string;
  name: string;
}

interface NavigationProps {
  slides: Slide[];
  currentSlide: number;
  onNavigate: (index: number) => void;
}

const NavContainer = styled.nav`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  background: linear-gradient(135deg, #6366F1 0%, #10B981 100%);
  border-bottom: 2px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const NavList = styled.ul`
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
  gap: 30px;
`;

const NavItem = styled.li<{ isActive: boolean }>`
  position: relative;
  cursor: pointer;
  color: ${props => props.isActive ? '#ffffff' : 'rgba(255, 255, 255, 0.7)'};
  font-weight: ${props => props.isActive ? 'bold' : 'normal'};
  font-size: 16px;
  transition: all 0.3s ease;
  
  &:hover {
    color: #ffffff;
    transform: translateY(-2px);
  }
  
  ${props => props.isActive && `
    &::after {
      content: '';
      position: absolute;
      bottom: -10px;
      left: 0;
      width: 100%;
      height: 3px;
      background: #ffffff;
      border-radius: 2px;
    }
  `}
`;

const Navigation: React.FC<NavigationProps> = ({ slides, currentSlide, onNavigate }) => {
  return (
    <NavContainer>
      <NavList>
        {slides.map((slide, index) => (
          <NavItem 
            key={slide.path}
            isActive={index === currentSlide}
            onClick={() => onNavigate(index)}
          >
            {slide.name}
          </NavItem>
        ))}
      </NavList>
    </NavContainer>
  );
};

export default Navigation;