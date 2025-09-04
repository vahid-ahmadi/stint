import React, { useState } from 'react';
import styled from 'styled-components';
import ImageModal from './ImageModal';

const ButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  margin: 2rem auto;
  max-width: 1200px;
`;

const AnalysisButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  display: flex;
  align-items: center;
  gap: 0.75rem;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const ButtonText = styled.span`
  font-weight: 600;
`;

interface ClickableImageProps {
  src: string;
  alt: string;
  buttonText?: string;
}

const ClickableImage: React.FC<ClickableImageProps> = ({ src, alt, buttonText = "📊 View Analysis Results" }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleButtonClick = () => {
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <ButtonContainer>
        <AnalysisButton onClick={handleButtonClick}>
          <ButtonText>{buttonText}</ButtonText>
        </AnalysisButton>
      </ButtonContainer>
      <ImageModal
        src={src}
        alt={alt}
        isOpen={isModalOpen}
        onClose={handleModalClose}
      />
    </>
  );
};

export default ClickableImage;