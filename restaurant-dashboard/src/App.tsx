import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { AnimatePresence } from 'framer-motion';
import useKeyPress from './hooks/useKeyPress';
import Navigation from './components/Navigation';
import SlideContainer from './components/SlideContainer';
import IntroSlide from './slides/IntroSlide';
import Part1 from './slides/Part1';
import Part2 from './slides/Part2';
import Part3 from './slides/Part3';

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
`;

const slides = [
  { path: '/', component: IntroSlide, name: 'Introduction' },
  { path: '/part1', component: Part1, name: 'Part 1: EDA Analysis' },
  { path: '/part2', component: Part2, name: 'Part 2: ML Models' },
  { path: '/part3', component: Part3, name: 'Part 3: Model Evaluation' }
];

function PresentationApp() {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentSlide, setCurrentSlide] = useState(0);

  const handleKeyPress = useCallback((event: KeyboardEvent) => {
    if (event.key === 'ArrowRight' && currentSlide < slides.length - 1) {
      const nextSlide = currentSlide + 1;
      setCurrentSlide(nextSlide);
      navigate(slides[nextSlide].path);
    } else if (event.key === 'ArrowLeft' && currentSlide > 0) {
      const prevSlide = currentSlide - 1;
      setCurrentSlide(prevSlide);
      navigate(slides[prevSlide].path);
    }
  }, [currentSlide, navigate]);

  useKeyPress(['ArrowLeft', 'ArrowRight'], handleKeyPress);

  useEffect(() => {
    const currentPath = location.pathname;
    const slideIndex = slides.findIndex(slide => slide.path === currentPath);
    if (slideIndex !== -1) {
      setCurrentSlide(slideIndex);
    }
  }, [location]);

  return (
    <AppContainer>
      <Navigation 
        slides={slides} 
        currentSlide={currentSlide} 
        onNavigate={(index) => {
          setCurrentSlide(index);
          navigate(slides[index].path);
        }}
      />
      <AnimatePresence mode="wait">
        <Routes>
          {slides.map(slide => {
            const Component = slide.component;
            return (
              <Route 
                key={slide.path}
                path={slide.path} 
                element={
                  <SlideContainer>
                    {Component && <Component />}
                  </SlideContainer>
                } 
              />
            );
          })}
        </Routes>
      </AnimatePresence>
    </AppContainer>
  );
}

function App() {
  return (
    <Router basename={process.env.PUBLIC_URL}>
      <PresentationApp />
    </Router>
  );
}

export default App;