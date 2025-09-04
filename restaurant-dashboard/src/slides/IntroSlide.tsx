import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const SlideContent = styled.div`
  max-width: 1200px;
  width: 100%;
  text-align: center;
  color: #333;
`;

const Title = styled(motion.h1)`
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 2rem;
  background: linear-gradient(135deg, #6366F1 0%, #10B981 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const Subtitle = styled(motion.h2)`
  font-size: 2rem;
  margin-bottom: 3rem;
  color: #666;
`;

const StatsGrid = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 3rem;
`;

const StatCard = styled(motion.div)`
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
  }
`;

const StatValue = styled.h3`
  font-size: 2.5rem;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.p`
  font-size: 1rem;
  color: #666;
`;

const IntroSlide: React.FC = () => {
  return (
    <SlideContent>
      <Title
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        Restaurant Demand Forecasting Dashboard
      </Title>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
        style={{
          maxWidth: '900px',
          margin: '3rem auto',
          padding: '3rem',
          background: 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%)',
          borderRadius: '20px',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.1)',
          textAlign: 'center',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          position: 'relative'
        }}
      >
        <h3 style={{
          position: 'absolute',
          top: '1.5rem',
          left: '2rem',
          color: '#6366F1',
          fontSize: '1.3rem',
          fontWeight: '600',
          margin: '0'
        }}>
          About Me
        </h3>
        
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          marginBottom: '2rem',
          gap: '1rem'
        }}>
          <img 
            src={`${process.env.PUBLIC_URL}/vahid_ahmadi.jpg`}
            alt="Vahid Ahmadi"
            style={{ 
              width: '180px', 
              height: '180px', 
              borderRadius: '50%',
              objectFit: 'cover',
              border: '4px solid #6366F1',
              boxShadow: '0 12px 24px rgba(0, 0, 0, 0.2)'
            }}
          />
          <div style={{ textAlign: 'left' }}>
            <h3 style={{ margin: '0', fontSize: '1.5rem', color: '#333', fontWeight: '600' }}>Vahid Ahmadi</h3>
            <p style={{ margin: '0', fontSize: '1rem', color: '#6366F1', fontWeight: '500' }}>Data Science & ML Engineering</p>
          </div>
        </div>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
          gap: '1.2rem', 
          textAlign: 'center',
          marginBottom: '3rem'
        }}>
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(99, 102, 241, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #6366F1'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6366F1', fontSize: '0.9rem', fontWeight: '600' }}>
              🏢 Current Role
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>PolicyEngine</strong><br/>Research Associate
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(139, 92, 246, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #8B5CF6'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#8B5CF6', fontSize: '0.9rem', fontWeight: '600' }}>
              🎓 Education
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>LSE</strong> Pre-Doctoral<br/><strong>Munich</strong> M.Sc. Economics
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(16, 185, 129, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #10B981'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#10B981', fontSize: '0.9rem', fontWeight: '600' }}>
              🤖 ML Production
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>3M+ Records</strong><br/>Credit Scoring Systems
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(52, 211, 153, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #34D399'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#34D399', fontSize: '0.9rem', fontWeight: '600' }}>
              ☁️ Cloud & MLOps
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>AWS SageMaker</strong><br/>Real-time Inference
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(107, 114, 128, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #6B7280'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#6B7280', fontSize: '0.9rem', fontWeight: '600' }}>
              📈 Forecasting
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>Time Series Analysis</strong><br/>Economic Forecasting
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(31, 41, 55, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #1F2937'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#1F2937', fontSize: '0.9rem', fontWeight: '600' }}>
              🔧 Algorithms
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>Random Forest</strong><br/>Deep Learning, PCA
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(16, 185, 129, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #10B981'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#10B981', fontSize: '0.9rem', fontWeight: '600' }}>
              🛠️ Tools
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>Python, SQL</strong><br/>Git, CI/CD
            </p>
          </div>
          
          <div style={{ 
            padding: '1rem', 
            background: 'rgba(139, 92, 246, 0.08)', 
            borderRadius: '12px',
            borderTop: '3px solid #8B5CF6'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: '#8B5CF6', fontSize: '0.9rem', fontWeight: '600' }}>
              ⚡ Frameworks
            </h4>
            <p style={{ margin: '0', lineHeight: '1.3', color: '#555', fontSize: '0.8rem' }}>
              <strong>TensorFlow, PyTorch</strong><br/>Scikit-learn
            </p>
          </div>
        </div>
        
        <h3 style={{
          color: '#6366F1',
          fontSize: '1.3rem',
          fontWeight: '600',
          margin: '0 0 1.5rem 0',
          textAlign: 'left'
        }}>
          Connect With Me
        </h3>
        
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '1.5rem',
          alignItems: 'center'
        }}>
          <a 
            href="mailto:va.vahidahmadi@gmail.com"
            style={{ textDecoration: 'none' }}
            title="Email"
          >
            <div style={{
              padding: '0.8rem 1.2rem',
              backgroundColor: '#6366F1',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '0.9rem',
              fontWeight: '600',
              transition: 'transform 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              Email
            </div>
          </a>
          
          <a 
            href="https://linkedin.com/in/vahid-ahmadi"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: 'none' }}
            title="LinkedIn"
          >
            <div style={{
              padding: '0.8rem 1.2rem',
              backgroundColor: '#10B981',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '0.9rem',
              fontWeight: '600',
              transition: 'transform 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              LinkedIn
            </div>
          </a>
          
          <a 
            href="https://github.com/vahidahmadi"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: 'none' }}
            title="GitHub"
          >
            <div style={{
              padding: '0.8rem 1.2rem',
              backgroundColor: '#1F2937',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '0.9rem',
              fontWeight: '600',
              transition: 'transform 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              GitHub
            </div>
          </a>
          
          <a 
            href="https://twitter.com/vahidahmadi"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: 'none' }}
            title="Twitter"
          >
            <div style={{
              padding: '0.8rem 1.2rem',
              backgroundColor: '#6B7280',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '0.9rem',
              fontWeight: '600',
              transition: 'transform 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              X
            </div>
          </a>
        </div>
      </motion.div>
    </SlideContent>
  );
};

export default IntroSlide;