import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import ClickableImage from '../components/ClickableImage';
import InfoTooltip from '../components/InfoTooltip';

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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const TabContainer = styled.div`
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  border-bottom: 2px solid #f0f0f0;
  width: 100%;
`;

const Tab = styled.button<{ active: boolean }>`
  flex: 1;
  padding: 1.5rem 2rem;
  border: none;
  background: ${props => props.active ? '#667eea' : 'transparent'};
  color: ${props => props.active ? 'white' : '#666'};
  border-radius: 8px 8px 0 0;
  font-weight: ${props => props.active ? 'bold' : 'normal'};
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 1.1rem;
  text-align: center;
  
  &:hover {
    background: ${props => props.active ? '#667eea' : '#f8fafc'};
    color: ${props => props.active ? 'white' : '#333'};
  }
`;

const TabContent = styled.div`
  display: block;
`;

const MethodologySection = styled.div`
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
`;

const MethodTitle = styled.h3`
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
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

const ExecutiveSummaryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const SummaryCard = styled(motion.div)`
  text-align: center;
  padding: 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px;
  transition: transform 0.2s ease;
  
  &:hover {
    transform: translateY(-4px);
  }
`;

const CardIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
`;

const CardTitle = styled.h4`
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
`;

const CardValue = styled.div`
  font-size: 1.8rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
`;

const CardDescription = styled.div`
  font-size: 0.9rem;
  opacity: 0.9;
`;

const ModelGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const ModelCard = styled(motion.div)<{ isWinner?: boolean }>`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-left: 4px solid ${props => props.isWinner ? '#22c55e' : '#667eea'};
  position: relative;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  }
`;

const WinnerBadge = styled.div`
  position: absolute;
  top: -10px;
  right: 15px;
  background: #22c55e;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: bold;
`;

const ModelName = styled.h4`
  font-size: 1.3rem;
  color: #333;
  margin-bottom: 1rem;
`;

const MetricGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-bottom: 1rem;
`;

const MetricRow = styled.div`
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

const MetricLabel = styled.span`
  color: #666;
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const MetricValue = styled.span`
  font-weight: bold;
  color: #333;
`;

const ImprovementBadge = styled.div`
  background: #dcfce7;
  color: #166534;
  padding: 0.5rem;
  border-radius: 8px;
  margin-top: 1rem;
  text-align: center;
  font-size: 0.9rem;
  font-weight: 600;
`;

const FeatureImportanceSection = styled.div`
  background: #f8fafc;
  padding: 1.5rem;
  border-radius: 12px;
  margin-top: 1rem;
`;

const FeatureBar = styled.div<{ importance: number }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0.5rem 0;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: ${props => props.importance * 100}%;
    background: linear-gradient(90deg, #667eea33, #764ba233);
    z-index: 1;
  }
`;

const FeatureName = styled.span`
  position: relative;
  z-index: 2;
  font-weight: 500;
`;

const FeatureValue = styled.span`
  position: relative;
  z-index: 2;
  font-weight: bold;
  color: #667eea;
`;

const RestaurantPerformanceGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
`;

const RestaurantCard = styled.div`
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid #667eea;
`;

const RestaurantName = styled.h5`
  margin-bottom: 0.5rem;
  text-transform: capitalize;
  color: #333;
`;

const RecommendationsList = styled.ul`
  list-style: none;
  padding: 0;
`;

const RecommendationItem = styled.li`
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: #f8fafc;
  border-radius: 8px;
  border-left: 3px solid #22c55e;
`;

const RecommendationNumber = styled.div`
  width: 24px;
  height: 24px;
  background: #22c55e;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 0.8rem;
  flex-shrink: 0;
`;

interface Part2Data {
  analysis_timestamp: string;
  model_configuration: {
    best_model: string;
    asymmetric_loss_ratio: number;
    features_used: number;
    training_samples: number;
    test_samples: number;
  };
  model_performance: Record<string, {
    mae: number;
    rmse: number;
    mape: number;
    asymmetric_loss: number;
    understaffing_rate_pct: number;
    severe_understaffing_rate_pct: number;
    avg_understaffing_magnitude: number;
    avg_overstaffing_magnitude: number;
    total_predictions: number;
  }>;
  forecasting_horizons: {
    "24_hours": string;
    "7_days": string;
    prediction_intervals: string[];
  };
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  business_impact: {
    understaffing_reduction_pct: number;
    recommended_penalty_ratio: string;
    implementation_readiness: string;
    update_frequency: string;
  };
  forecasts_sample: Record<string, {
    "24h_sample": number[];
    "7d_sample": number[];
  }>;
}

const Part2: React.FC = () => {
  const [activeTab, setActiveTab] = useState('A');
  const [data, setData] = useState<Part2Data | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL}/part2_option_a_forecasting_results.json`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log('Loaded Part 2 forecasting data:', data);
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading Part 2 data:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: '2rem' }}>Loading forecasting model results...</div>;
  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error loading data: {error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>No data available</div>;

  const sortedModels = Object.entries(data.model_performance)
    .sort(([,a], [,b]) => a.asymmetric_loss - b.asymmetric_loss);

  const sortedFeatures = data.feature_importance
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 6);

  const renderOptionA = () => (
    <TabContent>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🎯 Smart Loss Function
            <InfoTooltip text="Mathematical function: L(y,ŷ) = (y-ŷ)² × 3 if y>ŷ (understaffing), else (y-ŷ)². During high demand (>80th percentile): L × 1.5. This creates asymmetric penalties where missing 1 customer costs 3x more than over-predicting 1 customer, with extra protection during peak periods to prevent service failures." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>Business Logic:</strong> Understaffing costs 3x more than overstaffing</p>
            <p><strong>Peak Protection:</strong> Extra penalties during busy periods</p>
            <p><strong>Result:</strong> Prevents service quality issues while controlling labor costs</p>
          </div>
        </MethodologySection>

        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🚀 AI Models
            <InfoTooltip text="LightGBM: Fast gradient boosting framework with categorical feature optimization and early stopping (50 rounds). XGBoost: Extreme gradient boosting with custom objective functions. Both use learning rate 0.05, max depth 8, L1/L2 regularization 0.1, and 80% feature/sample subsampling for robust predictions." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>LightGBM & XGBoost:</strong> Advanced tree-based algorithms</p>
            <p><strong>Custom Training:</strong> Specialized for restaurant demand patterns</p>
            <p><strong>Auto-Optimization:</strong> Self-tuning parameters with early stopping</p>
          </div>
        </MethodologySection>

        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            📊 Smart Features
            <InfoTooltip text="57-feature engineering pipeline: Temporal (cyclical sin/cos encoding for hour/day/month), Lag features (1,2,48,336 periods = 30min,1hr,1day,1week historical demand), Rolling statistics (12,48,168,336 period moving averages/std), Weather interactions (temp×precipitation), Event correlations (social×local events), Restaurant-specific indicators (capacity utilization, revenue per customer), and Business context (weekend/peak period flags)." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>57 Predictors:</strong> Time patterns, weather, events, competition</p>
            <p><strong>Historical Memory:</strong> Learns from past demand (up to 1 week)</p>
            <p><strong>Real-time Factors:</strong> Current capacity, trends, seasonality</p>
          </div>
        </MethodologySection>

        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🔍 Explainable AI
            <InfoTooltip text="SHAP (SHapley Additive exPlanations) TreeExplainer computes exact feature contributions using game theory. For each prediction, SHAP values sum to explain the difference from baseline. Top features: weekend patterns (99.6% importance), capacity_available (99.2%), week_of_year (97.3%), viral_trend (96.2%), demand_ma_48 (95.2%). Enables managers to understand why demand spikes/drops occur." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>SHAP Analysis:</strong> Shows why each prediction was made</p>
            <p><strong>Feature Ranking:</strong> Identifies most important demand drivers</p>
            <p><strong>Decision Support:</strong> Actionable insights for managers</p>
          </div>
        </MethodologySection>

        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            ⚡ Production System
            <InfoTooltip text="REST API architecture: Accepts restaurant_type, timestamp, weather, events as input. Returns prediction + 80%/95% confidence intervals via quantile regression on residuals. Deployed with Docker containers, Redis caching, and Prometheus monitoring. Weekly retraining pipeline uses last 6 months data with feature drift detection and model performance validation before deployment." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>Real-time API:</strong> 30-minute resolution predictions</p>
            <p><strong>Confidence Bands:</strong> 80% & 95% uncertainty estimates</p>
            <p><strong>Auto-Update:</strong> Weekly retraining keeps models fresh</p>
          </div>
        </MethodologySection>

        <MethodologySection>
          <MethodTitle style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            ✅ Validation
            <InfoTooltip text="Time-series cross-validation with 80% temporal training split (183,129 samples) and 20% future test data (45,782 samples). Validation metrics include asymmetric loss evaluation, understaffing incident tracking, capacity constraint verification, and statistical significance testing. Out-of-sample performance ensures model generalizes to unseen future data without overfitting." />
          </MethodTitle>
          <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#555' }}>
            <p><strong>Rigorous Testing:</strong> {data.model_configuration.test_samples.toLocaleString()} predictions validated</p>
            <p><strong>Time-based Split:</strong> Tests future performance accurately</p>
            <p><strong>Business Metrics:</strong> Tracks operational impact, not just accuracy</p>
          </div>
        </MethodologySection>
      </div>

      <ClickableImage 
        src={`${process.env.PUBLIC_URL}/part2_option_a_forecasting_analysis.png`}
        alt="Machine Learning Forecasting Results - LightGBM vs XGBoost model comparison with asymmetric loss analysis and forecasting horizons"
      />

      <SummarySection>
        <SummaryTitle style={{ display: 'flex', alignItems: 'center' }}>
          🎯 Strategic Model Performance Results
          <InfoTooltip text="Advanced machine learning models with custom asymmetric loss functions designed to minimize understaffing incidents while controlling overstaffing costs. Models are evaluated on business impact metrics rather than just statistical accuracy." />
        </SummaryTitle>
        <ExecutiveSummaryGrid>
          <SummaryCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <CardIcon>🏆</CardIcon>
            <CardTitle>Best Performing Model</CardTitle>
            <CardValue>{data.model_configuration.best_model}</CardValue>
            <CardDescription>Lowest asymmetric loss achieved</CardDescription>
          </SummaryCard>
          <SummaryCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <CardIcon>📉</CardIcon>
            <CardTitle>Understaffing Reduction</CardTitle>
            <CardValue>{data.business_impact.understaffing_reduction_pct.toFixed(1)}%</CardValue>
            <CardDescription>Fewer understaffing incidents</CardDescription>
          </SummaryCard>
          <SummaryCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <CardIcon>🎯</CardIcon>
            <CardTitle>Features Used</CardTitle>
            <CardValue>{data.model_configuration.features_used}</CardValue>
            <CardDescription>Comprehensive feature set</CardDescription>
          </SummaryCard>
          <SummaryCard
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <CardIcon>💰</CardIcon>
            <CardTitle>Penalty Ratio</CardTitle>
            <CardValue>{data.business_impact.recommended_penalty_ratio}</CardValue>
            <CardDescription>Understaffing vs overstaffing</CardDescription>
          </SummaryCard>
        </ExecutiveSummaryGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle>🤖 Model Performance Comparison</SummaryTitle>
        <ModelGrid>
          {sortedModels.map(([modelName, performance], index) => (
            <ModelCard
              key={modelName}
              isWinner={index === 0}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {index === 0 && <WinnerBadge>🏆 BEST MODEL</WinnerBadge>}
              <ModelName style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {modelName}
                <InfoTooltip text={`${modelName} is a gradient boosting framework optimized for restaurant demand forecasting with custom asymmetric loss functions. Features automatic hyperparameter tuning and handles temporal patterns effectively.`} />
              </ModelName>
              <MetricGrid>
                <MetricRow>
                  <MetricLabel>
                    📊 MAE (customers)
                    <InfoTooltip text={`Average prediction error of ±${performance.mae.toFixed(2)} customers. This means roughly ${(performance.mae * 60).toFixed(0)} minutes of staff time variance per 30-minute period.`} />
                  </MetricLabel>
                  <MetricValue>{performance.mae.toFixed(2)}</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>
                    ⚖️ Business Loss Score
                    <InfoTooltip text={`Custom business metric optimized for restaurants. Score of ${performance.asymmetric_loss.toFixed(1)} balances service quality vs cost control. Lower is better.`} />
                  </MetricLabel>
                  <MetricValue>{performance.asymmetric_loss.toFixed(1)}</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>
                    📉 Service Risk
                    <InfoTooltip text={`${performance.understaffing_rate_pct.toFixed(1)}% of periods may have slower service due to understaffing. Approximately ${Math.round(performance.understaffing_rate_pct * 7 * 24 / 100)} hours per week.`} />
                  </MetricLabel>
                  <MetricValue>{performance.understaffing_rate_pct.toFixed(1)}%</MetricValue>
                </MetricRow>
                <MetricRow>
                  <MetricLabel>
                    🚨 Critical Incidents
                    <InfoTooltip text={`${performance.severe_understaffing_rate_pct.toFixed(1)}% severe understaffing events (>20% underestimation). About ${Math.round(performance.severe_understaffing_rate_pct * 168 / 100)} hours per week of potential service disruption.`} />
                  </MetricLabel>
                  <MetricValue>{performance.severe_understaffing_rate_pct.toFixed(1)}%</MetricValue>
                </MetricRow>
              </MetricGrid>
              
              <ImprovementBadge>
                🎯 {performance.total_predictions.toLocaleString()} predictions analyzed
              </ImprovementBadge>
            </ModelCard>
          ))}
        </ModelGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle style={{ display: 'flex', alignItems: 'center' }}>
          🔍 Feature Importance Analysis
          <InfoTooltip text="SHAP (SHapley Additive exPlanations) values showing which features contribute most to accurate demand predictions. Higher values indicate more important features." />
        </SummaryTitle>
        <FeatureImportanceSection>
          {sortedFeatures.map((item) => (
            <FeatureBar key={item.feature} importance={item.importance}>
              <FeatureName>{item.feature.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</FeatureName>
              <FeatureValue>{(item.importance * 100).toFixed(1)}%</FeatureValue>
            </FeatureBar>
          ))}
        </FeatureImportanceSection>
      </SummarySection>

      <SummarySection>
        <SummaryTitle style={{ display: 'flex', alignItems: 'center' }}>
          📊 Forecast Horizon Performance
          <InfoTooltip text="Multi-timeframe prediction capabilities designed for different operational decisions. Short-term for tactical planning, medium-term for strategic resource allocation." />
        </SummaryTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
          <div style={{ background: '#f8fafc', padding: '1.5rem', borderRadius: '12px', borderLeft: '4px solid #3b82f6' }}>
            <h4 style={{ color: '#1e40af', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              ⚡ 24-Hour Forecasts
              <InfoTooltip text="High-resolution 30-minute interval predictions for tactical operations. Enables precise staff scheduling, inventory ordering, and table reservation management. Prediction intervals quantify forecast uncertainty for risk management." />
            </h4>
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Resolution:</span>
                <strong>{data.forecasting_horizons["24_hours"]}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Confidence Bands:</span>
                <strong>{data.forecasting_horizons.prediction_intervals.join(', ')} intervals</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Use Case:</span>
                <strong>Staff scheduling, inventory</strong>
              </div>
            </div>
          </div>
          
          <div style={{ background: '#f0fdf4', padding: '1.5rem', borderRadius: '12px', borderLeft: '4px solid #22c55e' }}>
            <h4 style={{ color: '#15803d', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              📅 7-Day Forecasts
              <InfoTooltip text="Strategic weekly demand projections for resource planning. Aggregated daily averages enable workforce scheduling, procurement decisions, and capacity planning. Model retraining ensures adaptation to changing demand patterns." />
            </h4>
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Aggregation:</span>
                <strong>{data.forecasting_horizons["7_days"]}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Model Updates:</span>
                <strong>{data.business_impact.update_frequency}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Use Case:</span>
                <strong>Workforce & procurement planning</strong>
              </div>
            </div>
          </div>
        </div>
      </SummarySection>

      <SummarySection>
        <SummaryTitle style={{ display: 'flex', alignItems: 'center' }}>
          🏪 Restaurant-Specific Demand Forecasts
          <InfoTooltip text="Tailored predictions for each restaurant type based on their unique demand patterns, customer behavior, and operational characteristics. Values represent predicted customer counts." />
        </SummaryTitle>
        <RestaurantPerformanceGrid>
          {Object.entries(data.forecasts_sample).map(([restaurant, forecasts]) => {
            const avg24h = forecasts["24h_sample"].reduce((sum, val) => sum + val, 0) / forecasts["24h_sample"].length;
            const avg7d = forecasts["7d_sample"].reduce((sum, val) => sum + val, 0) / forecasts["7d_sample"].length;
            const peak24h = Math.max(...forecasts["24h_sample"]);
            const peak7d = Math.max(...forecasts["7d_sample"]);
            
            return (
              <RestaurantCard key={restaurant}>
                <RestaurantName>{restaurant}</RestaurantName>
                <div style={{ fontSize: '0.9rem' }}>
                  <div style={{ marginBottom: '1rem', background: '#f8fafc', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span><strong>24h Avg Demand:</strong></span>
                      <span style={{ color: '#3b82f6', fontWeight: 'bold' }}>{avg24h.toFixed(1)} customers</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span><strong>24h Peak:</strong></span>
                      <span style={{ color: '#ef4444', fontWeight: 'bold' }}>{peak24h.toFixed(1)} customers</span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.5rem' }}>
                      <strong>Next 10 periods:</strong> {forecasts["24h_sample"].slice(0, 5).map(val => val.toFixed(1)).join(', ')}...
                    </div>
                  </div>
                  <div style={{ background: '#f0fdf4', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span><strong>7d Avg Daily:</strong></span>
                      <span style={{ color: '#22c55e', fontWeight: 'bold' }}>{avg7d.toFixed(1)} customers</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span><strong>7d Peak Day:</strong></span>
                      <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>{peak7d.toFixed(1)} customers</span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.5rem' }}>
                      <strong>Next 7 days:</strong> {forecasts["7d_sample"].map(val => val.toFixed(1)).join(', ')}
                    </div>
                  </div>
                </div>
              </RestaurantCard>
            );
          })}
        </RestaurantPerformanceGrid>
      </SummarySection>

      <SummarySection>
        <SummaryTitle style={{ display: 'flex', alignItems: 'center' }}>
          🏆 Key Achievements
          <InfoTooltip text="Production deployment recommendations based on model performance analysis and business impact assessment." />
        </SummaryTitle>
        <RecommendationsList>
          <RecommendationItem>
            <RecommendationNumber>1</RecommendationNumber>
            <div>Deploy {data.model_configuration.best_model} model for production forecasting with {data.business_impact.recommended_penalty_ratio} asymmetric loss ratio</div>
          </RecommendationItem>
          <RecommendationItem>
            <RecommendationNumber>2</RecommendationNumber>
            <div>Implement {data.business_impact.update_frequency} model retraining schedule to maintain forecast accuracy</div>
          </RecommendationItem>
          <RecommendationItem>
            <RecommendationNumber>3</RecommendationNumber>
            <div>Focus on top features: weekend patterns, capacity management, and seasonal trends for optimal performance</div>
          </RecommendationItem>
          <RecommendationItem>
            <RecommendationNumber>4</RecommendationNumber>
            <div>Monitor understaffing incidents closely - current reduction of {data.business_impact.understaffing_reduction_pct}% shows significant improvement</div>
          </RecommendationItem>
        </RecommendationsList>
      </SummarySection>
    </TabContent>
  );

  const renderOptionB = () => (
    <TabContent>
      <div style={{ textAlign: 'center', padding: '4rem 2rem', background: '#f8fafc', borderRadius: '12px', border: '2px dashed #cbd5e1' }}>
        <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🧠</div>
        <h3 style={{ color: '#475569', marginBottom: '1rem' }}>Deep Learning with Custom Loss</h3>
        <p style={{ color: '#64748b', fontSize: '1.1rem', marginBottom: '1.5rem' }}>
          Neural network (LSTM/GRU) with asymmetric loss, attention mechanisms for external factors, and uncertainty quantification
        </p>
        <div style={{ background: '#e2e8f0', color: '#475569', padding: '1rem', borderRadius: '8px', fontWeight: 'bold' }}>
          Implementation Coming Soon
        </div>
      </div>
    </TabContent>
  );

  const renderOptionC = () => (
    <TabContent>
      <div style={{ textAlign: 'center', padding: '4rem 2rem', background: '#f0fdf4', borderRadius: '12px', border: '2px dashed #bbf7d0' }}>
        <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>📊</div>
        <h3 style={{ color: '#166534', marginBottom: '1rem' }}>Probabilistic Approach</h3>
        <p style={{ color: '#15803d', fontSize: '1.1rem', marginBottom: '1.5rem' }}>
          Models that output full distributions, quantile regression or probabilistic neural networks, showing how different percentiles map to staffing strategies
        </p>
        <div style={{ background: '#dcfce7', color: '#166534', padding: '1rem', borderRadius: '8px', fontWeight: 'bold' }}>
          Implementation Coming Soon
        </div>
      </div>
    </TabContent>
  );

  return (
    <Container>
      <Title>Part 2: ML Models</Title>
      
      <Question>
        Q: Can machine learning models with custom asymmetric loss functions significantly improve restaurant demand forecasting accuracy and reduce operational costs?
      </Question>

      <TabContainer>
        <Tab 
          active={activeTab === 'A'} 
          onClick={() => setActiveTab('A')}
        >
          Option A: Machine Learning with Custom Objective
        </Tab>
        <Tab 
          active={activeTab === 'B'} 
          onClick={() => setActiveTab('B')}
        >
          Option B: Deep Learning with Custom Loss
        </Tab>
        <Tab 
          active={activeTab === 'C'} 
          onClick={() => setActiveTab('C')}
        >
          Option C: Probabilistic Approach
        </Tab>
      </TabContainer>

      {activeTab === 'A' && renderOptionA()}
      {activeTab === 'B' && renderOptionB()}
      {activeTab === 'C' && renderOptionC()}

    </Container>
  );
};

export default Part2;