import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { 
  BarChart, Bar, LineChart, Line, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { AnalysisResults } from '../types/AnalysisTypes';

const SlideContent = styled.div`
  max-width: 1400px;
  width: 100%;
  padding: 2rem;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 2rem;
  text-align: center;
  color: #333;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
  gap: 2rem;
`;

const ChartContainer = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const ChartTitle = styled.h3`
  font-size: 1.3rem;
  font-weight: bold;
  margin-bottom: 1rem;
  color: #555;
`;

const DemandPatterns: React.FC = () => {
  const [data, setData] = useState<AnalysisResults | null>(null);

  useEffect(() => {
    fetch('/analysis_results.json')
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);

  if (!data) return <div>Loading...</div>;

  const weekendLiftData = Object.entries(data.demand_drivers).map(([type, info]) => ({
    restaurant_type: type.split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' '),
    weekend_lift: info.weekend_lift
  }));

  const dailyPatternsData = data.peak_analysis.daily_patterns.map(day => ({
    day: day.day_of_week.substring(0, 3),
    customers: Math.round(day.mean)
  }));

  const temperatureData = data.external_factors.temperature.map(item => ({
    temp: item.category,
    demand: Math.round(item.avg_demand)
  }));

  return (
    <SlideContent>
      <Title>Demand Pattern Analysis</Title>
      <ChartsGrid>
        <ChartContainer>
          <ChartTitle>Weekend Lift by Restaurant Type</ChartTitle>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={weekendLiftData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="restaurant_type" angle={-20} textAnchor="end" height={80} />
              <YAxis label={{ value: 'Lift (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="weekend_lift" fill="#00C49F" />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer>
          <ChartTitle>Daily Customer Patterns</ChartTitle>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={dailyPatternsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis label={{ value: 'Avg Customers', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="customers" stroke="#667eea" strokeWidth={3} dot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer>
          <ChartTitle>Temperature Impact on Demand</ChartTitle>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={temperatureData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="temp" />
              <YAxis label={{ value: 'Avg Demand', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="demand" fill="#FFBB28" />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </ChartsGrid>
    </SlideContent>
  );
};

export default DemandPatterns;