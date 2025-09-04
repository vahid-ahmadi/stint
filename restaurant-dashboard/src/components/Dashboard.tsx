import React, { useState } from 'react';
import { 
  BarChart, Bar, LineChart, Line, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { AnalysisResults } from '../types/AnalysisTypes';

interface DashboardProps {
  data: AnalysisResults;
}

const RESTAURANT_COLORS: Record<string, string> = {
  'casual bistro': '#6366F1',
  'seafood': '#10B981', 
  'family restaurant': '#8B5CF6',
  'fine dining': '#34D399',
  'fast casual': '#6B7280'
};

const Dashboard: React.FC<DashboardProps> = ({ data }) => {
  const [activeMainTab, setActiveMainTab] = useState('part1');
  const [activePart1Tab, setActivePart1Tab] = useState('overview');

  // Debug handlers with logging
  const handleMainTabClick = (tabId: string) => {
    console.log('Main tab clicked:', tabId);
    setActiveMainTab(tabId);
  };

  const handlePart1TabClick = (tabId: string) => {
    console.log('Part1 tab clicked:', tabId);
    setActivePart1Tab(tabId);
  };

  // Prepare data for visualizations
  const weekendLiftData = Object.entries(data.demand_drivers).map(([type, info]) => ({
    restaurant_type: type,
    weekend_lift: info.weekend_lift,
    avg_customers: info.avg_customers
  }));

  const temperatureData = data.external_factors.temperature.map(item => ({
    ...item,
    percentage_change: ((item.avg_demand - data.external_factors.temperature[0].avg_demand) / 
                       data.external_factors.temperature[0].avg_demand * 100).toFixed(1)
  }));

  const correlationData = Object.entries(data.demand_drivers).map(([type, info]) => ({
    restaurant_type: type,
    temperature: (info.correlations.temperature * 100).toFixed(1),
    economic: (info.correlations.economic_indicator * 100).toFixed(1),
    competition: (Math.abs(info.correlations.competitor_promo) * 100).toFixed(1),
    events: (info.correlations.local_event * 100).toFixed(1),
    reputation: (Math.abs(info.correlations.reputation_score) * 100).toFixed(1),
    social: (info.correlations.social_trend * 100).toFixed(1)
  }));

  const dailyPatternsData = data.peak_analysis.daily_patterns.map(day => ({
    ...day,
    day_short: day.day_of_week.substring(0, 3)
  }));

  const volatilityData = Object.entries(data.difficulty_analysis).map(([type, info]) => ({
    restaurant_type: type,
    peak_volatility: info.peak_volatility,
    non_peak_volatility: info.non_peak_volatility,
    cv_score: (info.overall_cv * 100).toFixed(1)
  }));

  const MainTabButton: React.FC<{id: string, label: string, status: string, isActive: boolean, onClick: () => void}> = 
    ({ id, label, status, isActive, onClick }) => (
      <button
        onClick={onClick}
        className={`px-6 py-3 font-medium rounded-xl transition-all flex items-center gap-2 ${
          isActive 
            ? 'bg-blue-600 text-white shadow-lg transform scale-105' 
            : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
        }`}
      >
        <span>{label}</span>
        <span className={`px-2 py-1 text-xs rounded-full ${
          status === 'Complete' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-yellow-100 text-yellow-800'
        }`}>
          {status}
        </span>
      </button>
    );

  const SubTabButton: React.FC<{id: string, label: string, isActive: boolean, onClick: () => void}> = 
    ({ id, label, isActive, onClick }) => (
      <button
        onClick={onClick}
        className={`px-4 py-2 font-medium rounded-lg transition-colors text-sm ${
          isActive 
            ? 'bg-blue-600 text-white shadow-md' 
            : 'bg-gray-50 text-gray-700 hover:bg-gray-100 border border-gray-200'
        }`}
      >
        {label}
      </button>
    );

  const MetricCard: React.FC<{title: string, value: string | number, subtitle?: string, color?: string, icon?: string}> = 
    ({ title, value, subtitle, color = 'bg-blue-50', icon }) => (
      <div className={`p-6 rounded-xl ${color} border-l-4 border-blue-600 shadow-sm hover:shadow-md transition-shadow`}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-gray-600 uppercase tracking-wide">{title}</h3>
          {icon && <span className="text-2xl">{icon}</span>}
        </div>
        <p className="text-3xl font-bold text-gray-900 mb-1">{value}</p>
        {subtitle && <p className="text-sm text-gray-600 mt-2">{subtitle}</p>}
      </div>
    );

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Executive Summary Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-2">📊 Restaurant Demand Analysis - Executive Summary</h2>
        <p className="text-blue-100 text-lg">
          Comprehensive analysis of {data.summary_stats.total_records.toLocaleString()} records across {data.summary_stats.restaurant_types.length} restaurant types
          spanning {data.summary_stats.date_range}
        </p>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          title="Total Data Points" 
          value={data.summary_stats.total_records.toLocaleString()} 
          subtitle="30-min intervals over 4 years"
          color="bg-blue-50"
          icon="📊"
        />
        <MetricCard 
          title="Peak Revenue Multiplier" 
          value={`${data.insights.peak_insights?.revenue_multiplier?.toFixed(1) || 'N/A'}%`}
          subtitle="Higher revenue during peaks"
          color="bg-green-50"
          icon="💰"
        />
        <MetricCard 
          title="Daily Customer Volume" 
          value={Math.round(data.summary_stats.avg_daily_customers).toLocaleString()} 
          subtitle="Across all restaurant locations"
          color="bg-yellow-50"
          icon="👥"
        />
        <MetricCard 
          title="Critical Peak Hours" 
          value={data.peak_analysis.overall_peak_hours.length} 
          subtitle={`${data.peak_analysis.overall_peak_hours.join(':00, ')}:00`}
          color="bg-purple-50"
          icon="⏰"
        />
      </div>

      {/* Business Impact Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-blue-500">
          <h3 className="text-lg font-bold text-blue-700 mb-3">🎯 Demand Drivers Impact</h3>
          <div className="space-y-3">
            {data.insights.quantified_impacts && Object.entries(data.insights.quantified_impacts).map(([factor, impact]) => (
              impact !== null && !isNaN(Number(impact)) && (
                <div key={factor} className="flex justify-between items-center">
                  <span className="text-sm capitalize">{factor.replace('_', ' ')}:</span>
                  <span className="font-semibold text-blue-600">{Number(impact).toFixed(1)}%</span>
                </div>
              )
            ))}
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-green-500">
          <h3 className="text-lg font-bold text-green-700 mb-3">📈 Peak Performance</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm">Peak Hours:</span>
              <span className="font-semibold">{data.peak_analysis.overall_peak_hours.join(', ')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Weekend Leader:</span>
              <span className="font-semibold text-green-600">Seafood (+52.2%)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Capacity Utilization:</span>
              <span className="font-semibold">53-78% at peak</span>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-orange-500">
          <h3 className="text-lg font-bold text-orange-700 mb-3">⚠️ Risk Factors</h3>
          <div className="space-y-3">
            {data.insights.volatility_analysis && Object.entries(data.insights.volatility_analysis).slice(0, 2).map(([type, analysis]) => (
              <div key={type} className="">
                <div className="flex justify-between text-sm">
                  <span className="capitalize">{type}:</span>
                  <span className={`font-semibold ${
                    analysis.operational_difficulty === 'High' ? 'text-red-600' : 
                    analysis.operational_difficulty === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {analysis.operational_difficulty}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Comprehensive Business Insights */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4 text-gray-800">🔍 Comprehensive Business Intelligence</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-semibold text-green-700 mb-2">📊 Key Findings</h3>
            <ul className="space-y-1">
              {data.insights.key_findings.map((finding, index) => (
                <li key={index} className="text-sm text-gray-600 flex items-start">
                  <span className="text-green-500 mr-2">•</span>
                  {finding}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-blue-700 mb-2">🎯 Recommendations</h3>
            <ul className="space-y-1">
              {data.insights.actionable_recommendations.map((rec, index) => (
                <li key={index} className="text-sm text-gray-600 flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-orange-700 mb-2">⚠️ Challenges</h3>
            <ul className="space-y-1">
              {data.insights.forecasting_challenges.map((challenge, index) => (
                <li key={index} className="text-sm text-gray-600 flex items-start">
                  <span className="text-orange-500 mr-2">•</span>
                  {challenge}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDemandAnalysis = () => (
    <div className="space-y-6">
      {/* Weekend vs Weekday Analysis */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Weekend Demand Lift by Restaurant Type</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={weekendLiftData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="restaurant_type" 
              angle={-45} 
              textAnchor="end" 
              height={100}
            />
            <YAxis label={{ value: 'Weekend Lift (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: any) => [`${value}%`, 'Weekend Lift']} />
            <Bar dataKey="weekend_lift" fill="#00C49F" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Daily Patterns */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Daily Demand Patterns</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={dailyPatternsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day_short" />
            <YAxis label={{ value: 'Avg Customers', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Line 
              type="monotone" 
              dataKey="mean" 
              stroke="#0088FE" 
              strokeWidth={3}
              dot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Temperature Impact */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Temperature Impact on Demand</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={temperatureData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" />
            <YAxis label={{ value: 'Avg Customers', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: any, name: string) => {
              if (name === 'avg_demand') return [`${value} customers`, 'Average Demand'];
              return [value, name];
            }} />
            <Bar dataKey="avg_demand" fill="#FFBB28" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderCorrelationAnalysis = () => (
    <div className="space-y-6">
      {/* Factor Correlation Radar Chart */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Demand Factor Correlations by Restaurant Type</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {correlationData.slice(0, 2).map((restaurant, index) => (
            <div key={restaurant.restaurant_type} className="space-y-2">
              <h3 className="text-lg font-semibold text-center capitalize">
                {restaurant.restaurant_type}
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={[{
                  factor: 'Temperature',
                  value: Math.abs(parseFloat(restaurant.temperature)),
                  fullMark: 100
                }, {
                  factor: 'Economic',
                  value: Math.abs(parseFloat(restaurant.economic)),
                  fullMark: 100
                }, {
                  factor: 'Competition',
                  value: Math.abs(parseFloat(restaurant.competition)),
                  fullMark: 100
                }, {
                  factor: 'Events',
                  value: Math.abs(parseFloat(restaurant.events)),
                  fullMark: 100
                }, {
                  factor: 'Reputation',
                  value: Math.abs(parseFloat(restaurant.reputation)),
                  fullMark: 100
                }, {
                  factor: 'Social',
                  value: Math.abs(parseFloat(restaurant.social)),
                  fullMark: 100
                }]}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="factor" tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis 
                    angle={45} 
                    domain={[0, 50]} 
                    tick={{ fontSize: 10 }}
                  />
                  <Radar
                    dataKey="value"
                    stroke={RESTAURANT_COLORS[restaurant.restaurant_type]}
                    fill={RESTAURANT_COLORS[restaurant.restaurant_type]}
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>

      {/* External Factor Impacts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard 
          title="Competitor Impact" 
          value={`${data.external_factors.competitor_promo_impact}%`}
          subtitle="When promotions active"
          color="bg-red-50"
        />
        <MetricCard 
          title="Local Event Boost" 
          value={`+${data.external_factors.local_event_lift}%`}
          subtitle="During events"
          color="bg-green-50"
        />
        <MetricCard 
          title="Weather Variance" 
          value={`${Math.max(...temperatureData.map(t => parseFloat(t.percentage_change)))}%`}
          subtitle="Peak temperature effect"
          color="bg-yellow-50"
        />
      </div>
    </div>
  );

  const renderVolatilityAnalysis = () => (
    <div className="space-y-6">
      {/* Volatility Comparison */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Demand Volatility Analysis</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={volatilityData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis 
              dataKey="restaurant_type" 
              type="category" 
              width={120}
              tick={{ fontSize: 12 }}
            />
            <Tooltip />
            <Legend />
            <Bar dataKey="peak_volatility" fill="#FF8042" name="Peak Period Volatility" />
            <Bar dataKey="non_peak_volatility" fill="#0088FE" name="Non-Peak Volatility" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Difficulty Scores */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Forecasting Difficulty by Restaurant Type</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {volatilityData.map((restaurant, index) => (
            <div key={restaurant.restaurant_type} 
                 className="p-4 border rounded-lg bg-gray-50">
              <h3 className="font-semibold text-lg capitalize mb-2">
                {restaurant.restaurant_type}
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">CV Score:</span>
                  <span className="font-medium">{restaurant.cv_score}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 h-2 rounded-full"
                    style={{ width: `${Math.min(parseFloat(restaurant.cv_score), 100)}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500">
                  {parseFloat(restaurant.cv_score) < 30 ? 'Low' : 
                   parseFloat(restaurant.cv_score) < 50 ? 'Medium' : 'High'} difficulty
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderBusinessImpact = () => {
    // Prepare capacity utilization data for visualization
    const capacityData = data.insights.peak_insights?.capacity_utilization ? 
      Object.entries(data.insights.peak_insights.capacity_utilization)
        .filter(([type, util]) => type !== 'null' && util.average !== null)
        .map(([type, util]) => ({
          restaurant_type: type,
          average_utilization: (util.average * 100).toFixed(1),
          peak_utilization: (util.peak * 100).toFixed(1),
          utilization_gap: ((util.peak - util.average) * 100).toFixed(1)
        })) : [];

    return (
      <div className="space-y-6">
        {/* Strategic Business Overview */}
        <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-3">💼 Strategic Business Impact Analysis</h2>
          <p className="text-green-100">
            Quantified insights for optimizing staffing, pricing, and operational efficiency across restaurant types
          </p>
        </div>

        {/* Capacity Utilization Analysis */}
        {capacityData.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">🏪 Capacity Utilization Efficiency</h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={capacityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="restaurant_type" 
                  angle={-45} 
                  textAnchor="end" 
                  height={100}
                />
                <YAxis label={{ value: 'Utilization (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value: any, name: string) => [`${value}%`, name]} />
                <Legend />
                <Bar dataKey="average_utilization" fill="#0088FE" name="Average Utilization" />
                <Bar dataKey="peak_utilization" fill="#FF8042" name="Peak Utilization" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Revenue Impact Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4 text-green-700">💰 Revenue Optimization Opportunities</h3>
            <div className="space-y-4">
              <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                <h4 className="font-semibold text-green-800">Peak Hour Revenue Multiplier</h4>
                <p className="text-2xl font-bold text-green-600">{data.insights.peak_insights?.revenue_multiplier?.toFixed(1)}%</p>
                <p className="text-sm text-green-700">Higher revenue during peak hours (18:00-20:00)</p>
              </div>
              
              <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                <h4 className="font-semibold text-blue-800">Weekend Premium</h4>
                <p className="text-2xl font-bold text-blue-600">+52.2%</p>
                <p className="text-sm text-blue-700">Seafood restaurants show highest weekend lift</p>
              </div>
              
              <div className="p-4 bg-purple-50 rounded-lg border-l-4 border-purple-500">
                <h4 className="font-semibold text-purple-800">Weather Sensitivity</h4>
                <p className="text-2xl font-bold text-purple-600">+9.4%</p>
                <p className="text-sm text-purple-700">Demand increase in mild-warm weather</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-bold mb-4 text-orange-700">⚠️ Risk Management Insights</h3>
            <div className="space-y-4">
              {data.insights.volatility_analysis && Object.entries(data.insights.volatility_analysis)
                .slice(0, 3)
                .map(([type, analysis]) => (
                <div key={type} className="p-3 bg-gray-50 rounded-lg border">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium capitalize">{type}</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      analysis.operational_difficulty === 'High' ? 'bg-red-100 text-red-800' :
                      analysis.operational_difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {analysis.operational_difficulty} Risk
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">
                    Staffing Challenge Score: {(analysis.staffing_challenge_score * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Actionable Recommendations Dashboard */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-bold mb-4 text-blue-700">🎯 Strategic Recommendations Implementation Guide</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-green-700 mb-3">Immediate Actions (0-30 days)</h4>
              <ul className="space-y-2">
                {data.insights.actionable_recommendations.slice(0, 3).map((rec, index) => (
                  <li key={index} className="flex items-start text-sm">
                    <span className="text-green-500 mr-2 mt-1">✓</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-700 mb-3">Strategic Initiatives (30-90 days)</h4>
              <ul className="space-y-2">
                {data.insights.actionable_recommendations.slice(3).map((rec, index) => (
                  <li key={index} className="flex items-start text-sm">
                    <span className="text-blue-500 mr-2 mt-1">→</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Main Task Navigation */}
        <div className="flex justify-center gap-4 mb-8">
          <MainTabButton 
            id="part1" 
            label="Part 1: Business Analysis" 
            status="Complete"
            isActive={activeMainTab === 'part1'} 
            onClick={() => handleMainTabClick('part1')} 
          />
          <MainTabButton 
            id="part2" 
            label="Part 2: Forecasting Model" 
            status="Incomplete"
            isActive={activeMainTab === 'part2'} 
            onClick={() => handleMainTabClick('part2')} 
          />
          <MainTabButton 
            id="part3" 
            label="Part 3: Model Evaluation" 
            status="Incomplete"
            isActive={activeMainTab === 'part3'} 
            onClick={() => handleMainTabClick('part3')} 
          />
        </div>

        {/* Part 1 Content */}
        {activeMainTab === 'part1' && (
          <div>
            {/* Sub-tab Navigation for Part 1 */}
            <div className="flex flex-wrap gap-2 mb-6 bg-white p-3 rounded-xl shadow-sm border border-gray-200">
              <SubTabButton 
                id="overview" 
                label="📊 Executive Overview" 
                isActive={activePart1Tab === 'overview'} 
                onClick={() => handlePart1TabClick('overview')} 
              />
              <SubTabButton 
                id="demand" 
                label="📈 Demand Patterns" 
                isActive={activePart1Tab === 'demand'} 
                onClick={() => handlePart1TabClick('demand')} 
              />
              <SubTabButton 
                id="correlation" 
                label="🔗 Factor Analysis" 
                isActive={activePart1Tab === 'correlation'} 
                onClick={() => handlePart1TabClick('correlation')} 
              />
              <SubTabButton 
                id="volatility" 
                label="⚡ Volatility Study" 
                isActive={activePart1Tab === 'volatility'} 
                onClick={() => handlePart1TabClick('volatility')} 
              />
              <SubTabButton 
                id="business-impact" 
                label="💼 Business Impact" 
                isActive={activePart1Tab === 'business-impact'} 
                onClick={() => handlePart1TabClick('business-impact')} 
              />
            </div>

            {/* Part 1 Sub-tab Content */}
            {activePart1Tab === 'overview' && renderOverview()}
            {activePart1Tab === 'demand' && renderDemandAnalysis()}
            {activePart1Tab === 'correlation' && renderCorrelationAnalysis()}
            {activePart1Tab === 'volatility' && renderVolatilityAnalysis()}
            {activePart1Tab === 'business-impact' && renderBusinessImpact()}
          </div>
        )}

        {/* Part 2 Placeholder */}
        {activeMainTab === 'part2' && (
          <div className="text-center py-20">
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-8 max-w-md mx-auto">
              <div className="text-6xl mb-4">🚧</div>
              <h2 className="text-2xl font-bold text-yellow-800 mb-2">Part 2: Coming Soon</h2>
              <p className="text-yellow-700">Advanced forecasting models with custom asymmetric loss functions</p>
            </div>
          </div>
        )}

        {/* Part 3 Placeholder */}
        {activeMainTab === 'part3' && (
          <div className="text-center py-20">
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-8 max-w-md mx-auto">
              <div className="text-6xl mb-4">🚧</div>
              <h2 className="text-2xl font-bold text-yellow-800 mb-2">Part 3: Coming Soon</h2>
              <p className="text-yellow-700">Model evaluation and peak performance analysis</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;