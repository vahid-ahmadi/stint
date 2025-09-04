export interface RestaurantCorrelations {
  temperature: number;
  economic_indicator: number;
  competitor_promo: number;
  social_trend: number;
  local_event: number;
  reputation_score: number;
}

export interface RestaurantDemandData {
  correlations: RestaurantCorrelations;
  weekend_lift: number;
  avg_customers: number;
}

export interface TemperatureImpact {
  category: string;
  avg_demand: number;
}

export interface ExternalFactors {
  temperature: TemperatureImpact[];
  competitor_promo_impact: number;
  local_event_lift: number;
}

export interface RestaurantPeakData {
  peak_hours: number[];
  peak_days: string[];
  peak_volatility: number;
  avg_peak_demand: number;
}

export interface DailyPattern {
  day_of_week: string;
  mean: number;
  std: number;
}

export interface PeakAnalysis {
  overall_peak_hours: number[];
  by_restaurant_type: Record<string, RestaurantPeakData>;
  daily_patterns: DailyPattern[];
}

export interface RestaurantDifficulty {
  difficult_hours: number[];
  peak_volatility: number;
  non_peak_volatility: number;
  overall_cv: number;
}

export interface CapacityUtilization {
  average: number;
  peak: number;
}

export interface PeakInsights {
  revenue_multiplier: number;
  peak_hours: number[];
  capacity_utilization: Record<string, CapacityUtilization>;
}

export interface QuantifiedImpacts {
  temperature_effect: number;
  competitor_promo_effect: number | null;
  local_event_effect: number;
  economic_effect: number;
}

export interface VolatilityAnalysis {
  staffing_challenge_score: number;
  peak_unpredictability: number;
  operational_difficulty: string;
}

export interface BusinessInsights {
  key_findings: string[];
  actionable_recommendations: string[];
  forecasting_challenges: string[];
  quantified_impacts?: QuantifiedImpacts;
  peak_insights?: PeakInsights;
  volatility_analysis?: Record<string, VolatilityAnalysis>;
}

export interface SummaryStats {
  total_records: number;
  date_range: string;
  restaurant_types: string[];
  avg_daily_customers: number;
}

export interface AnalysisResults {
  demand_drivers: Record<string, RestaurantDemandData>;
  external_factors: ExternalFactors;
  peak_analysis: PeakAnalysis;
  difficulty_analysis: Record<string, RestaurantDifficulty>;
  insights: BusinessInsights;
  summary_stats: SummaryStats;
}