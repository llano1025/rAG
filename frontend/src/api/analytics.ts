import { apiClient } from './client';
import { UsageStats } from '@/types';

export const analyticsApi = {
  getUsageStats: async (): Promise<UsageStats> => {
    return apiClient.get('/analytics/usage-stats');
  },

  getSystemHealth: async (): Promise<{
    status: string;
    components: Array<{
      name: string;
      status: 'healthy' | 'warning' | 'error';
      details?: string;
    }>;
  }> => {
    return apiClient.get('/health');
  },

  getPerformanceMetrics: async (): Promise<{
    response_times: Array<{ timestamp: string; value: number }>;
    error_rates: Array<{ timestamp: string; value: number }>;
    throughput: Array<{ timestamp: string; value: number }>;
  }> => {
    return apiClient.get('/analytics/performance');
  },
};