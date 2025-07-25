import { apiClient } from './client';
import { UsageStats } from '@/types';

export const analyticsApi = {
  getUsageStats: async (): Promise<UsageStats> => {
    return apiClient.get('/api/analytics/usage-stats');
  },

  getSystemHealth: async (): Promise<{
    status: string;
    components: Array<{
      name: string;
      status: 'healthy' | 'warning' | 'error';
      details?: any;
      type: 'system_resources' | 'disk_usage' | 'other';
    }>;
  }> => {
    const healthData: any = await apiClient.get('/api/analytics/system-health');
    
    // The analytics endpoint already returns the correct format
    return {
      status: healthData.status || 'healthy',
      components: healthData.components || []
    };
  },

  getPerformanceMetrics: async (): Promise<{
    response_times: Array<{ timestamp: string; value: number }>;
    error_rates: Array<{ timestamp: string; value: number }>;
    throughput: Array<{ timestamp: string; value: number }>;
  }> => {
    return apiClient.get('/api/analytics/performance');
  },
};