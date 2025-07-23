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
      details?: any;
      type: 'system_resources' | 'disk_usage' | 'other';
    }>;
  }> => {
    const healthData = await apiClient.get('/health');
    
    // Transform the health data format with better parsing
    const components = Object.entries(healthData.components || {}).map(([name, component]: [string, any]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      status: component.status || 'healthy',
      details: component.details || {},
      type: name as 'system_resources' | 'disk_usage' | 'other'
    }));
    
    return {
      status: healthData.status || 'healthy',
      components
    };
  },

  getPerformanceMetrics: async (): Promise<{
    response_times: Array<{ timestamp: string; value: number }>;
    error_rates: Array<{ timestamp: string; value: number }>;
    throughput: Array<{ timestamp: string; value: number }>;
  }> => {
    return apiClient.get('/analytics/performance');
  },
};