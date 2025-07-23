import { useState, useEffect } from 'react';
import {
  ChartBarIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  UserGroupIcon,
  ServerIcon,
} from '@heroicons/react/24/outline';
import Layout from '@/components/common/Layout';
import { analyticsApi } from '@/api/analytics';
import { UsageStats } from '@/types';
import toast from 'react-hot-toast';

export default function Analytics() {
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const [usageStats, healthData] = await Promise.all([
        analyticsApi.getUsageStats(),
        analyticsApi.getSystemHealth(),
      ]);
      
      setStats(usageStats);
      setSystemHealth(healthData);
    } catch (error: any) {
      toast.error('Failed to fetch analytics data');
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
  };

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getUsageColor = (percentage: number) => {
    if (percentage < 50) return 'bg-green-500';
    if (percentage < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const formatSystemResources = (details: any) => {
    if (!details) return null;
    
    return (
      <div className="space-y-3 mt-3">
        {details.cpu_usage_percent !== undefined && (
          <div>
            <div className="flex justify-between text-sm">
              <span>CPU Usage</span>
              <span className="font-medium">{details.cpu_usage_percent.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div 
                className={`h-2 rounded-full ${getUsageColor(details.cpu_usage_percent)}`}
                style={{ width: `${Math.min(details.cpu_usage_percent, 100)}%` }}
              ></div>
            </div>
          </div>
        )}
        
        {details.memory_usage_percent !== undefined && (
          <div>
            <div className="flex justify-between text-sm">
              <span>Memory Usage</span>
              <span className="font-medium">{details.memory_usage_percent.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div 
                className={`h-2 rounded-full ${getUsageColor(details.memory_usage_percent)}`}
                style={{ width: `${Math.min(details.memory_usage_percent, 100)}%` }}
              ></div>
            </div>
            {details.memory_available_gb && (
              <div className="text-xs text-gray-500 mt-1">
                Available: {details.memory_available_gb.toFixed(2)} GB
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const formatDiskUsage = (details: any) => {
    if (!details) return null;

    return (
      <div className="space-y-3 mt-3">
        {details.usage_percent !== undefined && (
          <div>
            <div className="flex justify-between text-sm">
              <span>Disk Usage</span>
              <span className="font-medium">{details.usage_percent.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
              <div 
                className={`h-2 rounded-full ${getUsageColor(details.usage_percent)}`}
                style={{ width: `${Math.min(details.usage_percent, 100)}%` }}
              ></div>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-3 gap-4 text-xs">
          {details.total_gb && (
            <div>
              <div className="text-gray-500">Total</div>
              <div className="font-medium">{details.total_gb.toFixed(1)} GB</div>
            </div>
          )}
          {details.used_gb && (
            <div>
              <div className="text-gray-500">Used</div>
              <div className="font-medium">{details.used_gb.toFixed(1)} GB</div>
            </div>
          )}
          {details.free_gb && (
            <div>
              <div className="text-gray-500">Free</div>
              <div className="font-medium text-green-600">{details.free_gb.toFixed(1)} GB</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <Layout>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-gray-200 rounded w-1/4"></div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="bg-white p-6 rounded-lg shadow">
                  <div className="h-6 bg-gray-200 rounded w-3/4 mb-2"></div>
                  <div className="h-8 bg-gray-200 rounded w-1/2"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
            <p className="mt-1 text-sm text-gray-500">
              System usage statistics and performance metrics.
            </p>
          </div>

          {/* Key Metrics */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <DocumentTextIcon className="h-6 w-6 text-gray-400" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">
                          Total Documents
                        </dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {stats.total_documents.toLocaleString()}
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <MagnifyingGlassIcon className="h-6 w-6 text-gray-400" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">
                          Total Searches
                        </dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {stats.total_searches.toLocaleString()}
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <UserGroupIcon className="h-6 w-6 text-gray-400" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">
                          Active Users
                        </dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {stats.active_users.toLocaleString()}
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <ServerIcon className="h-6 w-6 text-gray-400" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">
                          Storage Used
                        </dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {formatBytes(stats.storage_used)}
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* System Health */}
          {systemHealth && (
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    System Health
                  </h3>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getHealthStatusColor(systemHealth.status)}`}>
                    <div className={`w-2 h-2 rounded-full mr-2 ${systemHealth.status === 'healthy' ? 'bg-green-500' : systemHealth.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
                    {systemHealth.status.charAt(0).toUpperCase() + systemHealth.status.slice(1)}
                  </span>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {systemHealth.components?.map((component: any) => (
                    <div key={component.name} className="border rounded-lg p-4 hover:shadow-sm transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-sm font-medium text-gray-900 flex items-center">
                          {component.type === 'system_resources' && <ServerIcon className="h-4 w-4 mr-2 text-blue-500" />}
                          {component.type === 'disk_usage' && <ChartBarIcon className="h-4 w-4 mr-2 text-green-500" />}
                          {component.name}
                        </h4>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getHealthStatusColor(component.status)}`}>
                          {component.status}
                        </span>
                      </div>
                      
                      {/* Render formatted details based on component type */}
                      {component.type === 'system_resources' && formatSystemResources(component.details)}
                      {component.type === 'disk_usage' && formatDiskUsage(component.details)}
                      
                      {/* Fallback for other component types */}
                      {component.type === 'other' && component.details && (
                        <div className="mt-2 text-sm text-gray-600">
                          {typeof component.details === 'string' ? component.details : JSON.stringify(component.details, null, 2)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Usage Trends */}
          {stats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload Trends */}
              <div className="bg-white shadow rounded-lg p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Document Uploads (Last 30 Days)
                </h3>
                <div className="space-y-2">
                  {stats.upload_trends && stats.upload_trends.length > 0 ? (
                    stats.upload_trends.slice(-10).map((trend, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">{trend.date}</span>
                        <div className="flex items-center space-x-2">
                          <div className="bg-blue-200 h-2 rounded-full" style={{ width: `${Math.max(trend.count * 10, 10)}px` }}></div>
                          <span className="text-sm font-medium text-gray-900">{trend.count}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500">No upload data available</p>
                  )}
                </div>
              </div>

              {/* Search Trends */}
              <div className="bg-white shadow rounded-lg p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Search Activity (Last 30 Days)
                </h3>
                <div className="space-y-2">
                  {stats.search_trends && stats.search_trends.length > 0 ? (
                    stats.search_trends.slice(-10).map((trend, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">{trend.date}</span>
                        <div className="flex items-center space-x-2">
                          <div className="bg-green-200 h-2 rounded-full" style={{ width: `${Math.max(trend.count * 5, 10)}px` }}></div>
                          <span className="text-sm font-medium text-gray-900">{trend.count}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500">No search data available</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}