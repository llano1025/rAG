import { apiClient } from './client';
import { User, PaginatedResponse } from '@/types';

export const adminApi = {
  getUsers: async (page = 1, perPage = 20): Promise<PaginatedResponse<User>> => {
    return apiClient.get(`/admin/users?page=${page}&per_page=${perPage}`);
  },

  getUserById: async (id: string): Promise<User> => {
    return apiClient.get(`/admin/users/${id}`);
  },

  updateUser: async (id: string, updates: Partial<User>): Promise<User> => {
    return apiClient.put(`/admin/users/${id}`, updates);
  },

  deleteUser: async (id: string): Promise<void> => {
    return apiClient.delete(`/admin/users/${id}`);
  },

  toggleUserStatus: async (id: string): Promise<User> => {
    return apiClient.post(`/admin/users/${id}/toggle-status`);
  },

  getSystemSettings: async (): Promise<Record<string, any>> => {
    return apiClient.get('/admin/settings');
  },

  updateSystemSettings: async (settings: Record<string, any>): Promise<void> => {
    return apiClient.put('/admin/settings', settings);
  },

  getApiKeys: async (): Promise<Array<{
    id: string;
    name: string;
    key_preview: string;
    created_at: string;
    last_used?: string;
    is_active: boolean;
  }>> => {
    return apiClient.get('/admin/api-keys');
  },

  createApiKey: async (name: string, permissions: string[]): Promise<{
    id: string;
    name: string;
    key: string;
  }> => {
    return apiClient.post('/admin/api-keys', { name, permissions });
  },

  revokeApiKey: async (id: string): Promise<void> => {
    return apiClient.delete(`/admin/api-keys/${id}`);
  },
};