import { apiClient } from './client';
import { LoginCredentials, RegisterData, AuthResponse, User } from '@/types';

export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    return apiClient.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },

  register: async (data: RegisterData): Promise<User> => {
    return apiClient.post('/auth/register', data);
  },

  getCurrentUser: async (): Promise<User> => {
    return apiClient.get('/auth/me');
  },

  logout: async (): Promise<void> => {
    return apiClient.post('/auth/logout');
  },

  requestPasswordReset: async (email: string): Promise<void> => {
    return apiClient.post('/auth/request-password-reset', { email });
  },

  resetPassword: async (token: string, newPassword: string): Promise<void> => {
    return apiClient.post('/auth/reset-password', {
      token,
      new_password: newPassword,
    });
  },
};