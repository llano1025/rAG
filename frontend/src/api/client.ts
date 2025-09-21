import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import Cookies from 'js-cookie';
import toast from 'react-hot-toast';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      timeout: 30000, // Default timeout for most requests
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = Cookies.get('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling and format unwrapping
    this.client.interceptors.response.use(
      (response) => {
        // Check if response follows new StandardResponse format
        if (response.data && typeof response.data === 'object' && 'success' in response.data) {
          // New StandardResponse format - extract data
          if (response.data.success) {
            // For paginated responses, preserve pagination metadata at root level
            if (response.data.pagination) {
              return {
                ...response,
                data: {
                  ...response.data.data,
                  ...response.data.pagination
                }
              };
            } else {
              // For standard responses, return the data property
              return {
                ...response,
                data: response.data.data
              };
            }
          } else {
            // Handle error in StandardResponse format
            const error = response.data.error;
            const errorMessage = error?.message || 'An error occurred';
            toast.error(errorMessage);
            return Promise.reject(new Error(errorMessage));
          }
        }
        // Return response as-is for old format or non-API responses
        return response;
      },
      (error) => {
        if (error.response?.status === 401) {
          Cookies.remove('access_token');
          window.location.href = '/auth/login';
        } else if (error.response?.status >= 500) {
          toast.error('Server error occurred. Please try again later.');
        } else if (error.response?.data?.error) {
          // Handle new standardized error format
          const errorData = error.response.data.error;
          const errorMessage = errorData.message || 'An error occurred';
          toast.error(errorMessage);
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete(url, config);
    return response.data;
  }

  async upload<T>(url: string, formData: FormData, onProgress?: (progress: number) => void): Promise<T> {
    // Use extended timeout for document uploads (10 minutes)
    const isDocumentUpload = url.includes('/documents/upload') || url.includes('/documents/batch-upload');
    const timeout = isDocumentUpload ? 600000 : 30000; // 10 minutes vs 30 seconds

    const response = await this.client.post(url, formData, {
      timeout,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    return response.data;
  }

  async downloadBlob(url: string, config?: AxiosRequestConfig): Promise<Blob> {
    const response = await this.client.get(url, {
      ...config,
      responseType: 'blob',
    });
    // Return the blob directly, not response.data
    return response.data as Blob;
  }
}

export const apiClient = new ApiClient();