// User types
export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  role: 'admin' | 'user';
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

// Authentication types
export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// Document types
export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  processed_date?: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  metadata?: Record<string, any>;
  owner_id: string;
  is_shared: boolean;
  permissions: string[];
}

export interface DocumentUpload {
  file: File;
  metadata?: Record<string, any>;
}

// Search types
export interface SearchQuery {
  query: string;
  filters?: {
    file_type?: string[];
    date_range?: {
      start: string;
      end: string;
    };
    owner?: string;
  };
  limit?: number;
  offset?: number;
}

export interface SearchResult {
  document_id: string;
  filename: string;
  content_snippet: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  processing_time: number;
}

// Library types
export interface Library {
  id: string;
  name: string;
  description?: string;
  owner_id: string;
  is_public: boolean;
  document_count: number;
  created_at: string;
  updated_at: string;
}

// Analytics types
export interface UsageStats {
  total_documents: number;
  total_searches: number;
  active_users: number;
  storage_used: number;
  upload_trends: Array<{
    date: string;
    count: number;
  }>;
  search_trends: Array<{
    date: string;
    count: number;
  }>;
}

// API Response types
export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  error?: string;
  status: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}