// User types
export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  role?: 'admin' | 'user';
  roles?: string[];
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
  title: string;
  description?: string;
  content_type: string;
  file_size: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  version: number;
  is_public: boolean;
  user_id: string;
  created_at: string;
  updated_at: string;
  processed_at?: string;
  language?: string;
  tags: string[];
  chunks_count: number;
  embedding_model?: string;
}

export interface DocumentUpload {
  file: File;
  metadata?: Record<string, any>;
  embedding_model?: string;
}

// Search types
export interface SearchFilters {
  folder_ids?: string[];
  tags?: string[];  // Changed from tag_ids to tags
  tag_match_mode?: 'any' | 'all' | 'exact';  // New tag matching mode
  exclude_tags?: string[];  // New tag exclusion support
  file_types?: string[];
  date_range?: [string, string];
  file_size_range?: [number, number];  // New file size filtering
  language?: string;  // New language filtering
  is_public?: boolean;  // New public/private filtering
  metadata_filters?: Record<string, any>;
  embedding_model?: string;  // Filter by embedding model used
}

export interface SearchQuery {
  query: string;
  filters?: SearchFilters;
  semantic_search?: boolean;
  hybrid_search?: boolean;
  top_k?: number;
  similarity_threshold?: number;
  page?: number;
  page_size?: number;
  sort?: string;
  // Reranker settings
  enable_reranking?: boolean;
  reranker_model?: string;
  rerank_score_weight?: number;
  min_rerank_score?: number;
  // Embedding model selection
  embedding_model?: string;
}

export interface SearchResult {
  document_id: string;
  filename: string;
  content_snippet: string;
  score: number;
  metadata?: Record<string, any>;
  // Reranker result fields
  original_score?: number;
  rerank_score?: number;
  combined_score?: number;
  reranker_model?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total_hits: number;
  execution_time_ms: number;
  filters_applied?: SearchFilters;
  query_vector_id?: string;
  query?: string;
  processing_time?: number;
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

// Reranker types
export interface RerankerModel {
  alias: string;
  full_name: string;
  description: string;
  performance_tier: 'fast' | 'balanced' | 'accurate';
  provider: string;
}

export interface RerankerHealth {
  status: 'healthy' | 'error';
  model_name?: string;
  response_time_ms?: number;
  message?: string;
  timestamp: string;
}

// Embedding Model types
export interface EmbeddingModel {
  name: string;
  display_name: string;
  description: string;
  dimension: number;
  provider: string;
  max_length?: number;
}