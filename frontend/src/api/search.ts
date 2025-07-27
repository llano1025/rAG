import { apiClient } from './client';
import { SearchQuery, SearchResponse } from '@/types';

export interface SearchSuggestion {
  type: string;
  text: string;
  icon: string;
}

export interface FilterOption {
  value: string;
  label: string;
  count?: number;
  icon?: string;
}

export interface DateRange {
  min_date?: string;
  max_date?: string;
}

export interface FileSizeRange {
  min_size: number;
  max_size: number;
  avg_size: number;
}

export interface SearchType {
  value: string;
  label: string;
  description: string;
}

export interface AvailableFilters {
  file_types: FilterOption[];
  tags: FilterOption[];
  languages: FilterOption[];
  folders: FilterOption[];
  date_range: DateRange;
  file_size_range: FileSizeRange;
  search_types: SearchType[];
}

export interface SavedSearch {
  id: string;
  name: string;
  description?: string;
  query_text: string;
  search_type: string;
  filters: any;
  max_results: number;
  similarity_threshold?: number;
  usage_count: number;
  created_at: string;
  last_used?: string;
  tags: string[];
  is_public: boolean;
}

export interface RecentSearch {
  id: string;
  query_text: string;
  query_type: string;
  created_at: string;
  results_count: number;
  search_time_ms?: number;
  filters: any;
}

export const searchApi = {
  search: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/api/search', query);
  },

  semanticSearch: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/api/search/semantic', query);
  },

  hybridSearch: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/api/search/hybrid', query);
  },

  getSearchHistory: async (): Promise<RecentSearch[]> => {
    return apiClient.get('/api/search/history');
  },

  getRecentSearches: async (limit: number = 10): Promise<RecentSearch[]> => {
    return apiClient.get(`/api/search/recent?limit=${limit}`);
  },

  saveSearch: async (searchQuery: SearchQuery, name: string): Promise<{ message: string; id: string; name: string }> => {
    return apiClient.post('/api/search/save', {
      search_query: searchQuery,
      name: name
    });
  },

  getSavedSearches: async (): Promise<SavedSearch[]> => {
    return apiClient.get('/api/search/saved');
  },

  getSearchSuggestions: async (query: string, limit: number = 5): Promise<SearchSuggestion[]> => {
    return apiClient.get(`/api/search/suggestions?query=${encodeURIComponent(query)}&limit=${limit}`);
  },

  getAvailableFilters: async (): Promise<AvailableFilters> => {
    return apiClient.get('/api/search/filters');
  },
};