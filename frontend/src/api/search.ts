import { apiClient } from './client';
import { SearchQuery, SearchResponse } from '@/types';

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

  getSearchHistory: async (): Promise<Array<{ query: string; timestamp: string }>> => {
    return apiClient.get('/api/search/history');
  },

  saveSearch: async (query: string, name: string): Promise<void> => {
    return apiClient.post('/api/search/save', { query, name });
  },

  getSavedSearches: async (): Promise<Array<{ id: string; name: string; query: string }>> => {
    return apiClient.get('/api/search/saved');
  },
};