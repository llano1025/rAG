import { apiClient } from './client';
import { SearchQuery, SearchResponse } from '@/types';

export const searchApi = {
  search: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/search', query);
  },

  semanticSearch: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/search/semantic', query);
  },

  hybridSearch: async (query: SearchQuery): Promise<SearchResponse> => {
    return apiClient.post('/search/hybrid', query);
  },

  getSearchHistory: async (): Promise<Array<{ query: string; timestamp: string }>> => {
    return apiClient.get('/search/history');
  },

  saveSearch: async (query: string, name: string): Promise<void> => {
    return apiClient.post('/search/save', { query, name });
  },

  getSavedSearches: async (): Promise<Array<{ id: string; name: string; query: string }>> => {
    return apiClient.get('/search/saved');
  },
};