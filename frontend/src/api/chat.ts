// frontend/src/api/chat.ts

import { apiClient } from './client';
import { modelsApi, LoadedModel } from './models';
import { filterLLMModels, filterEmbeddingModels } from '../utils/modelFilters';
import { RerankerModel } from '@/types';
import Cookies from 'js-cookie';

export interface ChatSettings {
  llm_model: string;
  embedding_model: string;
  temperature: number;
  max_tokens: number;
  use_rag: boolean;
  search_type: 'semantic' | 'contextual' | 'keyword';
  // Reranker settings
  enable_reranking?: boolean;
  reranker_model?: string;
  rerank_score_weight?: number;
  min_rerank_score?: number;
  // Search filter settings
  file_type: string[];
  date_range: { start: string; end: string } | null;
  tags: string[];
  tag_match_mode: 'any' | 'all' | 'exact';
  exclude_tags: string[];
  languages: string[];
  file_size_range: [number, number] | null;
  language: string;
  is_public?: boolean;
  max_results: number;
  min_score: number;
}

export interface ChatSession {
  session_id: string;
  created_at: string;
  settings: ChatSettings;
  message_count?: number;
  last_activity?: string;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  model?: string;
  sources?: DocumentSource[];
}

export interface DocumentSource {
  document_id: number;
  filename: string;
  chunk_id: string;
  similarity_score: number;
  text_snippet: string;
}

export interface AvailableModels {
  llm_models: LoadedModel[];
  embedding_models: LoadedModel[];
  reranker_models?: RerankerModel[];
}

export const chatApi = {
  // Get available models
  async getAvailableModels(): Promise<AvailableModels> {
    try {
      // Try the simple chat models API first (this works like document upload)
      const chatModels: any = await apiClient.get('/api/chat/models');
      return {
        llm_models: chatModels.llm_models || [],
        embedding_models: chatModels.embedding_models || []
      };
    } catch (error) {
      console.warn('Failed to load chat models, trying registered models API:', error);
      
      // Fallback to complex registered models API
      try {
        const loadedModelsData = await modelsApi.getLoadedModels();
        
        return {
          llm_models: filterLLMModels(loadedModelsData.models),
          embedding_models: filterEmbeddingModels(loadedModelsData.models)
        };
      } catch (fallbackError) {
        console.error('Both model APIs failed:', fallbackError);
        return {
          llm_models: [],
          embedding_models: []
        };
      }
    }
  },

  // Create a new chat session
  async createSession(settings?: Partial<ChatSettings>): Promise<ChatSession> {
    return apiClient.post('/api/chat/sessions', {
      settings: settings || {}
    });
  },

  // List user's chat sessions
  async listSessions(): Promise<{ sessions: ChatSession[]; total_sessions: number }> {
    return apiClient.get('/api/chat/sessions');
  },

  // Get chat history for a session
  async getChatHistory(
    sessionId: string,
    limit: number = 50
  ): Promise<{
    session_id: string;
    messages: ChatMessage[];
    total_messages: number;
    created_at: string;
    last_activity: string;
    settings: ChatSettings;
  }> {
    return apiClient.get(`/api/chat/sessions/${sessionId}`, {
      params: { limit }
    });
  },

  // Delete a chat session
  async deleteSession(sessionId: string): Promise<{ session_id: string; deleted: boolean }> {
    return apiClient.delete(`/api/chat/sessions/${sessionId}`);
  },

  // Send a message and get streaming response
  async streamMessage(
    message: string,
    sessionId?: string,
    settings?: ChatSettings
  ): Promise<ReadableStream> {
    const token = Cookies.get('access_token');
    const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${baseURL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        settings
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.body as ReadableStream;
  },

  // Send a message and get complete response (non-streaming)
  async sendMessage(
    message: string,
    sessionId?: string,
    settings?: ChatSettings
  ): Promise<{
    session_id: string;
    response: string;
    sources: DocumentSource[];
    timestamp: string;
  }> {
    return apiClient.post('/api/chat/message', {
      message,
      session_id: sessionId,
      settings
    });
  },

  // Run health check on models
  async runHealthCheck(): Promise<{
    health_check_results: any;
    timestamp: string;
  }> {
    return apiClient.post('/api/chat/models/health-check');
  },

  // Get embedding model recommendations
  async getEmbeddingRecommendations(
    useCase: string = 'general',
    maxResults: number = 3
  ): Promise<{
    use_case: string;
    recommendations: Array<{
      model_id: string;
      display_name: string;
      provider: string;
      description: string;
      quality_score?: number;
      performance_tier?: string;
    }>;
  }> {
    return apiClient.get('/api/chat/models/embedding/recommendations', {
      params: { use_case: useCase, max_results: maxResults }
    });
  },

  // Get available reranker models
  async getAvailableRerankerModels(): Promise<RerankerModel[]> {
    return apiClient.get('/api/search/reranker/models');
  },

  // Admin: Clean up old sessions
  async cleanupSessions(maxAgeHours: number = 24): Promise<{
    message: string;
    timestamp: string;
  }> {
    return apiClient.post('/api/chat/admin/cleanup-sessions', null, {
      params: { max_age_hours: maxAgeHours }
    });
  }
};

export default chatApi;