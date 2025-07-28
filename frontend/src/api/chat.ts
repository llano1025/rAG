// frontend/src/api/chat.ts

import { apiClient } from './client';
import Cookies from 'js-cookie';

export interface ChatSettings {
  llm_model: string;
  embedding_model: string;
  temperature: number;
  max_tokens: number;
  use_rag: boolean;
  search_type: 'semantic' | 'hybrid' | 'basic';
  top_k_documents: number;
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

export interface ModelInfo {
  id: string;
  name: string;
  display_name: string;
  provider: string;
  description: string;
  embedding_dimension?: number;
  performance_tier?: string;
  quality_score?: number;
  use_cases?: string[];
  language_support?: string[];
  api_cost_per_1k_tokens?: number;
  model_size_mb?: number;
  memory_requirements_mb?: number;
  gpu_required?: boolean;
  status?: 'healthy' | 'unhealthy' | 'unknown';
  last_used?: string;
}

export interface AvailableModels {
  llm_models: ModelInfo[];
  embedding_models: ModelInfo[];
}

export const chatApi = {
  // Get available models
  async getAvailableModels(): Promise<AvailableModels> {
    return apiClient.get('/api/chat/models');
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