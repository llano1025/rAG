import { apiClient } from './client';

// Types
export interface DiscoveredModel {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  context_window?: number;
  max_tokens?: number;
  supports_streaming: boolean;
  supports_embeddings: boolean;
  is_available: boolean;
  metadata?: Record<string, any>;
}

export interface RegisteredModel {
  id: number;
  name: string;
  display_name?: string;
  description?: string;
  model_name: string;
  provider: string;
  is_active: boolean;
  is_public: boolean;
  usage_count: number;
  success_rate: number;
  average_response_time?: number;
  created_at: string;
  updated_at: string;
}

export interface Provider {
  name: string;
  display_name: string;
  description: string;
  available: boolean;
  supports_discovery: boolean;
  requires_api_key: boolean;
  supported_models: string[];
}

export interface ModelTemplate {
  model_name: string;
  max_tokens: number;
  temperature: number;
  top_p: number;
  presence_penalty: number;
  frequency_penalty: number;
  context_window?: number;
  api_base?: string;
  api_key?: string;
  stop_sequences?: string[];
  repeat_penalty?: number;
  top_k?: number;
  provider_config?: Record<string, any>;
}

export interface ModelRegistrationRequest {
  name: string;
  display_name?: string;
  description?: string;
  model_name: string;
  provider: string;
  config: Record<string, any>;
  provider_config?: Record<string, any>;
  is_public?: boolean;
  fallback_priority?: number;
}

export interface ModelUpdateRequest {
  name?: string;
  display_name?: string;
  description?: string;
  config?: Record<string, any>;
  provider_config?: Record<string, any>;
  is_public?: boolean;
  is_active?: boolean;
  fallback_priority?: number;
}

export interface ModelDiscoveryRequest {
  provider: string;
  api_key?: string;
  base_url?: string;
}

export interface ModelTestRequest {
  test_type: 'connectivity' | 'generation' | 'embedding';
  test_prompt?: string;
  timeout_seconds?: number;
}

export interface ModelTest {
  id: number;
  test_type: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'timeout';
  response_time_ms?: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface LoadedModel {
  model_id: string;
  model_name: string;
  provider: string;
  display_name: string;
  description?: string;
  context_window?: number;
  max_tokens?: number;
  supports_streaming: boolean;
  supports_embeddings: boolean;
  capabilities: string[];
}

// API Client
export const modelsApi = {
  // Provider and Discovery endpoints
  async getProviders(): Promise<{ providers: Provider[]; total: number }> {
    return apiClient.get('/api/models/providers');
  },

  async discoverModels(provider: string, request?: ModelDiscoveryRequest): Promise<{
    provider: string;
    models: DiscoveredModel[];
    total: number;
  }> {
    const url = `/api/models/discover/${provider}`;
    if (request) {
      return apiClient.post(url, request);
    }
    return apiClient.post(url);
  },

  async getModelTemplates(provider?: string): Promise<{
    templates: Record<string, Record<string, ModelTemplate>>;
    providers?: string[];
  }> {
    const params = provider ? { provider } : {};
    return apiClient.get('/api/models/templates', { params });
  },

  async testProviderConnectivity(provider: string, request?: ModelDiscoveryRequest): Promise<{
    provider: string;
    connected: boolean;
    status_code?: number;
    base_url?: string;
    message: string;
    error?: string;
  }> {
    const url = `/api/models/test-connectivity/${provider}`;
    if (request) {
      return apiClient.post(url, request);
    }
    return apiClient.post(url);
  },

  // Model Registration endpoints
  async registerModel(request: ModelRegistrationRequest): Promise<RegisteredModel> {
    return apiClient.post('/api/models/register', request);
  },

  async getRegisteredModels(params?: {
    provider?: string;
    active_only?: boolean;
    include_public?: boolean;
  }): Promise<RegisteredModel[]> {
    return apiClient.get('/api/models/registered', { params });
  },

  async getRegisteredModel(modelId: number): Promise<{
    id: number;
    name: string;
    display_name?: string;
    description?: string;
    model_name: string;
    provider: string;
    config: Record<string, any>;
    provider_config?: Record<string, any>;
    is_active: boolean;
    is_public: boolean;
    fallback_priority?: number;
    usage_count: number;
    success_rate: number;
    average_response_time?: number;
    total_tokens_used: number;
    estimated_cost: number;
    last_used?: string;
    created_at: string;
    updated_at: string;
    owner?: { id: number; username: string };
  }> {
    return apiClient.get(`/api/models/registered/${modelId}`);
  },

  async updateRegisteredModel(modelId: number, request: ModelUpdateRequest): Promise<RegisteredModel> {
    return apiClient.put(`/api/models/registered/${modelId}`, request);
  },

  async deleteRegisteredModel(modelId: number): Promise<{ message: string }> {
    return apiClient.delete(`/api/models/registered/${modelId}`);
  },

  // Model Loading and Management endpoints
  async loadRegisteredModels(userId?: number): Promise<{
    message: string;
    loaded_count: number;
    user_id?: number;
  }> {
    const params = userId ? { user_id: userId } : {};
    return apiClient.post('/api/models/load-registered', null, { params });
  },

  async syncRegisteredModels(): Promise<{
    message: string;
    statistics: {
      loaded: number;
      unloaded: number;
      reloaded: number;
      errors: number;
    };
  }> {
    return apiClient.post('/api/models/sync');
  },

  async getLoadedModels(): Promise<{
    models: LoadedModel[];
    total: number;
  }> {
    return apiClient.get('/api/models/loaded');
  },

  async reloadModel(modelId: number): Promise<{ message: string }> {
    return apiClient.post(`/api/models/registered/${modelId}/reload`);
  },

  // Model Testing endpoints
  async testModel(modelId: number, request: ModelTestRequest): Promise<ModelTest> {
    return apiClient.post(`/api/models/registered/${modelId}/test`, request);
  },

  async getModelTests(modelId: number, limit?: number): Promise<ModelTest[]> {
    const params = limit ? { limit } : {};
    return apiClient.get(`/api/models/registered/${modelId}/tests`, { params });
  },
};