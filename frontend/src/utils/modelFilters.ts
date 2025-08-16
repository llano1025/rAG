// frontend/src/utils/modelFilters.ts

import { LoadedModel, RegisteredModel } from '../api/models';

export interface ModelCapabilities {
  supportsVision: boolean;
  supportsEmbeddings: boolean;
  supportsStreaming: boolean;
  supportsChat: boolean;
  supportsCompletion: boolean;
}

/**
 * Analyze model capabilities based on model information
 */
export const analyzeModelCapabilities = (model: LoadedModel | RegisteredModel): ModelCapabilities => {
  const modelName = 'model_name' in model ? model.model_name : model.model_id;
  const provider = model.provider.toLowerCase();
  const capabilities = 'capabilities' in model ? model.capabilities : [];
  
  const isVisionModel = 
    modelName.toLowerCase().includes('vision') ||
    modelName.toLowerCase().includes('gpt-4o') ||
    modelName.toLowerCase().includes('claude-3') ||
    modelName.toLowerCase().includes('gemini') ||
    capabilities.some(cap => cap.toLowerCase().includes('vision'));

  // Enhanced embedding model detection
  const isEmbeddingByName = 
    modelName.toLowerCase().includes('embedding') ||
    modelName.toLowerCase().includes('text-embedding') ||
    modelName.toLowerCase().includes('sentence') ||
    modelName.toLowerCase().includes('bert') ||
    modelName.toLowerCase().includes('minilm') ||
    modelName.toLowerCase().includes('e5-') ||
    modelName.toLowerCase().includes('all-mpnet') ||
    modelName.toLowerCase().includes('bge-') ||
    modelName.toLowerCase().includes('gte-') ||
    modelName.toLowerCase().includes('instructor') ||
    modelName.toLowerCase().includes('nomic');

  // Provider-specific embedding model patterns
  const isEmbeddingByProvider = 
    (provider === 'openai' && modelName.toLowerCase().includes('ada')) ||
    (provider === 'huggingface' && (
      modelName.toLowerCase().includes('sentence-transformers') ||
      modelName.toLowerCase().includes('thenlper') ||
      modelName.toLowerCase().includes('baai')
    )) ||
    (provider === 'ollama' && modelName.toLowerCase().includes('embed'));

  const supportsEmbeddings = 
    'supports_embeddings' in model 
      ? model.supports_embeddings 
      : isEmbeddingByName ||
        isEmbeddingByProvider ||
        capabilities.some(cap => cap.toLowerCase().includes('embedding'));

  const supportsStreaming = 
    'supports_streaming' in model 
      ? model.supports_streaming 
      : !supportsEmbeddings; // Most non-embedding models support streaming

  return {
    supportsVision: isVisionModel,
    supportsEmbeddings,
    supportsStreaming,
    supportsChat: !supportsEmbeddings && supportsStreaming,
    supportsCompletion: !supportsEmbeddings
  };
};

/**
 * Filter models for chat/LLM usage
 */
export const filterLLMModels = (models: LoadedModel[]): LoadedModel[] => {
  return models.filter(model => {
    const capabilities = analyzeModelCapabilities(model);
    return capabilities.supportsChat || capabilities.supportsCompletion;
  });
};

/**
 * Filter models for embedding usage
 */
export const filterEmbeddingModels = (models: LoadedModel[]): LoadedModel[] => {
  return models.filter(model => {
    const capabilities = analyzeModelCapabilities(model);
    return capabilities.supportsEmbeddings;
  });
};

/**
 * Filter models for vision/OCR usage
 */
export const filterVisionModels = (models: LoadedModel[] | RegisteredModel[]): (LoadedModel | RegisteredModel)[] => {
  return models.filter(model => {
    const capabilities = analyzeModelCapabilities(model);
    return capabilities.supportsVision;
  });
};

/**
 * Get recommended models for specific use cases
 */
export const getRecommendedModels = (
  models: LoadedModel[], 
  useCase: 'chat' | 'embeddings' | 'vision' | 'code' | 'analysis'
): LoadedModel[] => {
  let filteredModels: LoadedModel[];
  
  switch (useCase) {
    case 'chat':
      filteredModels = filterLLMModels(models);
      break;
    case 'embeddings':
      filteredModels = filterEmbeddingModels(models);
      break;
    case 'vision':
      filteredModels = filterVisionModels(models) as LoadedModel[];
      break;
    case 'code':
      filteredModels = filterLLMModels(models).filter(model => 
        model.model_name.toLowerCase().includes('code') ||
        model.model_name.toLowerCase().includes('codex') ||
        model.description?.toLowerCase().includes('code')
      );
      break;
    case 'analysis':
      filteredModels = filterLLMModels(models).filter(model => 
        model.context_window && model.context_window > 50000
      );
      break;
    default:
      filteredModels = models;
  }

  // Sort by quality and performance
  return filteredModels.sort((a, b) => {
    // Prioritize models with larger context windows
    const aContext = a.context_window || 0;
    const bContext = b.context_window || 0;
    
    if (aContext !== bContext) {
      return bContext - aContext;
    }
    
    // Then by provider preference (OpenAI, Anthropic, Google, Others)
    const providerOrder = ['openai', 'anthropic', 'gemini', 'google'];
    const aProviderIndex = providerOrder.indexOf(a.provider.toLowerCase());
    const bProviderIndex = providerOrder.indexOf(b.provider.toLowerCase());
    
    if (aProviderIndex !== bProviderIndex) {
      return (aProviderIndex === -1 ? 999 : aProviderIndex) - 
             (bProviderIndex === -1 ? 999 : bProviderIndex);
    }
    
    // Finally by name
    return a.display_name.localeCompare(b.display_name);
  });
};

/**
 * Check if a model is suitable for a specific task
 */
export const isModelSuitableForTask = (
  model: LoadedModel | RegisteredModel, 
  task: string
): boolean => {
  const capabilities = analyzeModelCapabilities(model);
  const taskLower = task.toLowerCase();
  
  if (taskLower.includes('vision') || taskLower.includes('ocr') || taskLower.includes('image')) {
    return capabilities.supportsVision;
  }
  
  if (taskLower.includes('embedding') || taskLower.includes('search') || taskLower.includes('similarity')) {
    return capabilities.supportsEmbeddings;
  }
  
  if (taskLower.includes('chat') || taskLower.includes('conversation')) {
    return capabilities.supportsChat;
  }
  
  if (taskLower.includes('code') || taskLower.includes('programming')) {
    const modelName = 'model_name' in model ? model.model_name : model.model_id;
    return capabilities.supportsCompletion && 
           (modelName.toLowerCase().includes('code') || 
            modelName.toLowerCase().includes('codex'));
  }
  
  return capabilities.supportsCompletion;
};

/**
 * Get model performance tier based on context window and capabilities
 */
export const getModelPerformanceTier = (model: LoadedModel | RegisteredModel): 'fast' | 'balanced' | 'quality' => {
  const contextWindow = 'context_window' in model ? model.context_window : 
                       'model_name' in model && model.model_name.includes('gpt-4') ? 128000 : 4096;
  
  if (contextWindow && contextWindow > 100000) {
    return 'quality'; // High-capacity models
  } else if (contextWindow && contextWindow > 32000) {
    return 'balanced'; // Medium-capacity models
  } else {
    return 'fast'; // Fast, lower-capacity models
  }
};

/**
 * Estimate model cost per 1K tokens (rough estimates)
 */
export const estimateModelCost = (model: LoadedModel | RegisteredModel): number => {
  const modelName = 'model_name' in model ? model.model_name : model.model_id;
  const provider = model.provider.toLowerCase();
  
  // These are rough estimates for common models
  if (provider === 'openai') {
    if (modelName.includes('gpt-4o')) return 0.005;
    if (modelName.includes('gpt-4')) return 0.03;
    if (modelName.includes('gpt-3.5')) return 0.002;
    if (modelName.includes('embedding')) return 0.0001;
  } else if (provider === 'anthropic') {
    if (modelName.includes('claude-3')) return 0.015;
    if (modelName.includes('claude-2')) return 0.01;
  } else if (provider === 'google' || provider === 'gemini') {
    if (modelName.includes('gemini-pro')) return 0.002;
    if (modelName.includes('gemini-ultra')) return 0.01;
  } else if (provider === 'ollama' || provider === 'lmstudio') {
    return 0; // Local models are free
  }
  
  return 0.002; // Default estimate
};

/**
 * Debug function to analyze why a model was or wasn't categorized
 */
export const debugModelCategorization = (model: LoadedModel | RegisteredModel): string => {
  const modelName = 'model_name' in model ? model.model_name : model.model_id;
  const provider = model.provider.toLowerCase();
  const capabilities = 'capabilities' in model ? model.capabilities : [];
  const analysis = analyzeModelCapabilities(model);
  
  return `Model: ${modelName}
Provider: ${provider}
Capabilities: ${capabilities.join(', ')}
Analysis: ${JSON.stringify(analysis, null, 2)}
Explicit supports_embeddings: ${'supports_embeddings' in model ? model.supports_embeddings : 'N/A'}
Explicit supports_streaming: ${'supports_streaming' in model ? model.supports_streaming : 'N/A'}`;
};

export default {
  analyzeModelCapabilities,
  filterLLMModels,
  filterEmbeddingModels,
  filterVisionModels,
  getRecommendedModels,
  isModelSuitableForTask,
  getModelPerformanceTier,
  estimateModelCost,
  debugModelCategorization
};