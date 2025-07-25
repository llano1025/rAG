// frontend/src/components/settings/ModelSelector.tsx

import React, { useState, useEffect } from 'react';
import {
  CpuChipIcon,
  CloudIcon,
  ServerIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { apiClient } from '../../api/client';

interface ModelInfo {
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

interface ModelSelectorProps {
  selectedLLMModel: string;
  selectedEmbeddingModel: string;
  onLLMModelChange: (modelId: string) => void;
  onEmbeddingModelChange: (modelId: string) => void;
  showAdvanced?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedLLMModel,
  selectedEmbeddingModel,
  onLLMModelChange,
  onEmbeddingModelChange,
  showAdvanced = false
}) => {
  const [llmModels, setLLMModels] = useState<ModelInfo[]>([]);
  const [embeddingModels, setEmbeddingModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showLLMDetails, setShowLLMDetails] = useState(false);
  const [showEmbeddingDetails, setShowEmbeddingDetails] = useState(false);
  const [healthCheckInProgress, setHealthCheckInProgress] = useState(false);

  // Load available models
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const data: any = await apiClient.get('/api/chat/models');
      setLLMModels(data.llm_models || []);
      setEmbeddingModels(data.embedding_models || []);
    } catch (err: any) {
      setError(err.message);
      console.error('Error loading models:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const runHealthCheck = async () => {
    try {
      setHealthCheckInProgress(true);
      
      await apiClient.post('/api/chat/models/health-check');
      // Reload models to get updated health status
      await loadModels();
    } catch (err) {
      console.error('Health check failed:', err);
    } finally {
      setHealthCheckInProgress(false);
    }
  };

  const getProviderIcon = (provider: string) => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return <CloudIcon className="h-5 w-5 text-green-600" />;
      case 'ollama':
        return <ServerIcon className="h-5 w-5 text-blue-600" />;
      case 'huggingface':
        return <CpuChipIcon className="h-5 w-5 text-orange-600" />;
      case 'gemini':
        return <CloudIcon className="h-5 w-5 text-purple-600" />;
      default:
        return <CpuChipIcon className="h-5 w-5 text-gray-600" />;
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="h-4 w-4 text-green-600" />;
      case 'unhealthy':
        return <ExclamationTriangleIcon className="h-4 w-4 text-red-600" />;
      default:
        return <InformationCircleIcon className="h-4 w-4 text-gray-400" />;
    }
  };

  const getPerformanceTierColor = (tier?: string) => {
    switch (tier) {
      case 'fast':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'balanced':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'quality':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const formatFileSize = (sizeInMB?: number) => {
    if (!sizeInMB) return 'Unknown';
    if (sizeInMB < 1024) return `${sizeInMB} MB`;
    return `${(sizeInMB / 1024).toFixed(1)} GB`;
  };

  const formatCost = (cost?: number) => {
    if (!cost) return 'Free';
    return `$${cost.toFixed(6)} per 1K tokens`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-400">Loading models...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
        <div className="flex items-center">
          <ExclamationTriangleIcon className="h-5 w-5 text-red-600 mr-2" />
          <span className="text-red-800 dark:text-red-200">Error loading models: {error}</span>
        </div>
        <button
          onClick={loadModels}
          className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Model Selection
        </h3>
        
        <button
          onClick={runHealthCheck}
          disabled={healthCheckInProgress}
          className="flex items-center px-3 py-1 text-sm bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-4 w-4 mr-1 ${healthCheckInProgress ? 'animate-spin' : ''}`} />
          Health Check
        </button>
      </div>

      {/* LLM Model Selection */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Language Model (LLM)
          </label>
          <button
            onClick={() => setShowLLMDetails(!showLLMDetails)}
            className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400"
          >
            {showLLMDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        <div className="grid grid-cols-1 gap-3">
          {llmModels.map((model) => (
            <div
              key={model.id}
              className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                selectedLLMModel === model.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
              onClick={() => onLLMModelChange(model.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getProviderIcon(model.provider)}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {model.display_name || model.name}
                      </span>
                      {getStatusIcon(model.status)}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {model.provider} • {model.description}
                    </div>
                  </div>
                </div>

                {selectedLLMModel === model.id && (
                  <CheckCircleIcon className="h-5 w-5 text-blue-600" />
                )}
              </div>

              {showLLMDetails && (
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600 space-y-2">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {model.quality_score && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Quality Score:</span>
                        <span className="ml-2 font-medium">
                          {(model.quality_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                    
                    {model.api_cost_per_1k_tokens !== undefined && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Cost:</span>
                        <span className="ml-2 font-medium">
                          {formatCost(model.api_cost_per_1k_tokens)}
                        </span>
                      </div>
                    )}
                  </div>

                  {model.use_cases && model.use_cases.length > 0 && (
                    <div>
                      <span className="text-gray-500 dark:text-gray-400 text-sm">Use Cases:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {model.use_cases.map((useCase, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs rounded"
                          >
                            {useCase}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {model.last_used && (
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Last used: {new Date(model.last_used).toLocaleDateString()}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Embedding Model Selection */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Embedding Model
          </label>
          <button
            onClick={() => setShowEmbeddingDetails(!showEmbeddingDetails)}
            className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400"
          >
            {showEmbeddingDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        <div className="grid grid-cols-1 gap-3">
          {embeddingModels.map((model) => (
            <div
              key={model.id}
              className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                selectedEmbeddingModel === model.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
              onClick={() => onEmbeddingModelChange(model.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getProviderIcon(model.provider)}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {model.display_name || model.name}
                      </span>
                      {getStatusIcon(model.status)}
                      {model.performance_tier && (
                        <span className={`px-2 py-1 text-xs rounded-full ${getPerformanceTierColor(model.performance_tier)}`}>
                          {model.performance_tier}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {model.provider} • {model.embedding_dimension}D • {model.description}
                    </div>
                  </div>
                </div>

                {selectedEmbeddingModel === model.id && (
                  <CheckCircleIcon className="h-5 w-5 text-blue-600" />
                )}
              </div>

              {showEmbeddingDetails && (
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600 space-y-2">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Dimensions:</span>
                      <span className="ml-2 font-medium">{model.embedding_dimension}</span>
                    </div>
                    
                    {model.quality_score && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Quality Score:</span>
                        <span className="ml-2 font-medium">
                          {(model.quality_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                    
                    {model.model_size_mb && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Model Size:</span>
                        <span className="ml-2 font-medium">
                          {formatFileSize(model.model_size_mb)}
                        </span>
                      </div>
                    )}
                    
                    {model.memory_requirements_mb && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Memory:</span>
                        <span className="ml-2 font-medium">
                          {formatFileSize(model.memory_requirements_mb)}
                        </span>
                      </div>
                    )}
                  </div>

                  {model.language_support && model.language_support.length > 0 && (
                    <div>
                      <span className="text-gray-500 dark:text-gray-400 text-sm">Languages:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {model.language_support.map((lang, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs rounded"
                          >
                            {lang}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {model.use_cases && model.use_cases.length > 0 && (
                    <div>
                      <span className="text-gray-500 dark:text-gray-400 text-sm">Use Cases:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {model.use_cases.map((useCase, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs rounded"
                          >
                            {useCase}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                    <div>
                      {model.gpu_required ? 'GPU Required' : 'CPU Compatible'}
                    </div>
                    {model.last_used && (
                      <div>Last used: {new Date(model.last_used).toLocaleDateString()}</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Migration Notice */}
      {showAdvanced && (
        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-start">
            <InformationCircleIcon className="h-5 w-5 text-yellow-600 mr-2 mt-0.5" />
            <div className="text-sm">
              <p className="text-yellow-800 dark:text-yellow-200 font-medium">
                Model Migration
              </p>
              <p className="text-yellow-700 dark:text-yellow-300 mt-1">
                Changing the embedding model will require re-indexing all documents. 
                This process may take some time depending on your document count.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;