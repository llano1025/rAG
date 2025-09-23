// frontend/src/components/chat/ChatInterface.tsx

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon,
  StopIcon,
  DocumentIcon,
  CpuChipIcon,
  AdjustmentsHorizontalIcon,
  ClipboardDocumentListIcon,
  TrashIcon,
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  XMarkIcon,
  PlusIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../../hooks/useAuth';
import { apiClient } from '../../api/client';
import { chatApi } from '../../api/chat';
import { modelsApi, LoadedModel } from '../../api/models';
import ModelHealthIndicator from '../models/ModelHealthIndicator';
import RerankerModelSelector from '../models/RerankerModelSelector';
import EmbeddingModelSelector from '../models/EmbeddingModelSelector';
import TagInput from '../common/TagInput';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  model?: string;
  sources?: DocumentSource[];
  isStreaming?: boolean;
}

interface DocumentSource {
  document_id: number;
  filename: string;
  chunk_id: string;
  similarity_score: number;
  text_snippet: string;
}

interface ChatSettings {
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

interface ChatSessionInfo {
  session_id: string;
  created_at: Date;
  message_count: number;
}

const DEFAULT_SETTINGS: ChatSettings = {
  llm_model: 'openai-gpt35',
  embedding_model: 'hf-minilm-l6-v2',
  temperature: 1,
  max_tokens: 2048,
  use_rag: true,
  search_type: 'contextual',
  enable_reranking: false,
  rerank_score_weight: 0.5,
  // Search filter defaults
  file_type: [],
  date_range: null,
  tags: [],
  tag_match_mode: 'any',
  exclude_tags: [],
  languages: [],
  file_size_range: null,
  language: '',
  is_public: undefined,
  max_results: 20,
  min_score: 0.1,
};

const ChatInterface: React.FC = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [sessionInfo, setSessionInfo] = useState<ChatSessionInfo | null>(null);
  const [availableModels, setAvailableModels] = useState<{
    llm_models: LoadedModel[];
    embedding_models: LoadedModel[];
  }>({ llm_models: [], embedding_models: [] });
  const [modelsLoading, setModelsLoading] = useState(false);

  // Filter panel state
  const [showContentFilters, setShowContentFilters] = useState(false);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [showAdvancedFiltersSection, setShowAdvancedFiltersSection] = useState(false);
  const [showSearchParameters, setShowSearchParameters] = useState(false);
  const [fileSizeRange, setFileSizeRange] = useState({
    min: 0,
    max: 0,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Load available models function
  const loadAvailableModels = async () => {
    try {
      setModelsLoading(true);
      
      // Use chatApi.getAvailableModels() which includes enhanced filtering
      const availableModels = await chatApi.getAvailableModels();
      
      setAvailableModels({
        llm_models: availableModels.llm_models,
        embedding_models: availableModels.embedding_models
      });

      // Update default settings if current models are not available
      if (availableModels.llm_models.length > 0 && !availableModels.llm_models.find(m => m.model_id === settings.llm_model)) {
        setSettings(prev => ({ ...prev, llm_model: availableModels.llm_models[0].model_id }));
      }
      
      if (availableModels.embedding_models.length > 0 && !availableModels.embedding_models.find(m => m.model_id === settings.embedding_model)) {
        setSettings(prev => ({ ...prev, embedding_model: availableModels.embedding_models[0].model_id }));
      }
        
    } catch (error) {
      console.error('Failed to load available models:', error);
      // The chatApi.getAvailableModels() already includes fallback logic
      setAvailableModels({
        llm_models: [],
        embedding_models: []
      });
    } finally {
      setModelsLoading(false);
    }
  };

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
  }, []);

  // Create new chat session
  const createNewSession = async () => {
    try {
      const session: any = await apiClient.post('/api/chat/sessions', {
        settings: settings
      });
      
      setSessionInfo({
        session_id: session.session_id,
        created_at: new Date(session.created_at),
        message_count: 0
      });
      setMessages([]);
    } catch (error) {
      console.error('Failed to create new session:', error);
    }
  };

  // Send message
  const sendMessage = async () => {
    console.log('[CHAT_UI] Send message called - Input:', inputText.trim(), 'Streaming:', isStreaming);
    
    if (!inputText.trim() || isStreaming) {
      console.log('[CHAT_UI] Send message aborted - Empty input or already streaming');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    console.log('[CHAT_UI] User message created:', userMessage);

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsStreaming(true);
    setIsLoading(true);
    
    console.log('[CHAT_UI] UI state updated - Streaming: true, Loading: true');

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      model: settings.llm_model,
      isStreaming: true
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Create abort controller for streaming
      abortControllerRef.current = new AbortController();

      // Use the chat API for streaming
      console.log('[CHAT_UI] Starting stream request:', {
        message: userMessage.content,
        sessionId: sessionInfo?.session_id,
        settings: settings
      });

      // Debug logging for embedding model specifically
      console.log('[CHAT_UI] Embedding model setting:', {
        embedding_model: settings.embedding_model,
        use_rag: settings.use_rag,
        search_type: settings.search_type,
        max_results: settings.max_results
      });

      const stream = await chatApi.streamMessage(
        userMessage.content,
        sessionInfo?.session_id,
        settings
      );

      console.log('[CHAT_UI] Stream object received:', stream);

      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let chunkCount = 0;

      setIsLoading(false);
      console.log('[CHAT_UI] Starting to read stream chunks');

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          chunkCount++;
          
          console.log(`[CHAT_UI] Chunk ${chunkCount} - Done: ${done}, Value length: ${value?.length || 0}`);
          
          if (done) {
            console.log(`[CHAT_UI] Stream completed after ${chunkCount} chunks`);
            break;
          }

          const chunk = decoder.decode(value);
          console.log(`[CHAT_UI] Decoded chunk ${chunkCount}:`, chunk.slice(0, 200) + (chunk.length > 200 ? '...' : ''));
          
          const lines = chunk.split('\n');
          console.log(`[CHAT_UI] Chunk ${chunkCount} split into ${lines.length} lines`);

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const jsonData = line.slice(6);
                console.log(`[CHAT_UI] Parsing JSON data:`, jsonData);
                const data = JSON.parse(jsonData);
                console.log(`[CHAT_UI] Parsed data type:`, data.type, data);
                
                if (data.type === 'content') {
                  accumulatedContent += data.content;
                  console.log(`[CHAT_UI] Content received - Length: ${data.content.length}, Total: ${accumulatedContent.length}`);
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  ));
                } else if (data.type === 'sources') {
                  console.log(`[CHAT_UI] Sources received:`, data.sources.length, 'sources');
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, sources: data.sources }
                      : msg
                  ));
                } else if (data.type === 'done') {
                  console.log(`[CHAT_UI] Stream completion signal received`);
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, isStreaming: false }
                      : msg
                  ));
                  
                  // Update session info
                  if (sessionInfo) {
                    setSessionInfo(prev => prev ? {
                      ...prev,
                      message_count: prev.message_count + 2
                    } : null);
                  }
                  break;
                } else if (data.type === 'error') {
                  console.error('[CHAT_UI] Error received from stream:', data.error);
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { 
                          ...msg, 
                          content: `Error: ${data.error}`,
                          isStreaming: false
                        }
                      : msg
                  ));
                  break;
                }
              } catch (parseError) {
                console.error('[CHAT_UI] Error parsing streaming data:', parseError, 'Line:', line);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              console.log(`[CHAT_UI] Non-data line received:`, line);
            }
          }
        }
      } else {
        console.error('[CHAT_UI] No reader available from stream');
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('[CHAT_UI] Request was aborted');
      } else {
        console.error('[CHAT_UI] Error sending message:', error);
        console.error('[CHAT_UI] Error stack:', error.stack);
        
        // Show error message
        setMessages(prev => prev.map(msg => 
          msg.id === assistantMessageId 
            ? { 
                ...msg, 
                content: 'Sorry, I encountered an error while processing your message. Please try again.',
                isStreaming: false
              }
            : msg
        ));
      }
    } finally {
      console.log('[CHAT_UI] Send message cleanup - Setting streaming: false, loading: false');
      setIsStreaming(false);
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  // Stop streaming
  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setIsLoading(false);
    }
  };

  // Clear chat
  const clearChat = () => {
    setMessages([]);
    setSessionInfo(null);
  };

  // Handle Enter key in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputText]);

  // Filter helper functions
  const handleTagsChange = (tags: string[]) => {
    setSettings(prev => ({ ...prev, tags }));
  };

  const handleExcludeTagsChange = (excludeTags: string[]) => {
    setSettings(prev => ({ ...prev, exclude_tags: excludeTags }));
  };

  const handleTagMatchModeChange = (mode: 'any' | 'all' | 'exact') => {
    setSettings(prev => ({ ...prev, tag_match_mode: mode }));
  };

  const handleFileSizeRangeChange = (field: 'min' | 'max', value: number) => {
    const newFileSizeRange = { ...fileSizeRange, [field]: value };
    setFileSizeRange(newFileSizeRange);

    const hasValidRange = newFileSizeRange.min >= 0 && newFileSizeRange.max > newFileSizeRange.min;
    setSettings(prev => ({
      ...prev,
      file_size_range: hasValidRange ? [newFileSizeRange.min, newFileSizeRange.max] : null,
    }));
  };

  const clearFilters = () => {
    setFileSizeRange({ min: 0, max: 0 });
    setSettings(prev => ({
      ...prev,
      file_type: [],
      date_range: null,
      tags: [],
      tag_match_mode: 'any',
      exclude_tags: [],
      languages: [],
      file_size_range: null,
      language: '',
      is_public: undefined,
    }));
  };

  const hasActiveFilters = settings.file_type.length > 0 ||
    settings.tags.length > 0 ||
    settings.exclude_tags.length > 0 ||
    settings.file_size_range ||
    settings.language ||
    settings.is_public !== undefined ||
    settings.languages.length > 0;

  // Auto-expand sections if filters are active
  useEffect(() => {
    if (settings.max_results !== 20 || settings.min_score !== 0.1) {
      setShowSearchParameters(true);
    }
    if (settings.embedding_model || settings.enable_reranking || settings.reranker_model) {
      setShowContentFilters(true);
    }
    if (settings.tags.length > 0 || settings.exclude_tags.length > 0 || settings.language) {
      setShowAdvancedFilters(true);
    }
    if (settings.languages.length > 0 || settings.file_size_range || settings.is_public !== undefined) {
      setShowAdvancedFiltersSection(true);
    }
  }, [settings]);

  const inputClassName = "w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 h-9";

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <CpuChipIcon className="h-6 w-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">
              RAG Chat
            </h2>
          </div>
          
          {sessionInfo && (
            <div className="text-sm text-gray-500">
              {sessionInfo.message_count} messages
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-gray-700"
            title="Chat Settings"
          >
            <AdjustmentsHorizontalIcon className="h-5 w-5" />
          </button>
          
          <button
            onClick={clearChat}
            className="p-2 text-gray-500 hover:text-red-600"
            title="Clear Chat"
          >
            <TrashIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="p-4 bg-gray-50 border-b border-gray-200 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">Chat Settings</h3>
            <div className="flex items-center space-x-2">
              {hasActiveFilters && (
                <button
                  onClick={clearFilters}
                  className="text-xs text-blue-600 hover:text-blue-500"
                >
                  Clear Filters
                </button>
              )}
              <button
                onClick={loadAvailableModels}
                disabled={modelsLoading}
                className="flex items-center px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                <ArrowPathIcon className={`h-4 w-4 mr-1 ${modelsLoading ? 'animate-spin' : ''}`} />
                Refresh Models
              </button>
            </div>
          </div>

          {/* Chat Configuration - Always visible */}
          <div className="bg-gray-50 rounded p-3">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Chat Configuration</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  LLM Model
                </label>
                <div className="relative">
                  <select
                    value={settings.llm_model}
                    onChange={(e) => setSettings(prev => ({ ...prev, llm_model: e.target.value }))}
                    className={inputClassName}
                  >
                    {availableModels.llm_models.map(model => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.display_name} ({model.provider})
                        {model.context_window ? ` - ${(model.context_window / 1000).toFixed(0)}K context` : ''}
                      </option>
                    ))}
                  </select>
                  {availableModels.llm_models.length === 0 && (
                    <div className="absolute inset-y-0 right-3 flex items-center">
                      <ModelHealthIndicator status="not_loaded" size="sm" />
                    </div>
                  )}
                </div>
                {availableModels.llm_models.length === 0 && (
                  <p className="text-xs text-red-600 mt-1">
                    No LLM models are currently loaded. Contact admin to register and load models.
                  </p>
                )}
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Search Type
                </label>
                <select
                  value={settings.search_type}
                  onChange={(e) => setSettings(prev => ({ ...prev, search_type: e.target.value as any }))}
                  className={inputClassName}
                >
                  <option value="semantic">Semantic</option>
                  <option value="contextual">Contextual</option>
                  <option value="keyword">Keyword</option>
                </select>
              </div>
            </div>

            <div className="mt-3 space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Temperature: {settings.temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={settings.use_rag}
                    onChange={(e) => setSettings(prev => ({ ...prev, use_rag: e.target.checked }))}
                    className="rounded border-gray-300"
                  />
                  <span className="text-xs font-medium text-gray-700">
                    Use RAG
                  </span>
                </label>
              </div>
            </div>
          </div>

          {/* Search Parameters - Expandable */}
          {settings.use_rag && (
            <div className="border border-gray-200 rounded">
              <button
                onClick={() => setShowSearchParameters(!showSearchParameters)}
                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
              >
                <span className="text-sm font-medium text-gray-900">Search Parameters</span>
                {showSearchParameters ? (
                  <ChevronUpIcon className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4 text-gray-500" />
                )}
              </button>

              {showSearchParameters && (
                <div className="px-3 pb-3 border-t border-gray-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Max Results
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={settings.max_results}
                        onChange={(e) => setSettings(prev => ({ ...prev, max_results: Math.min(100, Math.max(1, parseInt(e.target.value) || 10)) }))}
                        className={inputClassName}
                        placeholder="20"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Min Score
                      </label>
                      <input
                        type="number"
                        min="0.0"
                        max="1.0"
                        step="0.1"
                        value={settings.min_score}
                        onChange={(e) => setSettings(prev => ({ ...prev, min_score: Math.min(1.0, Math.max(0.0, parseFloat(e.target.value) || 0.0)) }))}
                        className={inputClassName}
                        placeholder="0.1"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* AI & Model Options - Expandable */}
          {settings.use_rag && (
            <div className="border border-gray-200 rounded">
              <button
                onClick={() => setShowContentFilters(!showContentFilters)}
                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
              >
                <span className="text-sm font-medium text-gray-900">AI & Model Options</span>
                {showContentFilters ? (
                  <ChevronUpIcon className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4 text-gray-500" />
                )}
              </button>

              {showContentFilters && (
                <div className="px-3 pb-3 space-y-3 border-t border-gray-200">
                  {/* Embedding Model & Reranker Settings Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Embedding Model</label>
                      <EmbeddingModelSelector
                        selectedModel={settings.embedding_model}
                        onModelChange={(model) => setSettings(prev => ({ ...prev, embedding_model: model }))}
                        className=""
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-2">Reranker Settings</label>
                      <RerankerModelSelector
                        selectedModel={settings.reranker_model}
                        onModelChange={(model) => setSettings(prev => ({ ...prev, reranker_model: model }))}
                        enabled={settings.enable_reranking ?? true}
                        onEnabledChange={(enabled) => setSettings(prev => ({ ...prev, enable_reranking: enabled }))}
                        scoreWeight={settings.rerank_score_weight ?? 0.5}
                        onScoreWeightChange={(weight) => setSettings(prev => ({ ...prev, rerank_score_weight: weight }))}
                        minScore={settings.min_rerank_score}
                        onMinScoreChange={(score) => setSettings(prev => ({ ...prev, min_rerank_score: score }))}
                        compact={true}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Content Filters - Collapsible */}
          {settings.use_rag && (
            <div className="border border-gray-200 rounded">
              <button
                onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
              >
                <span className="text-sm font-medium text-gray-900">Content Filters</span>
                {showAdvancedFilters ? (
                  <ChevronUpIcon className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4 text-gray-500" />
                )}
              </button>

              {showAdvancedFilters && (
                <div className="px-3 pb-3 space-y-3 border-t border-gray-200">
                  {/* File Types */}
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-2">File Types</label>
                    <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                      {['pdf', 'docx', 'txt', 'html', 'png', 'jpg'].map((fileType) => (
                        <label key={fileType} className="flex items-center space-x-1">
                          <input
                            type="checkbox"
                            checked={settings.file_type.includes(fileType)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSettings(prev => ({ ...prev, file_type: [...prev.file_type, fileType] }));
                              } else {
                                setSettings(prev => ({ ...prev, file_type: prev.file_type.filter(ft => ft !== fileType) }));
                              }
                            }}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-xs text-gray-700">{fileType.toUpperCase()}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Tags Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Include Tags</label>
                      <TagInput
                        value={settings.tags}
                        onChange={handleTagsChange}
                        placeholder="Add tags..."
                        className="text-sm"
                        maxTags={3}
                      />
                      <select
                        value={settings.tag_match_mode}
                        onChange={(e) => handleTagMatchModeChange(e.target.value as 'any' | 'all' | 'exact')}
                        className="mt-1 px-2 py-1 text-xs border border-gray-300 rounded h-7"
                      >
                        <option value="any">ANY</option>
                        <option value="all">ALL</option>
                        <option value="exact">EXACT</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-red-600 mb-1">Exclude Tags</label>
                      <TagInput
                        value={settings.exclude_tags}
                        onChange={handleExcludeTagsChange}
                        placeholder="Exclude..."
                        className="text-sm"
                        maxTags={3}
                      />
                    </div>
                  </div>

                  {/* Language */}
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Language</label>
                    <select
                      value={settings.language}
                      onChange={(e) => setSettings(prev => ({ ...prev, language: e.target.value }))}
                      className={inputClassName}
                    >
                      <option value="">Any Language</option>
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                      <option value="zh">Chinese</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Advanced Filters - Collapsible */}
          {settings.use_rag && (
            <div className="border border-gray-200 rounded">
              <button
                onClick={() => setShowAdvancedFiltersSection(!showAdvancedFiltersSection)}
                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
              >
                <span className="text-sm font-medium text-gray-900">Advanced Options</span>
                {showAdvancedFiltersSection ? (
                  <ChevronUpIcon className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4 text-gray-500" />
                )}
              </button>

              {showAdvancedFiltersSection && (
                <div className="px-3 pb-3 border-t border-gray-200">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
                    {/* File Size */}
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">File Size (KB)</label>
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="number"
                          placeholder="Min"
                          value={fileSizeRange.min || ''}
                          onChange={(e) => handleFileSizeRangeChange('min', parseInt(e.target.value) || 0)}
                          className={inputClassName}
                        />
                        <input
                          type="number"
                          placeholder="Max"
                          value={fileSizeRange.max || ''}
                          onChange={(e) => handleFileSizeRangeChange('max', parseInt(e.target.value) || 0)}
                          className={inputClassName}
                        />
                      </div>
                    </div>

                    {/* Visibility */}
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Visibility</label>
                      <select
                        value={settings.is_public === undefined ? '' : settings.is_public.toString()}
                        onChange={(e) => {
                          const value = e.target.value;
                          setSettings(prev => ({ ...prev, is_public: value === '' ? undefined : value === 'true' }));
                        }}
                        className={inputClassName}
                      >
                        <option value="">All Documents</option>
                        <option value="true">Public Only</option>
                        <option value="false">Private Only</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500  mt-8">
            <CpuChipIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg mb-2">Welcome to RAG Chat!</p>
            <p className="text-sm">
              Ask questions about your documents and I&apos;ll provide answers using AI and search.
            </p>
            {!sessionInfo && (
              <button
                onClick={createNewSession}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Start New Chat
              </button>
            )}
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl px-4 py-2 rounded-lg ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              
              {message.isStreaming && (
                <div className="flex items-center mt-2 text-sm opacity-70">
                  <div className="animate-pulse flex space-x-1">
                    <div className="h-2 w-2 bg-current rounded-full"></div>
                    <div className="h-2 w-2 bg-current rounded-full"></div>
                    <div className="h-2 w-2 bg-current rounded-full"></div>
                  </div>
                  <span className="ml-2">Thinking...</span>
                </div>
              )}

              {message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <div className="flex items-center text-sm text-gray-600 mb-2">
                    <DocumentIcon className="h-4 w-4 mr-1" />
                    Sources ({message.sources.length})
                  </div>
                  <div className="space-y-2">
                    {message.sources.map((source, index) => (
                      <div
                        key={index}
                        className="text-xs bg-white p-2 rounded border"
                      >
                        <div className="font-medium">{source.filename}</div>
                        <div className="text-gray-600 mt-1">
                          {source.text_snippet}
                        </div>
                        <div className="text-gray-500 mt-1">
                          Similarity: {(source.similarity_score * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="text-xs opacity-70 mt-2">
                {message.timestamp.toLocaleTimeString()}
                {message.model && (
                  <span className="ml-2">• {message.model}</span>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-end space-x-2">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="w-full p-3 border border-gray-300 rounded-lg resize-none max-h-32"
              rows={1}
              disabled={isStreaming}
            />
          </div>
          
          <div className="flex space-x-2">
            {isStreaming ? (
              <button
                onClick={stopStreaming}
                className="p-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                title="Stop"
              >
                <StopIcon className="h-5 w-5" />
              </button>
            ) : (
              <button
                onClick={sendMessage}
                disabled={!inputText.trim() || isLoading}
                className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                title="Send"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>
        
        {isLoading && (
          <div className="flex items-center mt-2 text-sm text-gray-500">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
            Connecting...
          </div>
        )}
      </div>

    </div>
  );
};

export default ChatInterface;