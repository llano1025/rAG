// frontend/src/components/chat/ChatInterface.tsx

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  PaperAirplaneIcon, 
  StopIcon, 
  DocumentIcon,
  CpuChipIcon,
  AdjustmentsHorizontalIcon,
  ClipboardDocumentListIcon,
  TrashIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../../hooks/useAuth';
import { apiClient } from '../../api/client';
import { chatApi } from '../../api/chat';

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
  search_type: 'semantic' | 'hybrid' | 'basic';
  top_k_documents: number;
}

interface ChatSessionInfo {
  session_id: string;
  created_at: Date;
  message_count: number;
}

const DEFAULT_SETTINGS: ChatSettings = {
  llm_model: 'openai-gpt35',
  embedding_model: 'hf-minilm-l6-v2',
  temperature: 0.7,
  max_tokens: 2048,
  use_rag: true,
  search_type: 'semantic',
  top_k_documents: 5
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
    llm_models: Array<{ id: string; name: string; provider: string }>;
    embedding_models: Array<{ id: string; name: string; provider: string }>;
  }>({ llm_models: [], embedding_models: [] });

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

  // Load available models on component mount
  useEffect(() => {
    const loadAvailableModels = async () => {
      try {
        const models: any = await apiClient.get('/api/chat/models');
        setAvailableModels(models);
      } catch (error) {
        console.error('Failed to load available models:', error);
      }
    };

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

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <CpuChipIcon className="h-6 w-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              RAG Chat
            </h2>
          </div>
          
          {sessionInfo && (
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {sessionInfo.message_count} messages
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            title="Chat Settings"
          >
            <AdjustmentsHorizontalIcon className="h-5 w-5" />
          </button>
          
          <button
            onClick={clearChat}
            className="p-2 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
            title="Clear Chat"
          >
            <TrashIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                LLM Model
              </label>
              <select
                value={settings.llm_model}
                onChange={(e) => setSettings(prev => ({ ...prev, llm_model: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-md dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              >
                {availableModels.llm_models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.display_name} ({model.provider})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Embedding Model
              </label>
              <select
                value={settings.embedding_model}
                onChange={(e) => setSettings(prev => ({ ...prev, embedding_model: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-md dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              >
                {availableModels.embedding_models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.display_name} ({model.provider})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Temperature: {settings.temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.temperature}
                onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Search Type
              </label>
              <select
                value={settings.search_type}
                onChange={(e) => setSettings(prev => ({ ...prev, search_type: e.target.value as any }))}
                className="w-full p-2 border border-gray-300 rounded-md dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              >
                <option value="semantic">Semantic</option>
                <option value="hybrid">Hybrid</option>
                <option value="basic">Basic</option>
              </select>
            </div>

            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.use_rag}
                  onChange={(e) => setSettings(prev => ({ ...prev, use_rag: e.target.checked }))}
                  className="rounded border-gray-300"
                />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Use RAG
                </span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
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
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
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
                <div className="mt-3 pt-3 border-t border-gray-300 dark:border-gray-600">
                  <div className="flex items-center text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <DocumentIcon className="h-4 w-4 mr-1" />
                    Sources ({message.sources.length})
                  </div>
                  <div className="space-y-2">
                    {message.sources.map((source, index) => (
                      <div
                        key={index}
                        className="text-xs bg-white dark:bg-gray-700 p-2 rounded border"
                      >
                        <div className="font-medium">{source.filename}</div>
                        <div className="text-gray-600 dark:text-gray-400 mt-1">
                          {source.text_snippet}
                        </div>
                        <div className="text-gray-500 dark:text-gray-500 mt-1">
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
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-end space-x-2">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="w-full p-3 border border-gray-300 rounded-lg resize-none max-h-32 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
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
          <div className="flex items-center mt-2 text-sm text-gray-500 dark:text-gray-400">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
            Connecting...
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;