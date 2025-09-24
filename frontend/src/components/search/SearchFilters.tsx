import { useState, useEffect } from 'react';
import { searchApi, AvailableFilters } from '@/api/search';
import TagInput from '@/components/common/TagInput';
import EmbeddingModelSelector from '../models/EmbeddingModelSelector';
import RerankerModelSelector from '../models/RerankerModelSelector';
import MMRSettingsSelector from '../models/MMRSettingsSelector';
import {
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from '@heroicons/react/24/outline';

interface SearchFiltersProps {
  filters: {
    file_type: string[];
    date_range: { start: string; end: string } | null;
    owner: string;
    tags: string[];
    tag_match_mode: 'any' | 'all' | 'exact';
    exclude_tags: string[];
    folder_ids?: string[];
    languages?: string[];
    file_size_range?: [number, number] | null;
    language?: string;
    is_public?: boolean;
    embedding_model?: string;
  };
  onFiltersChange: (filters: any) => void;
  // Reranker props
  selectedEmbeddingModel?: string;
  onEmbeddingModelChange: (modelId: string) => void;
  rerankerEnabled: boolean;
  onRerankerEnabledChange: (enabled: boolean) => void;
  rerankerModel?: string;
  onRerankerModelChange: (model: string | undefined) => void;
  rerankerScoreWeight: number;
  onRerankerScoreWeightChange: (weight: number) => void;
  rerankerMinScore?: number;
  onRerankerMinScoreChange: (score: number | undefined) => void;
  // MMR (Maximal Marginal Relevance) diversification props
  mmrEnabled: boolean;
  onMmrEnabledChange: (enabled: boolean) => void;
  mmrLambda: number;
  onMmrLambdaChange: (lambda: number) => void;
  mmrSimilarityThreshold: number;
  onMmrSimilarityThresholdChange: (threshold: number) => void;
  mmrMaxResults?: number;
  onMmrMaxResultsChange: (maxResults: number | undefined) => void;
  mmrSimilarityMetric: 'cosine' | 'euclidean' | 'dot_product';
  onMmrSimilarityMetricChange: (metric: 'cosine' | 'euclidean' | 'dot_product') => void;
  // Search parameters
  maxResults: number;
  onMaxResultsChange: (value: number) => void;
  minScore: number;
  onMinScoreChange: (value: number) => void;
}

export default function SearchFilters({
  filters,
  onFiltersChange,
  selectedEmbeddingModel,
  onEmbeddingModelChange,
  rerankerEnabled,
  onRerankerEnabledChange,
  rerankerModel,
  onRerankerModelChange,
  rerankerScoreWeight,
  onRerankerScoreWeightChange,
  rerankerMinScore,
  onRerankerMinScoreChange,
  mmrEnabled,
  onMmrEnabledChange,
  mmrLambda,
  onMmrLambdaChange,
  mmrSimilarityThreshold,
  onMmrSimilarityThresholdChange,
  mmrMaxResults,
  onMmrMaxResultsChange,
  mmrSimilarityMetric,
  onMmrSimilarityMetricChange,
  maxResults,
  onMaxResultsChange,
  minScore,
  onMinScoreChange
}: SearchFiltersProps) {
  const [fileSizeRange, setFileSizeRange] = useState({
    min: filters.file_size_range?.[0] || 0,
    max: filters.file_size_range?.[1] || 0,
  });
  const [availableFilters, setAvailableFilters] = useState<AvailableFilters | null>(null);
  const [loading, setLoading] = useState(true);
  const [showContentFilters, setShowContentFilters] = useState(false);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);

  useEffect(() => {
    const loadAvailableFilters = async () => {
      try {
        setLoading(true);
        const filters = await searchApi.getAvailableFilters();
        setAvailableFilters(filters);
      } catch (error) {
        console.error('Failed to load available filters:', error);
        setAvailableFilters({
          file_types: [
            { value: 'pdf', label: 'PDF' },
            { value: 'docx', label: 'Word' },
            { value: 'txt', label: 'Text' },
            { value: 'html', label: 'HTML' },
            { value: 'png', label: 'PNG' },
            { value: 'jpg', label: 'JPEG' },
          ],
          tags: [],
          languages: [],
          folders: [],
          date_range: {},
          file_size_range: { min_size: 0, max_size: 0, avg_size: 0 },
          search_types: []
        });
      } finally {
        setLoading(false);
      }
    };
    loadAvailableFilters();
  }, []);

  // Auto-expand sections if filters are active
  useEffect(() => {
    if (filters.tags.length > 0 || filters.exclude_tags.length > 0 || filters.language ||
        filters.embedding_model || selectedEmbeddingModel || rerankerEnabled || mmrEnabled) {
      setShowContentFilters(true);
    }
    if (filters.folder_ids?.length || filters.languages?.length || filters.file_size_range ||
        filters.is_public !== undefined) {
      setShowAdvancedFilters(true);
    }
  }, [filters, selectedEmbeddingModel, rerankerEnabled, mmrEnabled]);

  const handleFileTypeChange = (fileTypes: string[]) => {
    onFiltersChange({
      ...filters,
      file_type: fileTypes,
    });
  };

  const handleTagsChange = (tags: string[]) => {
    onFiltersChange({
      ...filters,
      tags,
    });
  };

  const handleTagMatchModeChange = (mode: 'any' | 'all' | 'exact') => {
    onFiltersChange({
      ...filters,
      tag_match_mode: mode,
    });
  };

  const handleExcludeTagsChange = (excludeTags: string[]) => {
    onFiltersChange({
      ...filters,
      exclude_tags: excludeTags,
    });
  };

  const handleFileSizeRangeChange = (field: 'min' | 'max', value: number) => {
    const newFileSizeRange = { ...fileSizeRange, [field]: value };
    setFileSizeRange(newFileSizeRange);

    const hasValidRange = newFileSizeRange.min >= 0 && newFileSizeRange.max > newFileSizeRange.min;
    onFiltersChange({
      ...filters,
      file_size_range: hasValidRange ? [newFileSizeRange.min, newFileSizeRange.max] : null,
    });
  };

  const handleLanguageChange = (language: string) => {
    onFiltersChange({
      ...filters,
      language: language || undefined,
    });
  };

  const handleIsPublicChange = (isPublic: boolean | undefined) => {
    onFiltersChange({
      ...filters,
      is_public: isPublic,
    });
  };

  const clearFilters = () => {
    setFileSizeRange({ min: 0, max: 0 });
    onFiltersChange({
      file_type: [],
      date_range: null,
      owner: '',
      tags: [],
      tag_match_mode: 'any',
      exclude_tags: [],
      folder_ids: [],
      languages: [],
      file_size_range: null,
      language: '',
      is_public: undefined,
      embedding_model: undefined,
    });
    // Reset model selections
    onEmbeddingModelChange('');
    onRerankerEnabledChange(false);
    onRerankerModelChange(undefined);
  };

  const hasActiveFilters = filters.file_type.length > 0 ||
    filters.tags.length > 0 ||
    filters.exclude_tags.length > 0 ||
    filters.file_size_range ||
    filters.language ||
    filters.is_public !== undefined ||
    filters.embedding_model ||
    selectedEmbeddingModel ||
    rerankerEnabled ||
    mmrEnabled ||
    (filters.folder_ids && filters.folder_ids.length > 0) ||
    (filters.languages && filters.languages.length > 0);

  const inputClassName = "w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 h-9";

  if (loading) {
    return (
      <div className="flex items-center space-x-2 text-sm text-gray-500 py-2">
        <ArrowPathIcon className="h-4 w-4 animate-spin" />
        <span>Loading filters...</span>
      </div>
    );
  }

  if (!availableFilters) {
    return (
      <div className="text-sm text-red-500 py-2">Failed to load filters</div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-900">Filters</h3>
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="text-xs text-blue-600 hover:text-blue-500"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Search Parameters - Top Section */}
      <div className="bg-gray-50 rounded p-3">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Search Parameters</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Max Results
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={maxResults}
              onChange={(e) => onMaxResultsChange(Math.min(100, Math.max(1, parseInt(e.target.value) || 10)))}
              className={inputClassName}
              placeholder="10"
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
              value={minScore}
              onChange={(e) => onMinScoreChange(Math.min(1.0, Math.max(0.0, parseFloat(e.target.value) || 0.0)))}
              className={inputClassName}
              placeholder="0.0"
            />
          </div>
        </div>
      </div>

      {/* Content & AI Filters - Expandable */}
      <div className="border border-gray-200 rounded">
        <button
          onClick={() => setShowContentFilters(!showContentFilters)}
          className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
        >
          <span className="text-sm font-medium text-gray-900">Content Options</span>
          {showContentFilters ? (
            <ChevronUpIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          )}
        </button>

        {showContentFilters && (
          <div className="px-3 pb-3 space-y-3 border-t border-gray-200">
            {/* Tags Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Include Tags</label>
                <TagInput
                  value={filters.tags}
                  onChange={handleTagsChange}
                  placeholder="Add tags..."
                  className="text-sm"
                  maxTags={3}
                />
                <select
                  value={filters.tag_match_mode}
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
                  value={filters.exclude_tags}
                  onChange={handleExcludeTagsChange}
                  placeholder="Exclude..."
                  className="text-sm"
                  maxTags={3}
                />
              </div>
            </div>

            {/* Language & Embedding Model Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Language</label>
                <select
                  value={filters.language || ''}
                  onChange={(e) => handleLanguageChange(e.target.value)}
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

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Embedding Model</label>
                <EmbeddingModelSelector
                  selectedModel={selectedEmbeddingModel}
                  onModelChange={onEmbeddingModelChange}
                  className=""
                />
              </div>
            </div>

            {/* Reranker & MMR Settings Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">Reranker Settings</label>
                <RerankerModelSelector
                  selectedModel={rerankerModel}
                  onModelChange={onRerankerModelChange}
                  enabled={rerankerEnabled}
                  onEnabledChange={onRerankerEnabledChange}
                  scoreWeight={rerankerScoreWeight}
                  onScoreWeightChange={onRerankerScoreWeightChange}
                  minScore={rerankerMinScore}
                  onMinScoreChange={onRerankerMinScoreChange}
                  compact={true}
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">Result Diversification</label>
                <MMRSettingsSelector
                  enabled={mmrEnabled}
                  onEnabledChange={onMmrEnabledChange}
                  lambda={mmrLambda}
                  onLambdaChange={onMmrLambdaChange}
                  similarityThreshold={mmrSimilarityThreshold}
                  onSimilarityThresholdChange={onMmrSimilarityThresholdChange}
                  maxResults={mmrMaxResults}
                  onMaxResultsChange={onMmrMaxResultsChange}
                  similarityMetric={mmrSimilarityMetric}
                  onSimilarityMetricChange={onMmrSimilarityMetricChange}
                  compact={true}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Advanced Filters - Expandable */}
      <div className="border border-gray-200 rounded">
        <button
          onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
          className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50"
        >
          <span className="text-sm font-medium text-gray-900">Advanced Options</span>
          {showAdvancedFilters ? (
            <ChevronUpIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          )}
        </button>

        {showAdvancedFilters && (
          <div className="px-3 pb-3 border-t border-gray-200">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
              {/* File Size & Visibility */}
              <div className="space-y-3">
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

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">Visibility</label>
                  <select
                    value={filters.is_public === undefined ? '' : filters.is_public.toString()}
                    onChange={(e) => {
                      const value = e.target.value;
                      handleIsPublicChange(value === '' ? undefined : value === 'true');
                    }}
                    className={inputClassName}
                  >
                    <option value="">All Documents</option>
                    <option value="true">Public Only</option>
                    <option value="false">Private Only</option>
                  </select>
                </div>
              </div>

              {/* Folders & Languages */}
              <div className="space-y-3">
                {availableFilters.folders.length > 0 && (
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Folders</label>
                    <select
                      multiple
                      value={filters.folder_ids || []}
                      onChange={(e) => {
                        const values = Array.from(e.target.selectedOptions, option => option.value);
                        onFiltersChange({
                          ...filters,
                          folder_ids: values,
                        });
                      }}
                      className={`${inputClassName} h-auto`}
                      size={3}
                    >
                      {availableFilters.folders.map((folder) => (
                        <option key={folder.value} value={folder.value}>
                          {folder.label} {folder.count && `(${folder.count})`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {availableFilters.languages.length > 0 && (
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Content Languages</label>
                    <select
                      multiple
                      value={filters.languages || []}
                      onChange={(e) => {
                        const values = Array.from(e.target.selectedOptions, option => option.value);
                        onFiltersChange({
                          ...filters,
                          languages: values,
                        });
                      }}
                      className={`${inputClassName} h-auto`}
                      size={3}
                    >
                      {availableFilters.languages.map((language) => (
                        <option key={language.value} value={language.value}>
                          {language.label} {language.count && `(${language.count})`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}