import { useState, useEffect } from 'react';
import { libraryApi, Tag } from '@/api/library';
import { searchApi, AvailableFilters, FilterOption } from '@/api/search';
import TagInput from '@/components/common/TagInput';

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
  };
  onFiltersChange: (filters: any) => void;
}

export default function SearchFilters({ filters, onFiltersChange }: SearchFiltersProps) {
  const [dateRange, setDateRange] = useState({
    start: filters.date_range?.start || '',
    end: filters.date_range?.end || '',
  });
  const [fileSizeRange, setFileSizeRange] = useState({
    min: filters.file_size_range?.[0] || 0,
    max: filters.file_size_range?.[1] || 0,
  });
  const [availableFilters, setAvailableFilters] = useState<AvailableFilters | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAvailableFilters = async () => {
      try {
        setLoading(true);
        const filters = await searchApi.getAvailableFilters();
        setAvailableFilters(filters);
      } catch (error) {
        console.error('Failed to load available filters:', error);
        // Fallback to default file types if API fails
        setAvailableFilters({
          file_types: [
            { value: 'pdf', label: 'PDF' },
            { value: 'docx', label: 'Word Document' },
            { value: 'txt', label: 'Text File' },
            { value: 'html', label: 'HTML' },
            { value: 'png', label: 'PNG Image' },
            { value: 'jpg', label: 'JPEG Image' },
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

  const handleFileTypeChange = (fileType: string, checked: boolean) => {
    const newFileTypes = checked
      ? [...filters.file_type, fileType]
      : filters.file_type.filter(type => type !== fileType);
    
    onFiltersChange({
      ...filters,
      file_type: newFileTypes,
    });
  };

  const handleDateRangeChange = (field: 'start' | 'end', value: string) => {
    const newDateRange = { ...dateRange, [field]: value };
    setDateRange(newDateRange);
    
    const hasValidRange = newDateRange.start && newDateRange.end;
    onFiltersChange({
      ...filters,
      date_range: hasValidRange ? newDateRange : null,
    });
  };

  const handleOwnerChange = (owner: string) => {
    onFiltersChange({
      ...filters,
      owner,
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
    setDateRange({ start: '', end: '' });
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
    });
  };

  const hasActiveFilters = filters.file_type.length > 0 || 
    filters.date_range || 
    filters.owner || 
    filters.tags.length > 0 || 
    filters.exclude_tags.length > 0 ||
    filters.file_size_range ||
    filters.language ||
    filters.is_public !== undefined ||
    (filters.folder_ids && filters.folder_ids.length > 0) || 
    (filters.languages && filters.languages.length > 0);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="text-sm text-gray-500">Loading filters...</div>
      </div>
    );
  }

  if (!availableFilters) {
    return (
      <div className="space-y-6">
        <div className="text-sm text-red-500">Failed to load filters</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-900">Search Filters</h3>
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="text-xs text-primary-600 hover:text-primary-500"
          >
            Clear all filters
          </button>
        )}
      </div>

      {/* File Type Filter */}
      {availableFilters.file_types.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">File Type</h4>
          <div className="grid grid-cols-2 gap-2">
            {availableFilters.file_types.map((type) => (
              <label key={type.value} className="flex items-center">
                <input
                  type="checkbox"
                  checked={filters.file_type.includes(type.value)}
                  onChange={(e) => handleFileTypeChange(type.value, e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-600">
                  {type.label}
                  {type.count && <span className="text-gray-400 ml-1">({type.count})</span>}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Date Range Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">Date Range</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">From</label>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => handleDateRangeChange('start', e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">To</label>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => handleDateRangeChange('end', e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            />
          </div>
        </div>
      </div>

      {/* Owner Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">Owner</h4>
        <input
          type="text"
          value={filters.owner}
          onChange={(e) => handleOwnerChange(e.target.value)}
          placeholder="Enter owner username"
          className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
        />
      </div>

      {/* Enhanced Tags Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">Tags</h4>
        
        {/* Tag Match Mode */}
        <div className="mb-4">
          <label className="block text-xs font-medium text-gray-600 mb-2">
            Match Mode
            <span className="ml-1 text-gray-400" title="Choose how to match the selected tags">ℹ️</span>
          </label>
          <div className="flex flex-wrap gap-3">
            <label className="flex items-center">
              <input
                type="radio"
                name="tagMatchMode"
                value="any"
                checked={filters.tag_match_mode === 'any'}
                onChange={() => handleTagMatchModeChange('any')}
                className="h-3 w-3 text-primary-600 focus:ring-primary-500 border-gray-300"
              />
              <span className="ml-1.5 text-xs text-gray-600">
                ANY <span className="text-gray-400">(OR)</span>
              </span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="tagMatchMode"
                value="all"
                checked={filters.tag_match_mode === 'all'}
                onChange={() => handleTagMatchModeChange('all')}
                className="h-3 w-3 text-primary-600 focus:ring-primary-500 border-gray-300"
              />
              <span className="ml-1.5 text-xs text-gray-600">
                ALL <span className="text-gray-400">(AND)</span>
              </span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="tagMatchMode"
                value="exact"
                checked={filters.tag_match_mode === 'exact'}
                onChange={() => handleTagMatchModeChange('exact')}
                className="h-3 w-3 text-primary-600 focus:ring-primary-500 border-gray-300"
              />
              <span className="ml-1.5 text-xs text-gray-600">
                EXACT
              </span>
            </label>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {filters.tag_match_mode === 'any' && 'Find documents that have ANY of the selected tags'}
            {filters.tag_match_mode === 'all' && 'Find documents that have ALL of the selected tags'}
            {filters.tag_match_mode === 'exact' && 'Find documents that have EXACTLY these tags (no more, no less)'}
          </p>
        </div>

        {/* Include Tags */}
        <div className="mb-4">
          <label className="block text-xs font-medium text-gray-600 mb-2">Include Tags</label>
          {availableFilters.tags.length > 0 && (
            <div className="max-h-32 overflow-y-auto space-y-1 mb-2">
              {availableFilters.tags.map((tag) => (
                <label key={tag.value} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.tags.includes(tag.value)}
                    onChange={(e) => {
                      const newTags = e.target.checked
                        ? [...filters.tags, tag.value]
                        : filters.tags.filter(t => t !== tag.value);
                      handleTagsChange(newTags);
                    }}
                    className="h-3 w-3 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-xs text-gray-600">
                    {tag.label}
                    {tag.count && <span className="text-gray-400 ml-1">({tag.count})</span>}
                  </span>
                </label>
              ))}
            </div>
          )}
          <TagInput
            value={filters.tags}
            onChange={handleTagsChange}
            placeholder="Add tags to include..."
            className="w-full"
            maxTags={10}
          />
        </div>

        {/* Exclude Tags */}
        <div>
          <label className="block text-xs font-medium text-red-600 mb-2">Exclude Tags</label>
          <TagInput
            value={filters.exclude_tags}
            onChange={handleExcludeTagsChange}
            placeholder="Add tags to exclude..."
            className="w-full"
            maxTags={5}
          />
          <p className="text-xs text-gray-400 mt-1">
            Documents with these tags will be excluded from results
          </p>
        </div>
      </div>

      {/* Folders Filter */}
      {availableFilters.folders.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Folders</h4>
          <div className="max-h-32 overflow-y-auto space-y-2">
            {availableFilters.folders.map((folder) => (
              <label key={folder.value} className="flex items-center">
                <input
                  type="checkbox"
                  checked={filters.folder_ids?.includes(folder.value) || false}
                  onChange={(e) => {
                    const newFolders = e.target.checked
                      ? [...(filters.folder_ids || []), folder.value]
                      : (filters.folder_ids || []).filter(f => f !== folder.value);
                    onFiltersChange({
                      ...filters,
                      folder_ids: newFolders,
                    });
                  }}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-600">
                  {folder.label}
                  {folder.count && <span className="text-gray-400 ml-1">({folder.count})</span>}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Languages Filter */}
      {availableFilters.languages.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Languages</h4>
          <div className="max-h-32 overflow-y-auto space-y-2">
            {availableFilters.languages.map((language) => (
              <label key={language.value} className="flex items-center">
                <input
                  type="checkbox"
                  checked={filters.languages?.includes(language.value) || false}
                  onChange={(e) => {
                    const newLanguages = e.target.checked
                      ? [...(filters.languages || []), language.value]
                      : (filters.languages || []).filter(l => l !== language.value);
                    onFiltersChange({
                      ...filters,
                      languages: newLanguages,
                    });
                  }}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-600">
                  {language.label}
                  {language.count && <span className="text-gray-400 ml-1">({language.count})</span>}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* File Size Range Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">File Size Range</h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Min Size (KB)</label>
            <input
              type="number"
              min="0"
              value={fileSizeRange.min || ''}
              onChange={(e) => handleFileSizeRangeChange('min', parseInt(e.target.value) || 0)}
              placeholder="0"
              className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Max Size (KB)</label>
            <input
              type="number"
              min="0"
              value={fileSizeRange.max || ''}
              onChange={(e) => handleFileSizeRangeChange('max', parseInt(e.target.value) || 0)}
              placeholder="No limit"
              className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            />
          </div>
        </div>
      </div>

      {/* Language Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">Document Language</h4>
        <select
          value={filters.language || ''}
          onChange={(e) => handleLanguageChange(e.target.value)}
          className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
        >
          <option value="">Any Language</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
          <option value="zh">Chinese</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="ru">Russian</option>
        </select>
      </div>

      {/* Public/Private Filter */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">Document Visibility</h4>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              name="visibility"
              checked={filters.is_public === undefined}
              onChange={() => handleIsPublicChange(undefined)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
            />
            <span className="ml-2 text-sm text-gray-600">All Documents</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="visibility"
              checked={filters.is_public === true}
              onChange={() => handleIsPublicChange(true)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
            />
            <span className="ml-2 text-sm text-gray-600">Public Only</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="visibility"
              checked={filters.is_public === false}
              onChange={() => handleIsPublicChange(false)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
            />
            <span className="ml-2 text-sm text-gray-600">Private Only</span>
          </label>
        </div>
      </div>

      {/* Active Filters Summary */}
      {hasActiveFilters && (
        <div className="pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Active Filters:</h4>
          <div className="flex flex-wrap gap-2">
            {filters.file_type.map((type) => (
              <span
                key={type}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
              >
                {availableFilters.file_types.find(t => t.value === type)?.label || type}
                <button
                  onClick={() => handleFileTypeChange(type, false)}
                  className="ml-1 text-primary-600 hover:text-primary-500"
                >
                  ×
                </button>
              </span>
            ))}
            {filters.date_range && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                {filters.date_range.start} to {filters.date_range.end}
                <button
                  onClick={() => {
                    setDateRange({ start: '', end: '' });
                    onFiltersChange({ ...filters, date_range: null });
                  }}
                  className="ml-1 text-primary-600 hover:text-primary-500"
                >
                  ×
                </button>
              </span>
            )}
            {filters.owner && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                Owner: {filters.owner}
                <button
                  onClick={() => handleOwnerChange('')}
                  className="ml-1 text-primary-600 hover:text-primary-500"
                >
                  ×
                </button>
              </span>
            )}
            {filters.tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
              >
                Include: {tag}
                <button
                  onClick={() => handleTagsChange(filters.tags.filter(t => t !== tag))}
                  className="ml-1 text-blue-600 hover:text-blue-500"
                >
                  ×
                </button>
              </span>
            ))}
            {filters.exclude_tags.map((tag) => (
              <span
                key={`exclude-${tag}`}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800"
              >
                Exclude: {tag}
                <button
                  onClick={() => handleExcludeTagsChange(filters.exclude_tags.filter(t => t !== tag))}
                  className="ml-1 text-red-600 hover:text-red-500"
                >
                  ×
                </button>
              </span>
            ))}
            {filters.tags.length > 0 && filters.tag_match_mode !== 'any' && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                Match: {filters.tag_match_mode.toUpperCase()}
              </span>
            )}
            {filters.file_size_range && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Size: {filters.file_size_range[0]} - {filters.file_size_range[1]} KB
                <button
                  onClick={() => {
                    setFileSizeRange({ min: 0, max: 0 });
                    onFiltersChange({ ...filters, file_size_range: null });
                  }}
                  className="ml-1 text-green-600 hover:text-green-500"
                >
                  ×
                </button>
              </span>
            )}
            {filters.language && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                Language: {filters.language}
                <button
                  onClick={() => handleLanguageChange('')}
                  className="ml-1 text-yellow-600 hover:text-yellow-500"
                >
                  ×
                </button>
              </span>
            )}
            {filters.is_public !== undefined && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                {filters.is_public ? 'Public' : 'Private'} Documents
                <button
                  onClick={() => handleIsPublicChange(undefined)}
                  className="ml-1 text-gray-600 hover:text-gray-500"
                >
                  ×
                </button>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}