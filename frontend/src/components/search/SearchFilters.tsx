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
    folder_ids?: string[];
    languages?: string[];
  };
  onFiltersChange: (filters: any) => void;
}

export default function SearchFilters({ filters, onFiltersChange }: SearchFiltersProps) {
  const [dateRange, setDateRange] = useState({
    start: filters.date_range?.start || '',
    end: filters.date_range?.end || '',
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

  const clearFilters = () => {
    setDateRange({ start: '', end: '' });
    onFiltersChange({
      file_type: [],
      date_range: null,
      owner: '',
      tags: [],
      folder_ids: [],
      languages: [],
    });
  };

  const hasActiveFilters = filters.file_type.length > 0 || filters.date_range || filters.owner || filters.tags.length > 0 || (filters.folder_ids && filters.folder_ids.length > 0) || (filters.languages && filters.languages.length > 0);

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

      {/* Tags Filter */}
      {availableFilters.tags.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Tags</h4>
          <div className="max-h-40 overflow-y-auto space-y-2">
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
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-600">
                  {tag.label}
                  {tag.count && <span className="text-gray-400 ml-1">({tag.count})</span>}
                </span>
              </label>
            ))}
          </div>
          <div className="mt-2">
            <TagInput
              value={filters.tags}
              onChange={handleTagsChange}
              placeholder="Add custom tags..."
              className="w-full"
              maxTags={10}
            />
            <p className="text-xs text-gray-500 mt-1">
              Select from available tags above or add custom tags.
            </p>
          </div>
        </div>
      )}

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
                Tag: {tag}
                <button
                  onClick={() => handleTagsChange(filters.tags.filter(t => t !== tag))}
                  className="ml-1 text-blue-600 hover:text-blue-500"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}