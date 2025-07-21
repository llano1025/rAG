import { useState } from 'react';

interface SearchFiltersProps {
  filters: {
    file_type: string[];
    date_range: { start: string; end: string } | null;
    owner: string;
  };
  onFiltersChange: (filters: any) => void;
}

const FILE_TYPES = [
  { value: 'pdf', label: 'PDF' },
  { value: 'docx', label: 'Word Document' },
  { value: 'txt', label: 'Text File' },
  { value: 'html', label: 'HTML' },
  { value: 'png', label: 'PNG Image' },
  { value: 'jpg', label: 'JPEG Image' },
];

export default function SearchFilters({ filters, onFiltersChange }: SearchFiltersProps) {
  const [dateRange, setDateRange] = useState({
    start: filters.date_range?.start || '',
    end: filters.date_range?.end || '',
  });

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

  const clearFilters = () => {
    setDateRange({ start: '', end: '' });
    onFiltersChange({
      file_type: [],
      date_range: null,
      owner: '',
    });
  };

  const hasActiveFilters = filters.file_type.length > 0 || filters.date_range || filters.owner;

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
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-3">File Type</h4>
        <div className="grid grid-cols-2 gap-2">
          {FILE_TYPES.map((type) => (
            <label key={type.value} className="flex items-center">
              <input
                type="checkbox"
                checked={filters.file_type.includes(type.value)}
                onChange={(e) => handleFileTypeChange(type.value, e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-600">{type.label}</span>
            </label>
          ))}
        </div>
      </div>

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
                {FILE_TYPES.find(t => t.value === type)?.label}
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
          </div>
        </div>
      )}
    </div>
  );
}