import { useState, useEffect, useRef, useMemo } from 'react';
import { XMarkIcon, ChevronDownIcon, TagIcon, PlusIcon } from '@heroicons/react/24/outline';
import { CheckIcon } from '@heroicons/react/20/solid';
import { libraryApi, Tag } from '@/api/library';
import toast from 'react-hot-toast';

interface TagSelectorProps {
  value: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  className?: string;
  maxTags?: number;
  mode?: 'input' | 'dropdown' | 'multi-select';
  onBulkApply?: (tags: string[], documentIds: number[]) => Promise<void>;
  selectedDocuments?: number[];
  showBulkActions?: boolean;
  disabled?: boolean;
}

export default function TagSelector({ 
  value = [], 
  onChange, 
  placeholder = "Select tags...", 
  className = "",
  maxTags = 20,
  mode = 'dropdown',
  onBulkApply,
  selectedDocuments = [],
  showBulkActions = false,
  disabled = false
}: TagSelectorProps) {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState<Tag[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [allTags, setAllTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load existing tags
  useEffect(() => {
    const loadTags = async () => {
      try {
        setLoading(true);
        const tags = await libraryApi.getTags();
        if (Array.isArray(tags)) {
          setAllTags(tags);
        } else {
          console.warn('Invalid tags response format:', tags);
          setAllTags([]);
        }
      } catch (error) {
        console.error('Failed to load tags:', error);
        setAllTags([]);
        if (process.env.NODE_ENV === 'development') {
          toast.error('Failed to load tag suggestions');
        }
      } finally {
        setLoading(false);
      }
    };
    loadTags();
  }, []);

  // Filter suggestions
  const filteredTags = useMemo(() => {
    if (!inputValue.trim()) return allTags;
    
    return allTags.filter(tag => 
      tag?.tag_name && 
      typeof tag.tag_name === 'string' &&
      tag.tag_name.toLowerCase().includes(inputValue.toLowerCase())
    );
  }, [inputValue, allTags]);

  // Check for exact match
  const exactMatch = filteredTags.find(tag => 
    tag.tag_name.toLowerCase() === inputValue.trim().toLowerCase()
  );

  const addTag = async (tagName: string) => {
    const trimmedTag = tagName.trim().toLowerCase();
    if (!trimmedTag || value.includes(trimmedTag) || value.length >= maxTags) {
      return;
    }

    // Add to local state immediately
    onChange([...value, trimmedTag]);
    setInputValue('');
    
    // Create tag if it doesn't exist
    if (!exactMatch && inputValue.trim()) {
      try {
        setCreating(true);
        await libraryApi.createTag({
          tag_name: trimmedTag,
          description: `Auto-created tag: ${trimmedTag}`
        });
        
        // Refresh tags list
        const updatedTags = await libraryApi.getTags();
        if (Array.isArray(updatedTags)) {
          setAllTags(updatedTags);
        }
        
        toast.success(`Created new tag: ${trimmedTag}`);
      } catch (error) {
        console.error('Failed to create tag:', error);
        // Don't remove from local state since user intended to add it
      } finally {
        setCreating(false);
      }
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(value.filter(tag => tag !== tagToRemove));
  };

  const toggleTag = (tagName: string) => {
    if (value.includes(tagName)) {
      removeTag(tagName);
    } else {
      addTag(tagName);
    }
  };

  const handleBulkAdd = async () => {
    if (!onBulkApply || selectedDocuments.length === 0 || value.length === 0) {
      return;
    }

    try {
      setLoading(true);
      await onBulkApply(value, selectedDocuments);
      toast.success(`Added tags to ${selectedDocuments.length} documents`);
    } catch (error) {
      console.error('Bulk apply failed:', error);
      toast.error('Failed to apply tags to documents');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (inputValue.trim()) {
        addTag(inputValue);
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setShowDropdown(true);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (mode === 'input') {
    // Simple input mode (similar to existing TagInput)
    return (
      <div className={`relative ${className}`} ref={dropdownRef}>
        <div className="flex flex-wrap items-center gap-1 p-2 border border-gray-300 rounded-md min-h-[42px] focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-primary-500">
          {value.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800"
            >
              {tag}
              {!disabled && (
                <button
                  type="button"
                  onClick={() => removeTag(tag)}
                  className="ml-1 hover:text-primary-600"
                >
                  <XMarkIcon className="h-3 w-3" />
                </button>
              )}
            </span>
          ))}
          
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setShowDropdown(true)}
            placeholder={value.length === 0 ? placeholder : ''}
            className="flex-1 min-w-[120px] outline-none bg-transparent"
            disabled={disabled || value.length >= maxTags}
          />
        </div>

        {showDropdown && filteredTags.length > 0 && (
          <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-48 overflow-y-auto">
            {filteredTags.map((tag) => (
              <button
                key={tag.tag_name}
                type="button"
                onClick={() => addTag(tag.tag_name)}
                className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center justify-between"
              >
                <span className="text-sm">{tag.tag_name}</span>
                <span className="text-xs text-gray-500">
                  {tag.document_count} docs
                </span>
              </button>
            ))}
            
            {inputValue.trim() && !exactMatch && (
              <button
                type="button"
                onClick={() => addTag(inputValue)}
                className="w-full px-3 py-2 text-left hover:bg-gray-50 border-t border-gray-200 flex items-center"
                disabled={creating}
              >
                <PlusIcon className="h-4 w-4 mr-2 text-primary-600" />
                <span className="text-sm text-primary-600">
                  Create "{inputValue.trim().toLowerCase()}"
                </span>
                {creating && <span className="ml-2 text-xs text-gray-500">Creating...</span>}
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  // Dropdown or multi-select mode
  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Selected tags display */}
      <div className="flex flex-wrap items-center gap-1 mb-2">
        {value.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800"
          >
            <TagIcon className="h-3 w-3 mr-1" />
            {tag}
            {!disabled && (
              <button
                type="button"
                onClick={() => removeTag(tag)}
                className="ml-1 hover:text-primary-600"
              >
                <XMarkIcon className="h-3 w-3" />
              </button>
            )}
          </span>
        ))}
      </div>

      {/* Dropdown trigger */}
      <button
        type="button"
        onClick={() => setShowDropdown(!showDropdown)}
        disabled={disabled || loading}
        className={`w-full flex items-center justify-between px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white text-sm text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
      >
        <span className="flex items-center">
          <TagIcon className="h-4 w-4 mr-2 text-gray-400" />
          {value.length > 0 ? `${value.length} tags selected` : placeholder}
        </span>
        <ChevronDownIcon className={`h-4 w-4 transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
      </button>

      {/* Search input */}
      {showDropdown && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg">
          <div className="p-2 border-b border-gray-200">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search or create tags..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              autoFocus
            />
          </div>

          {/* Tags list */}
          <div className="max-h-48 overflow-y-auto">
            {loading ? (
              <div className="px-3 py-4 text-center text-gray-500">
                Loading tags...
              </div>
            ) : filteredTags.length > 0 ? (
              filteredTags.map((tag) => (
                <button
                  key={tag.tag_name}
                  type="button"
                  onClick={() => toggleTag(tag.tag_name)}
                  className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center justify-between"
                >
                  <div className="flex items-center">
                    <div className={`w-4 h-4 mr-3 border rounded flex items-center justify-center ${
                      value.includes(tag.tag_name) 
                        ? 'bg-primary-600 border-primary-600' 
                        : 'border-gray-300'
                    }`}>
                      {value.includes(tag.tag_name) && (
                        <CheckIcon className="h-3 w-3 text-white" />
                      )}
                    </div>
                    <span className="text-sm">{tag.tag_name}</span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {tag.document_count} docs
                  </span>
                </button>
              ))
            ) : inputValue.trim() ? (
              <div className="px-3 py-2 text-gray-500 text-sm">
                No tags found matching "{inputValue}"
              </div>
            ) : (
              <div className="px-3 py-2 text-gray-500 text-sm">
                No tags available
              </div>
            )}

            {/* Create new tag option */}
            {inputValue.trim() && !exactMatch && (
              <button
                type="button"
                onClick={() => addTag(inputValue)}
                disabled={creating}
                className="w-full px-3 py-2 text-left hover:bg-gray-50 border-t border-gray-200 flex items-center"
              >
                <PlusIcon className="h-4 w-4 mr-3 text-primary-600" />
                <span className="text-sm text-primary-600">
                  Create "{inputValue.trim().toLowerCase()}"
                </span>
                {creating && <span className="ml-2 text-xs text-gray-500">Creating...</span>}
              </button>
            )}
          </div>

          {/* Bulk actions */}
          {showBulkActions && selectedDocuments.length > 0 && value.length > 0 && (
            <div className="border-t border-gray-200 p-3">
              <button
                type="button"
                onClick={handleBulkAdd}
                disabled={loading}
                className="w-full px-3 py-2 bg-primary-600 text-white text-sm rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Applying...' : `Apply to ${selectedDocuments.length} documents`}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Tag limit indicator */}
      {value.length >= maxTags && (
        <p className="text-xs text-gray-500 mt-1">
          Maximum {maxTags} tags allowed
        </p>
      )}
    </div>
  );
}