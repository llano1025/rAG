import { useState, useEffect, useRef } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { libraryApi, Tag } from '@/api/library';
import toast from 'react-hot-toast';

interface TagInputProps {
  value: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  className?: string;
  maxTags?: number;
}

export default function TagInput({ 
  value = [], 
  onChange, 
  placeholder = "Add tags...", 
  className = "",
  maxTags = 20 
}: TagInputProps) {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState<Tag[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [allTags, setAllTags] = useState<Tag[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load existing tags for suggestions
  useEffect(() => {
    const loadTags = async () => {
      try {
        const tags = await libraryApi.getTags();
        if (Array.isArray(tags)) {
          setAllTags(tags);
        } else {
          console.warn('Invalid tags response format:', tags);
          setAllTags([]);
        }
      } catch (error) {
        console.error('Failed to load tags:', error);
        // Don't show error to user, just silently use empty tags
        setAllTags([]);
        // Optional: Show a subtle warning in development
        if (process.env.NODE_ENV === 'development') {
          toast.error('Failed to load tag suggestions');
        }
      }
    };
    loadTags();
  }, []);

  // Filter suggestions based on input
  useEffect(() => {
    if (inputValue.trim()) {
      try {
        const filtered = allTags.filter(tag => 
          tag && 
          tag.tag_name && 
          typeof tag.tag_name === 'string' &&
          tag.tag_name.toLowerCase().includes(inputValue.toLowerCase()) &&
          !value.includes(tag.tag_name.toLowerCase())
        );
        setSuggestions(filtered);
        setShowSuggestions(filtered.length > 0);
      } catch (error) {
        console.error('Error filtering tag suggestions:', error);
        setSuggestions([]);
        setShowSuggestions(false);
      }
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [inputValue, allTags, value]);

  const addTag = (tag: string) => {
    const trimmedTag = tag.trim().toLowerCase();
    if (trimmedTag && !value.includes(trimmedTag) && value.length < maxTags) {
      onChange([...value, trimmedTag]);
      setInputValue('');
      setShowSuggestions(false);
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(value.filter(tag => tag !== tagToRemove));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      if (inputValue.trim()) {
        addTag(inputValue);
      }
    } else if (e.key === 'Backspace' && !inputValue && value.length > 0) {
      removeTag(value[value.length - 1]);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (tag: Tag) => {
    addTag(tag.tag_name);
  };

  return (
    <div className={`relative ${className}`}>
      <div className="flex flex-wrap items-center gap-1 p-2 border border-gray-300 rounded-md min-h-[42px] focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-primary-500">
        {/* Existing tags */}
        {value.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-primary-100 text-primary-800"
          >
            {tag}
            <button
              type="button"
              onClick={() => removeTag(tag)}
              className="ml-1 hover:text-primary-600"
            >
              <XMarkIcon className="h-3 w-3" />
            </button>
          </span>
        ))}
        
        {/* Input field */}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => inputValue.trim() && setShowSuggestions(suggestions.length > 0)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          placeholder={value.length === 0 ? placeholder : ''}
          className="flex-1 min-w-[120px] outline-none bg-transparent"
          disabled={value.length >= maxTags}
        />
      </div>

      {/* Tag limit indicator */}
      {value.length >= maxTags && (
        <p className="text-xs text-gray-500 mt-1">
          Maximum {maxTags} tags allowed
        </p>
      )}

      {/* Suggestions dropdown */}
      {showSuggestions && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-48 overflow-y-auto">
          {suggestions.map((tag) => (
            <button
              key={tag.tag_name}
              type="button"
              onClick={() => handleSuggestionClick(tag)}
              className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-center justify-between"
            >
              <span className="text-sm">{tag.tag_name}</span>
              <span className="text-xs text-gray-500">
                {tag.document_count} doc{tag.document_count !== 1 ? 's' : ''}
              </span>
            </button>
          ))}
          
          {/* Create new tag option */}
          {inputValue.trim() && !suggestions.some(s => s.tag_name === inputValue.trim().toLowerCase()) && (
            <button
              type="button"
              onClick={() => addTag(inputValue)}
              className="w-full px-3 py-2 text-left hover:bg-gray-50 border-t border-gray-200"
            >
              <span className="text-sm text-primary-600">
                Create tag: "{inputValue.trim().toLowerCase()}"
              </span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}