import { useState, useEffect, useCallback } from 'react';
import { useForm } from 'react-hook-form';
import {
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  ClockIcon,
  BookmarkIcon,
} from '@heroicons/react/24/outline';
import { searchApi, SearchSuggestion, RecentSearch, SavedSearch } from '@/api/search';
import { SearchQuery, SearchResponse, SearchFilters as SearchFiltersType } from '@/types';
import SearchResults from './SearchResults';
import SearchFilters from './SearchFilters';
import toast from 'react-hot-toast';

interface SearchForm {
  query: string;
  searchType: 'basic' | 'semantic' | 'hybrid';
}

export default function SearchInterface() {
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [searchHistory, setSearchHistory] = useState<RecentSearch[]>([]);
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestionLoading, setSuggestionLoading] = useState(false);
  const [filters, setFilters] = useState({
    file_type: [] as string[],
    date_range: null as { start: string; end: string } | null,
    owner: '',
    tags: [] as string[],
    folder_ids: [] as string[],
    languages: [] as string[],
  });

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<SearchForm>({
    defaultValues: {
      query: '',
      searchType: 'hybrid',
    }
  });

  const searchType = watch('searchType');

  useEffect(() => {
    fetchSearchHistory();
    fetchSavedSearches();
  }, []);

  const fetchSearchHistory = async () => {
    try {
      const history = await searchApi.getRecentSearches(10);
      setSearchHistory(history);
    } catch (error) {
      // Silently fail for search history
    }
  };

  const fetchSavedSearches = async () => {
    try {
      const saved = await searchApi.getSavedSearches();
      setSavedSearches(saved);
    } catch (error) {
      // Silently fail for saved searches
    }
  };

  // Debounced search suggestions
  const fetchSuggestions = useCallback(async (query: string) => {
    if (!query.trim() || query.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    setSuggestionLoading(true);
    try {
      const suggestionsData = await searchApi.getSearchSuggestions(query, 5);
      setSuggestions(suggestionsData);
      setShowSuggestions(true);
    } catch (error) {
      setSuggestions([]);
      setShowSuggestions(false);
    } finally {
      setSuggestionLoading(false);
    }
  }, []);

  // Debounce suggestions
  useEffect(() => {
    const timer = setTimeout(() => {
      const currentQuery = watch('query');
      if (currentQuery) {
        fetchSuggestions(currentQuery);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [watch('query'), fetchSuggestions]);

  const onSearch = async (data: SearchForm) => {
    if (!data.query.trim()) return;

    setLoading(true);
    try {
      // Build search filters in backend format
      const searchFilters: SearchFiltersType = {};
      
      if (filters.file_type.length > 0) {
        searchFilters.file_types = filters.file_type;
      }
      
      if (filters.tags.length > 0) {
        searchFilters.tag_ids = filters.tags;
      }

      if (filters.folder_ids && filters.folder_ids.length > 0) {
        searchFilters.folder_ids = filters.folder_ids;
      }
      
      if (filters.date_range) {
        searchFilters.date_range = [filters.date_range.start, filters.date_range.end];
      }
      
      if (filters.owner) {
        // Handle owner filter through metadata_filters
        searchFilters.metadata_filters = { owner: filters.owner };
      }

      if (filters.languages && filters.languages.length > 0) {
        searchFilters.metadata_filters = {
          ...searchFilters.metadata_filters,
          languages: filters.languages
        };
      }

      const searchQuery: SearchQuery = {
        query: data.query,
        filters: Object.keys(searchFilters).length > 0 ? searchFilters : undefined,
        page: 1,
        page_size: 20,
        top_k: 20,
        similarity_threshold: 0.0,
        semantic_search: data.searchType === 'semantic',
        hybrid_search: data.searchType === 'hybrid',
      };

      let response: SearchResponse;
      
      switch (data.searchType) {
        case 'semantic':
          response = await searchApi.semanticSearch(searchQuery);
          break;
        case 'hybrid':
          response = await searchApi.hybridSearch(searchQuery);
          break;
        default:
          response = await searchApi.search(searchQuery);
      }

      setSearchResponse(response);
      fetchSearchHistory(); // Refresh history after search
      
    } catch (error: any) {
      toast.error('Search failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const saveCurrentSearch = async () => {
    const query = watch('query');
    const searchType = watch('searchType');
    if (!query.trim()) return;

    const name = prompt('Enter a name for this search:');
    if (!name) return;

    try {
      // Build search query object
      const searchFilters: SearchFiltersType = {};
      
      if (filters.file_type.length > 0) {
        searchFilters.file_types = filters.file_type;
      }
      
      if (filters.tags.length > 0) {
        searchFilters.tag_ids = filters.tags;
      }

      if (filters.folder_ids && filters.folder_ids.length > 0) {
        searchFilters.folder_ids = filters.folder_ids;
      }
      
      if (filters.date_range) {
        searchFilters.date_range = [filters.date_range.start, filters.date_range.end];
      }
      
      if (filters.owner) {
        searchFilters.metadata_filters = { owner: filters.owner };
      }

      if (filters.languages && filters.languages.length > 0) {
        searchFilters.metadata_filters = {
          ...searchFilters.metadata_filters,
          languages: filters.languages
        };
      }

      const searchQuery: SearchQuery = {
        query: query,
        filters: Object.keys(searchFilters).length > 0 ? searchFilters : undefined,
        page: 1,
        page_size: 20,
        top_k: 20,
        similarity_threshold: 0.0,
        semantic_search: searchType === 'semantic',
        hybrid_search: searchType === 'hybrid',
      };

      await searchApi.saveSearch(searchQuery, name);
      toast.success('Search saved successfully');
      fetchSavedSearches();
    } catch (error: any) {
      toast.error('Failed to save search');
    }
  };

  const loadSavedSearch = (savedSearch: SavedSearch) => {
    setValue('query', savedSearch.query_text);
    setValue('searchType', savedSearch.search_type as 'basic' | 'semantic' | 'hybrid');
    
    // Load filters if available
    if (savedSearch.filters) {
      const searchFilters = savedSearch.filters;
      setFilters({
        file_type: searchFilters.file_types || [],
        tags: searchFilters.tag_ids || [],
        folder_ids: searchFilters.folder_ids || [],
        languages: searchFilters.metadata_filters?.languages || [],
        date_range: searchFilters.date_range ? {
          start: searchFilters.date_range[0],
          end: searchFilters.date_range[1]
        } : null,
        owner: searchFilters.metadata_filters?.owner || '',
      });
    }
    
    setShowSuggestions(false);
  };

  const loadHistorySearch = (historyItem: RecentSearch) => {
    setValue('query', historyItem.query_text);
    setValue('searchType', historyItem.query_type as 'basic' | 'semantic' | 'hybrid');
    
    // Load filters if available
    if (historyItem.filters) {
      const searchFilters = historyItem.filters;
      setFilters({
        file_type: searchFilters.file_types || [],
        tags: searchFilters.tag_ids || [],
        folder_ids: searchFilters.folder_ids || [],
        languages: searchFilters.metadata_filters?.languages || [],
        date_range: searchFilters.date_range ? {
          start: searchFilters.date_range[0],
          end: searchFilters.date_range[1]
        } : null,
        owner: searchFilters.metadata_filters?.owner || '',
      });
    }
    
    setShowSuggestions(false);
  };

  const selectSuggestion = (suggestion: SearchSuggestion) => {
    setValue('query', suggestion.text);
    setShowSuggestions(false);
    // Optionally trigger search immediately
    handleSubmit(onSearch)();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-6">
        {/* Search Form */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <form onSubmit={handleSubmit(onSearch)} className="space-y-4">
            {/* Search Input */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                {...register('query', { required: 'Please enter a search query' })}
                type="text"
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                placeholder="Search documents..."
                onFocus={() => {
                  const currentQuery = watch('query');
                  if (currentQuery && suggestions.length > 0) {
                    setShowSuggestions(true);
                  }
                }}
                onBlur={() => {
                  // Delay hiding suggestions to allow clicks
                  setTimeout(() => setShowSuggestions(false), 200);
                }}
              />
              {errors.query && (
                <p className="mt-1 text-sm text-red-600">{errors.query.message}</p>
              )}
              
              {/* Search Suggestions Dropdown */}
              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                  {suggestionLoading && (
                    <div className="px-3 py-2 text-sm text-gray-500">Loading suggestions...</div>
                  )}
                  {suggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      className="px-3 py-2 cursor-pointer hover:bg-gray-50 flex items-center space-x-2"
                      onClick={() => selectSuggestion(suggestion)}
                    >
                      <span className="text-gray-400">
                        {suggestion.icon === 'clock' && '⏰'}
                        {suggestion.icon === 'tag' && '🏷️'}
                        {suggestion.icon === 'document-text' && '📄'}
                        {suggestion.icon === 'bookmark' && '🔖'}
                        {suggestion.icon === 'light-bulb' && '💡'}
                        {suggestion.icon === 'magnifying-glass' && '🔍'}
                      </span>
                      <span className="text-sm text-gray-700">{suggestion.text}</span>
                      <span className="text-xs text-gray-400 ml-auto">
                        {suggestion.type === 'history' && 'Recent'}
                        {suggestion.type === 'tag' && 'Tag'}
                        {suggestion.type === 'document_title' && 'Document'}
                        {suggestion.type === 'saved_search' && 'Saved'}
                        {suggestion.type === 'suggestion' && 'Suggestion'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Search Options */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                {/* Search Type */}
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium text-gray-700">Search Type:</label>
                  <select
                    {...register('searchType')}
                    className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="hybrid">Hybrid (Recommended)</option>
                    <option value="semantic">Semantic</option>
                    <option value="basic">Basic</option>
                  </select>
                </div>

                {/* Filters Toggle */}
                <button
                  type="button"
                  onClick={() => setShowFilters(!showFilters)}
                  className="inline-flex items-center px-3 py-1 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  <AdjustmentsHorizontalIcon className="h-4 w-4 mr-1" />
                  Filters
                </button>
              </div>

              <div className="flex items-center space-x-2">
                {watch('query') && (
                  <button
                    type="button"
                    onClick={saveCurrentSearch}
                    className="inline-flex items-center px-3 py-1 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                  >
                    <BookmarkIcon className="h-4 w-4 mr-1" />
                    Save
                  </button>
                )}
                
                <button
                  type="submit"
                  disabled={loading}
                  className="btn-primary"
                >
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>

            {/* Search Type Description */}
            <div className="text-xs text-gray-500">
              {searchType === 'basic' && 'Keyword-based search using traditional text matching'}
              {searchType === 'semantic' && 'AI-powered semantic search understanding context and meaning'}
              {searchType === 'hybrid' && '🚀 Recommended: Combined keyword and semantic search for best results'}
            </div>
          </form>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <SearchFilters
                filters={filters}
                onFiltersChange={setFilters}
              />
            </div>
          )}
        </div>

        {/* Sidebar with History and Saved Searches */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-6">
            {/* Search History */}
            {searchHistory.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center">
                  <ClockIcon className="h-4 w-4 mr-1" />
                  Recent Searches
                </h3>
                <div className="space-y-2">
                  {searchHistory.slice(0, 5).map((item, index) => (
                    <button
                      key={index}
                      onClick={() => loadHistorySearch(item)}
                      className="block w-full text-left px-2 py-1 text-sm text-gray-600 hover:bg-gray-50 rounded truncate"
                      title={item.query_text}
                    >
                      <div className="truncate">{item.query_text}</div>
                      <div className="text-xs text-gray-400">
                        {item.results_count} results • {new Date(item.created_at).toLocaleDateString()}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Saved Searches */}
            {savedSearches.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center">
                  <BookmarkIcon className="h-4 w-4 mr-1" />
                  Saved Searches
                </h3>
                <div className="space-y-2">
                  {savedSearches.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => loadSavedSearch(item)}
                      className="block w-full text-left px-2 py-1 text-sm text-gray-600 hover:bg-gray-50 rounded"
                      title={item.description || item.query_text}
                    >
                      <div className="font-medium truncate">{item.name}</div>
                      <div className="text-xs text-gray-500 truncate">{item.query_text}</div>
                      <div className="text-xs text-gray-400">
                        {item.search_type} • Used {item.usage_count} times
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Search Results */}
          <div className="lg:col-span-3">
            {searchResponse && (
              <SearchResults
                response={searchResponse}
                loading={loading}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}