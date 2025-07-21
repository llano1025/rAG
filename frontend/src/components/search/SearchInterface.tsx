import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import {
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  ClockIcon,
  BookmarkIcon,
} from '@heroicons/react/24/outline';
import { searchApi } from '@/api/search';
import { SearchQuery, SearchResponse } from '@/types';
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
  const [searchHistory, setSearchHistory] = useState<Array<{ query: string; timestamp: string }>>([]);
  const [savedSearches, setSavedSearches] = useState<Array<{ id: string; name: string; query: string }>>([]);
  const [filters, setFilters] = useState({
    file_type: [] as string[],
    date_range: null as { start: string; end: string } | null,
    owner: '',
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
      const history = await searchApi.getSearchHistory();
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

  const onSearch = async (data: SearchForm) => {
    if (!data.query.trim()) return;

    setLoading(true);
    try {
      const searchQuery: SearchQuery = {
        query: data.query,
        filters: {
          file_type: filters.file_type.length > 0 ? filters.file_type : undefined,
          date_range: filters.date_range || undefined,
          owner: filters.owner || undefined,
        },
        limit: 20,
        offset: 0,
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
    if (!query.trim()) return;

    const name = prompt('Enter a name for this search:');
    if (!name) return;

    try {
      await searchApi.saveSearch(query, name);
      toast.success('Search saved successfully');
      fetchSavedSearches();
    } catch (error: any) {
      toast.error('Failed to save search');
    }
  };

  const loadSavedSearch = (query: string) => {
    setValue('query', query);
  };

  const loadHistorySearch = (query: string) => {
    setValue('query', query);
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
              />
              {errors.query && (
                <p className="mt-1 text-sm text-red-600">{errors.query.message}</p>
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
                    <option value="basic">Basic</option>
                    <option value="semantic">Semantic</option>
                    <option value="hybrid">Hybrid</option>
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
              {searchType === 'hybrid' && 'Combined keyword and semantic search for best results'}
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
                      onClick={() => loadHistorySearch(item.query)}
                      className="block w-full text-left px-2 py-1 text-sm text-gray-600 hover:bg-gray-50 rounded truncate"
                      title={item.query}
                    >
                      {item.query}
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
                      onClick={() => loadSavedSearch(item.query)}
                      className="block w-full text-left px-2 py-1 text-sm text-gray-600 hover:bg-gray-50 rounded"
                    >
                      <div className="font-medium truncate">{item.name}</div>
                      <div className="text-xs text-gray-500 truncate">{item.query}</div>
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