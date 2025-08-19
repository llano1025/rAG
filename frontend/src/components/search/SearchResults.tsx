import { useState } from 'react';
import { format } from 'date-fns';
import {
  DocumentTextIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { StarIcon as StarIconSolid } from '@heroicons/react/24/solid';
import { SearchResponse } from '@/types';

interface SearchResultsProps {
  response: SearchResponse;
  loading: boolean;
}

export default function SearchResults({ response, loading }: SearchResultsProps) {
  const [selectedResult, setSelectedResult] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="border-b border-gray-200 pb-4">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gray-200 rounded"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  <div className="h-3 bg-gray-200 rounded w-full"></div>
                  <div className="h-3 bg-gray-200 rounded w-2/3"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!response || response.results.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No results found</h3>
        <p className="mt-1 text-sm text-gray-500">
          Try adjusting your search query or filters.
        </p>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const formatScore = (score: number) => {
    return Math.round(score * 100);
  };

  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 px-1 rounded">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-md">
      {/* Results Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-medium text-gray-900">Search Results</h2>
            <div className="flex items-center space-x-4">
              <p className="text-sm text-gray-500">
                {response.total_hits} result{response.total_hits !== 1 ? 's' : ''} found in {response.processing_time ? response.processing_time.toFixed(3) : 'N/A'}s
              </p>
              {response.results.some(r => r.rerank_score !== undefined) && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                  🎯 Reranked Results
                </span>
              )}
            </div>
          </div>
          <div className="text-sm text-gray-500">
            Query: <span className="font-medium">&ldquo;{response.query || 'No query'}&rdquo;</span>
          </div>
        </div>
      </div>

      {/* Results List */}
      <div className="divide-y divide-gray-200">
        {response.results.map((result, index) => (
          <div
            key={result.document_id}
            className={`p-6 hover:bg-gray-50 transition-colors ${
              selectedResult === result.document_id ? 'bg-blue-50' : ''
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                {/* Document Title and Score */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <DocumentTextIcon className="h-5 w-5 text-gray-400" />
                    <h3 className="text-base font-medium text-gray-900 truncate">
                      {result.filename}
                    </h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(result.score)}`}>
                      {formatScore(result.score)}% match
                    </span>
                    {result.rerank_score !== undefined && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800" title="Reranked score">
                        🎯 {formatScore(result.rerank_score)}%
                      </span>
                    )}
                    <span className="text-xs text-gray-500">#{index + 1}</span>
                  </div>
                </div>

                {/* Content Snippet */}
                <div className="mb-3">
                  <p className="text-sm text-gray-600 leading-relaxed">
                    {highlightText(result.content_snippet, response.query || '')}
                  </p>
                </div>

                {/* Metadata */}
                {result.metadata && (
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    {result.metadata.upload_date && (
                      <span>
                        Uploaded: {format(new Date(result.metadata.upload_date), 'MMM d, yyyy')}
                      </span>
                    )}
                    {result.metadata.file_size && (
                      <span>
                        Size: {(result.metadata.file_size / 1024 / 1024).toFixed(1)} MB
                      </span>
                    )}
                    {result.metadata.owner && (
                      <span>
                        Owner: {result.metadata.owner}
                      </span>
                    )}
                    {result.metadata.tags && result.metadata.tags.length > 0 && (
                      <span>
                        Tags: {result.metadata.tags.join(', ')}
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center space-x-2 ml-4">
                <button
                  onClick={() => setSelectedResult(
                    selectedResult === result.document_id ? null : result.document_id
                  )}
                  className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
                  title="Preview"
                >
                  <EyeIcon className="h-4 w-4" />
                </button>
                
                <button
                  className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
                  title="Download"
                >
                  <ArrowDownTrayIcon className="h-4 w-4" />
                </button>
                
                <button
                  className="p-2 text-gray-400 hover:text-yellow-500 rounded-md hover:bg-gray-100"
                  title="Bookmark"
                >
                  <StarIcon className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Expanded Details */}
            {selectedResult === result.document_id && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Document Details</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Document ID:</span>
                      <span className="ml-2 font-mono text-xs">{result.document_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Final Score:</span>
                      <span className="ml-2">{result.score.toFixed(4)}</span>
                    </div>
                    {result.original_score !== undefined && (
                      <div>
                        <span className="text-gray-500">Original Score:</span>
                        <span className="ml-2">{result.original_score.toFixed(4)}</span>
                      </div>
                    )}
                    {result.rerank_score !== undefined && (
                      <div>
                        <span className="text-gray-500">Rerank Score:</span>
                        <span className="ml-2">{result.rerank_score.toFixed(4)}</span>
                      </div>
                    )}
                    {result.combined_score !== undefined && (
                      <div>
                        <span className="text-gray-500">Combined Score:</span>
                        <span className="ml-2">{result.combined_score.toFixed(4)}</span>
                      </div>
                    )}
                    {result.reranker_model && (
                      <div>
                        <span className="text-gray-500">Reranker Model:</span>
                        <span className="ml-2 font-mono text-xs">{result.reranker_model}</span>
                      </div>
                    )}
                    {result.metadata && Object.entries(result.metadata).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-gray-500 capitalize">{key.replace('_', ' ')}:</span>
                        <span className="ml-2">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Load More Button (if needed) */}
      {response.total_hits > response.results.length && (
        <div className="px-6 py-4 border-t border-gray-200 text-center">
          <button className="btn-secondary">
            Load more results
          </button>
        </div>
      )}
    </div>
  );
}