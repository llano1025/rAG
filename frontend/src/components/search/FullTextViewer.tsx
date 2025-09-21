import { useState, useEffect, useRef } from 'react';
import {
  ClipboardDocumentIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  ChevronUpIcon,
  ChevronDownIcon
} from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import toast from 'react-hot-toast';

interface FullTextViewerProps {
  documentId: string;
  chunkId?: string;
  filename: string;
  searchQuery?: string;
  onClose?: () => void;
}

interface ChunkContent {
  chunk_id: string;
  chunk_index: number;
  text: string;
  text_length: number;
  start_char?: number;
  end_char?: number;
  context_before?: string;
  context_after?: string;
}

export default function FullTextViewer({ documentId, chunkId, filename, searchQuery, onClose }: FullTextViewerProps) {
  const [content, setContent] = useState<ChunkContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState(searchQuery || '');
  const [searchResults, setSearchResults] = useState<number[]>([]);
  const [currentSearchIndex, setCurrentSearchIndex] = useState(0);
  const [showSearch, setShowSearch] = useState(false);
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadContent();
  }, [documentId, chunkId]);

  useEffect(() => {
    if (searchQuery) {
      setSearchTerm(searchQuery);
      setShowSearch(true);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (searchTerm && content) {
      findSearchMatches();
    } else {
      setSearchResults([]);
      setCurrentSearchIndex(0);
    }
  }, [searchTerm, content]);

  const loadContent = async () => {
    try {
      setLoading(true);
      setError(null);

      if (chunkId) {
        // Load specific chunk content
        const data = await documentsApi.getChunkContent(documentId, chunkId);
        setContent(data);
      } else {
        // Fallback to full document content
        const data = await documentsApi.getDocumentContent(documentId);
        // Transform document content to chunk-like structure
        setContent({
          chunk_id: 'full_document',
          chunk_index: 0,
          text: data.extracted_text,
          text_length: data.text_length,
        });
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load content');
      toast.error('Failed to load content');
    } finally {
      setLoading(false);
    }
  };

  const findSearchMatches = () => {
    if (!searchTerm || !content) return;

    const text = content.text.toLowerCase();
    const term = searchTerm.toLowerCase();
    const matches: number[] = [];
    let index = 0;

    while ((index = text.indexOf(term, index)) !== -1) {
      matches.push(index);
      index += term.length;
    }

    setSearchResults(matches);
    setCurrentSearchIndex(0);

    // Scroll to first match
    if (matches.length > 0) {
      scrollToMatch(0);
    }
  };

  const scrollToMatch = (matchIndex: number) => {
    if (!textRef.current || searchResults.length === 0) return;

    const textContainer = textRef.current;
    const text = content?.text || '';
    const matchPosition = searchResults[matchIndex];

    // Calculate approximate scroll position
    const totalLength = text.length;
    const scrollPercentage = matchPosition / totalLength;
    const maxScroll = textContainer.scrollHeight - textContainer.clientHeight;
    const targetScroll = scrollPercentage * maxScroll;

    textContainer.scrollTo({
      top: targetScroll,
      behavior: 'smooth'
    });
  };

  const navigateSearch = (direction: 'prev' | 'next') => {
    if (searchResults.length === 0) return;

    let newIndex: number;
    if (direction === 'next') {
      newIndex = (currentSearchIndex + 1) % searchResults.length;
    } else {
      newIndex = currentSearchIndex === 0 ? searchResults.length - 1 : currentSearchIndex - 1;
    }

    setCurrentSearchIndex(newIndex);
    scrollToMatch(newIndex);
  };

  const highlightText = (text: string, term: string) => {
    if (!term) return text;

    const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) => {
      if (regex.test(part)) {
        return (
          <mark key={index} className="bg-yellow-200 px-1 rounded font-medium">
            {part}
          </mark>
        );
      }
      return part;
    });
  };

  const copyToClipboard = async () => {
    if (!content) return;

    try {
      await navigator.clipboard.writeText(content.text);
      toast.success('Text copied to clipboard');
    } catch (err) {
      toast.error('Failed to copy text');
    }
  };

  const downloadText = () => {
    if (!content) return;

    const blob = new Blob([content.text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Text file downloaded');
  };

  if (loading) {
    return (
      <div className="bg-gray-50 rounded-lg p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading {chunkId ? 'chunk' : 'document'} content...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 rounded-lg p-6">
        <div className="text-center">
          <p className="text-red-600">{error}</p>
          <button
            onClick={loadContent}
            className="mt-2 btn-secondary text-sm"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!content) {
    return (
      <div className="bg-gray-50 rounded-lg p-6">
        <p className="text-center text-gray-600">No content available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex-1">
          <h3 className="text-lg font-medium text-gray-900 truncate">{filename}</h3>
          <p className="text-sm text-gray-500">
            {chunkId ? (
              <>
                Chunk {content.chunk_index + 1} • {content.text_length.toLocaleString()} characters
                {content.start_char !== undefined && content.end_char !== undefined && (
                  <> • Position {content.start_char}-{content.end_char}</>
                )}
              </>
            ) : (
              `${content.text_length.toLocaleString()} characters • Full document`
            )}
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {/* Search Toggle */}
          <button
            onClick={() => setShowSearch(!showSearch)}
            className={`p-2 rounded-md transition-colors ${
              showSearch ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
            }`}
            title="Search in text"
          >
            <MagnifyingGlassIcon className="h-4 w-4" />
          </button>

          {/* Copy Button */}
          <button
            onClick={copyToClipboard}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
            title="Copy text"
          >
            <ClipboardDocumentIcon className="h-4 w-4" />
          </button>

          {/* Download Button */}
          <button
            onClick={downloadText}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
            title="Download as text file"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
          </button>

          {/* Close Button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
              title="Close"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Search Bar */}
      {showSearch && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center space-x-2">
            <div className="flex-1 relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search in document..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {searchResults.length > 0 && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">
                  {currentSearchIndex + 1} of {searchResults.length}
                </span>
                <button
                  onClick={() => navigateSearch('prev')}
                  className="p-1 text-gray-400 hover:text-gray-600 rounded"
                  disabled={searchResults.length === 0}
                >
                  <ChevronUpIcon className="h-4 w-4" />
                </button>
                <button
                  onClick={() => navigateSearch('next')}
                  className="p-1 text-gray-400 hover:text-gray-600 rounded"
                  disabled={searchResults.length === 0}
                >
                  <ChevronDownIcon className="h-4 w-4" />
                </button>
              </div>
            )}

            {searchTerm && searchResults.length === 0 && (
              <span className="text-sm text-red-600">No matches found</span>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div
        ref={textRef}
        className="p-6 max-h-96 overflow-y-auto"
        style={{ lineHeight: '1.6' }}
      >
        {/* Context Before */}
        {content.context_before && (
          <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded mb-4 border-l-4 border-gray-300">
            <p className="text-xs text-gray-400 mb-1">Context before:</p>
            <div className="italic">
              {searchTerm ? highlightText(content.context_before, searchTerm) : content.context_before}
            </div>
          </div>
        )}

        {/* Main Chunk Content */}
        <div className="text-sm text-gray-800 whitespace-pre-wrap font-mono leading-relaxed bg-blue-50 p-4 rounded border-l-4 border-blue-500">
          <p className="text-xs text-blue-600 mb-2 font-sans">
            {chunkId ? `Chunk ${content.chunk_index + 1} content:` : 'Document content:'}
          </p>
          {searchTerm ? highlightText(content.text, searchTerm) : content.text}
        </div>

        {/* Context After */}
        {content.context_after && (
          <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded mt-4 border-l-4 border-gray-300">
            <p className="text-xs text-gray-400 mb-1">Context after:</p>
            <div className="italic">
              {searchTerm ? highlightText(content.context_after, searchTerm) : content.context_after}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}