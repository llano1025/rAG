import { useState, useEffect, useCallback, useRef } from 'react';
import { XMarkIcon, ArrowDownTrayIcon, MagnifyingGlassIcon, ClipboardDocumentIcon } from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import { Document } from '@/types';

interface DocumentPreviewProps {
  document: Document;
  isOpen: boolean;
  onClose: () => void;
}

interface DocumentContent {
  document_id: number;
  filename: string;
  extracted_text: string;
  text_length: number;
  chunks_count: number;
  language: string;
  content_type: string;
}

type TabType = 'original' | 'extracted' | 'keywords';

// Utility function to convert MIME types to user-friendly labels
const getFileTypeLabel = (contentType: string | null): string => {
  if (!contentType) return 'Unknown';
  
  const mimeType = contentType.toLowerCase();
  
  // Image types
  if (mimeType === 'image/jpeg' || mimeType === 'image/jpg') return 'JPEG Image';
  if (mimeType === 'image/png') return 'PNG Image';
  if (mimeType === 'image/gif') return 'GIF Image';
  if (mimeType === 'image/tiff' || mimeType === 'image/tif') return 'TIFF Image';
  if (mimeType === 'image/webp') return 'WebP Image';
  if (mimeType === 'image/bmp') return 'BMP Image';
  
  // Document types
  if (mimeType === 'application/pdf') return 'PDF Document';
  if (mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return 'Word Document';
  if (mimeType === 'application/msword') return 'Word Document';
  
  // Text types
  if (mimeType === 'text/plain') return 'Text File';
  if (mimeType === 'text/html') return 'HTML Document';
  if (mimeType === 'text/css') return 'CSS File';
  if (mimeType === 'text/javascript') return 'JavaScript File';
  if (mimeType === 'text/markdown') return 'Markdown File';
  if (mimeType === 'text/csv') return 'CSV File';
  
  // Application types
  if (mimeType === 'application/json') return 'JSON File';
  if (mimeType === 'application/xml') return 'XML File';
  if (mimeType === 'application/zip') return 'ZIP Archive';
  
  // Generic fallback
  if (mimeType.startsWith('image/')) return 'Image';
  if (mimeType.startsWith('text/')) return 'Text File';
  if (mimeType.startsWith('application/')) return 'Document';
  
  return contentType.toUpperCase();
};

export default function DocumentPreview({ document, isOpen, onClose }: DocumentPreviewProps) {
  const [activeTab, setActiveTab] = useState<TabType>('original');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [documentContent, setDocumentContent] = useState<DocumentContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [contentLoading, setContentLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [copySuccess, setCopySuccess] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  
  // Use ref to track current blob URL for proper cleanup
  const currentBlobUrlRef = useRef<string | null>(null);

  const loadPreview = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setImageError(null);
      
      console.log('🔄 Loading preview for document:', {
        id: document.id,
        filename: document.filename,
        contentType: document.content_type,
        fileSize: document.file_size
      });
      
      // Clean up previous blob URL if it exists
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
      }
      
      const blob = await documentsApi.downloadDocument(document.id);
      
      console.log('📦 Blob received:', {
        size: blob.size,
        type: blob.type,
        isValid: blob && blob.size > 0
      });
      
      if (blob && blob.size > 0) {
        // Additional validation for image files
        const contentType = document.content_type?.toLowerCase() || '';
        if (contentType.startsWith('image/')) {
          console.log('🖼️ Processing image file:', {
            documentContentType: contentType,
            blobType: blob.type,
            size: blob.size
          });
          
          // Validate blob type matches expected image type
          if (blob.type && !blob.type.startsWith('image/')) {
            console.warn('⚠️ Blob type mismatch - expected image, got:', blob.type);
          }
        }
        
        const url = URL.createObjectURL(blob);
        currentBlobUrlRef.current = url;
        setPreviewUrl(url);
        
        console.log('✅ Preview URL created successfully:', url.substring(0, 50) + '...');
      } else {
        throw new Error('Empty or invalid blob received');
      }
    } catch (err: any) {
      console.error('❌ Document preview error:', err);
      console.error('Error details:', {
        message: err.message,
        status: err.response?.status,
        statusText: err.response?.statusText,
        documentId: document.id,
        filename: document.filename
      });
      
      // Provide more specific error messages based on server response
      let errorMessage = 'Failed to load document preview';
      const errorDetail = err.response?.data?.detail;
      
      if (err.response?.status === 404) {
        if (errorDetail?.error) {
          // Use server-provided error message
          errorMessage = errorDetail.message || errorDetail.error;
          
          // Add additional context for specific error types
          if (errorDetail.file_path_missing) {
            errorMessage += ' The file path was not properly stored during upload.';
          } else if (errorDetail.expected_path) {
            errorMessage += ` Expected file location: ${errorDetail.expected_path}`;
          }
          
          // Add suggestions if provided
          if (errorDetail.suggestions && errorDetail.suggestions.length > 0) {
            errorMessage += `\n\nSuggestions:\n• ${errorDetail.suggestions.join('\n• ')}`;
          }
        } else if (err.response?.data?.detail?.includes('missing from storage')) {
          errorMessage = 'Original image file is missing from storage. Please re-upload the file.';
        } else {
          errorMessage = 'Document not found. It may have been deleted or moved.';
        }
      } else if (err.response?.status === 500) {
        errorMessage = 'Server error loading document. Please try again or contact support.';
      } else if (err.message?.includes('Network Error')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      }
      
      setError(errorMessage);
      setPreviewUrl(null);
    } finally {
      setLoading(false);
    }
  }, [document.id, document.filename, document.content_type, document.file_size]);

  const loadDocumentContent = useCallback(async () => {
    if (documentContent) return; // Already loaded
    
    try {
      setContentLoading(true);
      setContentError(null);
      
      const content = await documentsApi.getDocumentContent(document.id);
      setDocumentContent(content);
    } catch (err: any) {
      setContentError('Failed to load extracted text');
    } finally {
      setContentLoading(false);
    }
  }, [document.id, documentContent]);

  // Effect to load content when tab changes
  useEffect(() => {
    if (isOpen && document) {
      if (activeTab === 'original') {
        loadPreview();
      } else if (activeTab === 'extracted') {
        loadDocumentContent();
      }
    }
  }, [isOpen, document, activeTab, loadPreview, loadDocumentContent]);

  // Effect to reset states when document changes
  useEffect(() => {
    if (isOpen && document) {
      setActiveTab('original');
      setSearchTerm('');
      setCopySuccess(false);
      setError(null);
      setContentError(null);
      setDocumentContent(null);
      setPreviewUrl(null);
      setImageLoading(false);
      setImageError(null);
      
      // Clean up previous blob URL when document changes
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
      }
    }
  }, [document, isOpen]);

  // Cleanup effect when component unmounts or modal closes
  useEffect(() => {
    return () => {
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
      }
    };
  }, []);

  const handleDownload = async (type: 'original' | 'text' = 'original') => {
    try {
      let blob: Blob;
      let filename: string;
      
      if (type === 'text') {
        blob = await documentsApi.downloadExtractedText(document.id);
        filename = `${document.filename.replace(/\.[^/.]+$/, '')}_extracted.txt`;
      } else {
        blob = await documentsApi.downloadDocument(document.id);
        filename = document.filename;
      }
      
      // Verify blob is valid
      if (!blob || blob.size === 0) {
        throw new Error('Empty or invalid file received');
      }
      
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = filename;
      window.document.body.appendChild(a);
      a.click();
      
      // Clean up
      window.URL.revokeObjectURL(url);
      window.document.body.removeChild(a);
    } catch (error: any) {
      console.error('Download error:', error);
      const errorMessage = type === 'text' 
        ? 'Failed to download extracted text' 
        : 'Failed to download document';
      setError(errorMessage);
    }
  };

  const handleCopyText = async () => {
    if (!documentContent?.extracted_text) return;
    
    try {
      await navigator.clipboard.writeText(documentContent.extracted_text);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  const highlightSearchTerm = (text: string, term: string) => {
    if (!term.trim()) return text;
    
    const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
  };

  const getSearchMatches = (text: string, term: string) => {
    if (!term.trim()) return 0;
    const regex = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    return (text.match(regex) || []).length;
  };

  const renderOriginalPreview = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="text-center max-w-2xl">
            <div className="text-red-600 mb-4">
              {error.split('\n').map((line, index) => (
                <p key={index} className={index > 0 ? 'mt-2 text-sm' : ''}>
                  {line}
                </p>
              ))}
            </div>
            <div className="space-x-2">
              <button
                onClick={loadPreview}
                className="btn-primary"
              >
                Retry
              </button>
              <button
                onClick={() => window.location.reload()}
                className="btn-secondary"
              >
                Refresh Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    if (!previewUrl) {
      return (
        <div className="flex items-center justify-center h-96">
          <p className="text-gray-500">No preview available</p>
        </div>
      );
    }

    // Handle different file types
    const contentType = document.content_type ? document.content_type.toLowerCase() : '';
    
    // PDF Documents
    if (contentType === 'application/pdf') {
      return (
        <iframe
          src={previewUrl}
          className="w-full h-96 border-0"
          title="PDF Document Preview"
        />
      );
    }
    
    // Image files - comprehensive support
    if ([
      'image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'image/webp',
      'image/tiff', 'image/tif', 'image/bmp', 'image/svg+xml'
    ].includes(contentType)) {
      
      // Handle image loading states and errors
      if (imageLoading) {
        return (
          <div className="flex justify-center items-center min-h-96 bg-gray-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
              <p className="text-sm text-gray-500">Loading image...</p>
            </div>
          </div>
        );
      }
      
      if (imageError) {
        // Create direct download URL for testing
        const directUrl = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/documents/${document.id}/download`;
        
        return (
          <div className="flex justify-center items-center min-h-96 bg-gray-50">
            <div className="text-center max-w-md">
              <div className="mb-4">
                <svg className="w-16 h-16 text-red-400 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                  <path d="M6 8a1 1 0 100-2 1 1 0 000 2z" />
                </svg>
              </div>
              <p className="text-red-600 mb-2">Failed to load image</p>
              <p className="text-sm text-gray-500 mb-4">{imageError}</p>
              
              {/* Debug information */}
              <div className="bg-gray-100 rounded p-3 mb-4 text-xs text-left">
                <p><strong>Debug Info:</strong></p>
                <p><strong>Document ID:</strong> {document.id}</p>
                <p><strong>Content Type:</strong> {document.content_type}</p>
                <p><strong>Blob URL:</strong> {previewUrl ? previewUrl.substring(0, 50) + '...' : 'None'}</p>
                <p><strong>Direct URL:</strong></p>
                <a 
                  href={directUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline text-xs break-all"
                >
                  {directUrl}
                </a>
              </div>
              
              <div className="space-x-2">
                <button
                  onClick={() => {
                    setImageError(null);
                    setImageLoading(true);
                    loadPreview();
                  }}
                  className="btn-secondary text-sm"
                >
                  Retry
                </button>
                <button
                  onClick={() => handleDownload('original')}
                  className="btn-primary text-sm"
                >
                  Download
                </button>
                <button
                  onClick={() => window.open(directUrl, '_blank')}
                  className="btn-secondary text-sm"
                >
                  Test Direct URL
                </button>
              </div>
            </div>
          </div>
        );
      }
      
      return (
        <div className="flex justify-center items-center min-h-96 bg-gray-50">
          <img
            src={previewUrl}
            alt="Document Preview"
            className="max-w-full max-h-96 h-auto mx-auto shadow-lg rounded"
            onLoad={() => {
              console.log('✅ Image loaded successfully:', document.filename);
              setImageLoading(false);
              setImageError(null);
            }}
            onError={(e) => {
              console.error('❌ Image failed to load:', {
                filename: document.filename,
                previewUrl: previewUrl,
                error: e
              });
              setImageLoading(false);
              setImageError(`Unable to display ${getFileTypeLabel(document.content_type)} file`);
            }}
            onLoadStart={() => {
              console.log('🔄 Image load started:', document.filename);
              setImageLoading(true);
              setImageError(null);
            }}
          />
        </div>
      );
    }
    
    // Text and code files
    if ([
      'text/plain', 'text/html', 'text/css', 'text/javascript', 
      'text/markdown', 'text/csv', 'application/json', 'application/xml'
    ].includes(contentType)) {
      return (
        <iframe
          src={previewUrl}
          className="w-full h-96 border border-gray-200 rounded"
          title="Text Document Preview"
        />
      );
    }
    
    // Microsoft Word documents
    if ([
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword'
    ].includes(contentType)) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="mb-4">
              <svg className="w-16 h-16 text-blue-500 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="text-gray-700 mb-4">Word Document</p>
            <p className="text-sm text-gray-500 mb-4">Preview not available for Word documents</p>
            <button
              onClick={() => handleDownload('original')}
              className="btn-primary"
            >
              Download to view
            </button>
          </div>
        </div>
      );
    }

    // For other unsupported file types, show download option with file type info
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="mb-4">
            <svg className="w-16 h-16 text-gray-400 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
            </svg>
          </div>
          <p className="text-gray-700 mb-2">{getFileTypeLabel(document.content_type)}</p>
          <p className="text-sm text-gray-500 mb-4">Preview not available for this file type</p>
          <button
            onClick={() => handleDownload('original')}
            className="btn-primary"
          >
            Download to view
          </button>
        </div>
      </div>
    );
  };

  const renderExtractedText = () => {
    if (contentLoading) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      );
    }

    if (contentError) {
      return (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <p className="text-red-600 mb-4">{contentError}</p>
            <button
              onClick={loadDocumentContent}
              className="btn-primary"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    if (!documentContent?.extracted_text) {
      return (
        <div className="flex items-center justify-center h-96">
          <p className="text-gray-500">No extracted text available</p>
        </div>
      );
    }

    const searchMatches = getSearchMatches(documentContent.extracted_text, searchTerm);
    const highlightedText = highlightSearchTerm(documentContent.extracted_text, searchTerm);

    return (
      <div className="space-y-4">
        {/* Search and Actions Bar */}
        <div className="flex flex-wrap items-center gap-3 p-3 bg-gray-50 rounded-lg">
          <div className="flex-1 min-w-64">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search in extracted text..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
            {searchTerm && (
              <p className="text-sm text-gray-600 mt-1">
                {searchMatches} matches found
              </p>
            )}
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={handleCopyText}
              className="btn-secondary flex items-center gap-2"
              title="Copy extracted text"
            >
              <ClipboardDocumentIcon className="h-4 w-4" />
              {copySuccess ? 'Copied!' : 'Copy'}
            </button>
            
            <button
              onClick={() => handleDownload('text')}
              className="btn-secondary flex items-center gap-2"
              title="Download extracted text"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              Download
            </button>
          </div>
        </div>

        {/* Text Content */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 max-h-96 overflow-y-auto">
          <div className="text-sm text-gray-600 mb-4">
            <span className="font-medium">Language:</span> {documentContent.language || 'Unknown'} • 
            <span className="font-medium">Length:</span> {documentContent.text_length.toLocaleString()} characters • 
            <span className="font-medium">Chunks:</span> {documentContent.chunks_count}
          </div>
          
          <div 
            className="prose prose-sm max-w-none text-gray-900 whitespace-pre-wrap leading-relaxed"
            dangerouslySetInnerHTML={{ __html: highlightedText }}
          />
        </div>
      </div>
    );
  };

  const renderKeywords = () => {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-gray-500 mb-2">Keyword extraction feature</p>
          <p className="text-sm text-gray-400">Coming soon...</p>
        </div>
      </div>
    );
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'original':
        return renderOriginalPreview();
      case 'extracted':
        return renderExtractedText();
      case 'keywords':
        return renderKeywords();
      default:
        return renderOriginalPreview();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        {/* Backdrop */}
        <div
          className="fixed inset-0 transition-opacity bg-gray-500 bg-opacity-75"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="inline-block w-full max-w-6xl my-8 overflow-hidden text-left align-middle transition-all transform bg-white shadow-xl rounded-lg">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-medium text-gray-900 truncate">
                {document.filename}
              </h3>
              <p className="text-sm text-gray-500">
                {getFileTypeLabel(document.content_type)} • {(document.file_size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleDownload('original')}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md"
                title="Download Original"
              >
                <ArrowDownTrayIcon className="h-5 w-5" />
              </button>
              
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md"
                title="Close"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6" aria-label="Tabs">
              <button
                onClick={() => setActiveTab('original')}
                className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'original'
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Original Document
              </button>
              <button
                onClick={() => setActiveTab('extracted')}
                className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'extracted'
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Extracted Text
              </button>
              <button
                onClick={() => setActiveTab('keywords')}
                className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'keywords'
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Keywords
                <span className="ml-1 text-xs text-gray-400">(Soon)</span>
              </button>
            </nav>
          </div>

          {/* Content */}
          <div className="px-6 py-4">
            {renderTabContent()}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div>
                Created: {document.created_at ? new Date(document.created_at).toLocaleDateString() : 'Unknown'}
              </div>
              <div>
                Status: <span className="capitalize">{document.status}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}