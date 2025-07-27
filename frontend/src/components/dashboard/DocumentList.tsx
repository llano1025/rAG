import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import {
  DocumentTextIcon,
  EyeIcon,
  TrashIcon,
  ShareIcon,
  ArrowDownTrayIcon,
  FunnelIcon,
  TagIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import { libraryApi, Tag } from '@/api/library';
import { Document, PaginatedResponse } from '@/types';
import DocumentPreview from './DocumentPreview';
import TagInput from '@/components/common/TagInput';
import TagSelector from '@/components/common/TagSelector';
import toast from 'react-hot-toast';

interface DocumentListProps {
  refreshTrigger?: number;
}

// Backend response structure
interface DocumentsResponse {
  documents: Document[];
  total_count: number;
  skip: number;
  limit: number;
  filters: Record<string, any>;
}

export default function DocumentList({ refreshTrigger }: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [filterType, setFilterType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size'>('date');
  const [previewDocument, setPreviewDocument] = useState<Document | null>(null);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [availableTags, setAvailableTags] = useState<Tag[]>([]);
  
  // New state for tag editing and bulk operations
  const [editingTags, setEditingTags] = useState<{ [key: string]: boolean }>({});
  const [editingTagValues, setEditingTagValues] = useState<{ [key: string]: string[] }>({});
  const [selectedDocuments, setSelectedDocuments] = useState<number[]>([]);
  const [showBulkTagSelector, setShowBulkTagSelector] = useState(false);
  const [bulkTagsToApply, setBulkTagsToApply] = useState<string[]>([]);

  useEffect(() => {
    fetchDocuments();
  }, [currentPage, refreshTrigger]);

  // Reset to first page when filters change
  useEffect(() => {
    if (currentPage !== 1) {
      setCurrentPage(1);
    }
  }, [filterType, tagFilter]);

  useEffect(() => {
    const loadTags = async () => {
      try {
        const tags = await libraryApi.getTags();
        setAvailableTags(tags);
      } catch (error) {
        console.error('Failed to load tags:', error);
      }
    };
    loadTags();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await documentsApi.getDocuments(currentPage, 20);
      setDocuments(response as DocumentsResponse);
    } catch (error: any) {
      // Handle different error types
      if (error.code === 'ERR_CANCELED' || error.name === 'AbortError') {
        // Request was cancelled, don't show error
        return;
      }
      
      const errorMessage = error.response?.data?.detail || 'Failed to fetch documents';
      toast.error(errorMessage);
      
      // Set empty state on error to prevent crashes
      setDocuments({
        documents: [],
        total_count: 0,
        skip: 0,
        limit: 20,
        filters: {}
      });
    } finally {
      setLoading(false);
    }
  };

  // Filter documents based on current filters
  const filteredDocuments = documents?.documents?.filter(doc => {
    // Filter by file type
    if (filterType !== 'all') {
      const docType = doc.content_type.toLowerCase();
      if (filterType === 'pdf' && !docType.includes('pdf')) return false;
      if (filterType === 'docx' && !docType.includes('word')) return false;
      if (filterType === 'txt' && !docType.includes('text')) return false;
      if (filterType === 'html' && !docType.includes('html')) return false;
    }

    // Filter by tags
    if (tagFilter.length > 0) {
      if (!doc.tags || doc.tags.length === 0) return false;
      const hasMatchingTag = tagFilter.some(filterTag => 
        doc.tags.some(docTag => docTag.toLowerCase().includes(filterTag.toLowerCase()))
      );
      if (!hasMatchingTag) return false;
    }

    return true;
  }) || [];

  // Sort filtered documents
  const sortedDocuments = [...filteredDocuments].sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.filename.localeCompare(b.filename);
      case 'size':
        return b.file_size - a.file_size;
      case 'date':
      default:
        const dateA = new Date(a.created_at || 0).getTime();
        const dateB = new Date(b.created_at || 0).getTime();
        return dateB - dateA;
    }
  });

  const handleDelete = async (id: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      await documentsApi.deleteDocument(id);
      toast.success('Document deleted successfully');
      fetchDocuments();
    } catch (error: any) {
      toast.error('Failed to delete document');
    }
  };

  const handleDownload = async (id: string, filename: string) => {
    try {
      const blob = await documentsApi.downloadDocument(id);
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = filename;
      window.document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      window.document.body.removeChild(a);
    } catch (error: any) {
      toast.error('Failed to download document');
    }
  };

  const handleShare = async (id: string) => {
    try {
      await documentsApi.shareDocument(id, ['read']);
      toast.success('Document shared successfully');
      fetchDocuments();
    } catch (error: any) {
      toast.error('Failed to share document');
    }
  };

  // Tag editing functions
  const startEditingTags = (docId: string, currentTags: string[]) => {
    setEditingTags({ ...editingTags, [docId]: true });
    setEditingTagValues({ ...editingTagValues, [docId]: [...currentTags] });
  };

  const cancelEditingTags = (docId: string) => {
    setEditingTags({ ...editingTags, [docId]: false });
    setEditingTagValues({ ...editingTagValues, [docId]: [] });
  };

  const saveDocumentTags = async (docId: string) => {
    try {
      const newTags = editingTagValues[docId] || [];
      
      // Update document tags using the documents API
      await documentsApi.updateDocument(docId, { tags: newTags });
      
      // Update local state
      if (documents?.documents) {
        const updatedDocuments = documents.documents.map(doc =>
          doc.id === docId ? { ...doc, tags: newTags } : doc
        );
        setDocuments({ ...documents, documents: updatedDocuments });
      }
      
      setEditingTags({ ...editingTags, [docId]: false });
      toast.success('Tags updated successfully');
    } catch (error: any) {
      toast.error('Failed to update tags');
      console.error('Tag update error:', error);
    }
  };

  // Bulk operations
  const toggleDocumentSelection = (docId: number) => {
    setSelectedDocuments(prev => 
      prev.includes(docId) 
        ? prev.filter(id => id !== docId)
        : [...prev, docId]
    );
  };

  const selectAllDocuments = () => {
    const allIds = sortedDocuments.map(doc => parseInt(doc.id));
    setSelectedDocuments(allIds);
  };

  const clearSelection = () => {
    setSelectedDocuments([]);
  };

  const handleBulkTagApply = async (tags: string[], documentIds: number[]) => {
    try {
      await libraryApi.applyTagsToDocuments(documentIds, tags);
      
      // Refresh documents to show updated tags
      fetchDocuments();
      
      // Clear selection
      setSelectedDocuments([]);
      setShowBulkTagSelector(false);
      
      toast.success(`Applied tags to ${documentIds.length} documents`);
    } catch (error: any) {
      toast.error('Failed to apply tags to documents');
      console.error('Bulk tag apply error:', error);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedDocuments.length === 0) return;
    
    if (!confirm(`Are you sure you want to delete ${selectedDocuments.length} selected documents?`)) {
      return;
    }

    try {
      // Delete documents one by one (could be optimized with batch API)
      for (const docId of selectedDocuments) {
        await documentsApi.deleteDocument(docId.toString());
      }
      
      toast.success(`Deleted ${selectedDocuments.length} documents`);
      setSelectedDocuments([]);
      fetchDocuments();
    } catch (error: any) {
      toast.error('Failed to delete selected documents');
    }
  };

  const getFileIcon = (fileType: string) => {
    return <DocumentTextIcon className="h-8 w-8 text-gray-400" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getStatusBadge = (status: string) => {
    const statusStyles = {
      completed: 'bg-green-100 text-green-800',
      processing: 'bg-yellow-100 text-yellow-800',
      failed: 'bg-red-100 text-red-800',
      uploading: 'bg-blue-100 text-blue-800',
    };

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
        statusStyles[status as keyof typeof statusStyles] || 'bg-gray-100 text-gray-800'
      }`}>
        {status}
      </span>
    );
  };

  const renderTags = (doc: Document) => {
    const { tags = [], id } = doc;
    const isEditing = editingTags[id];
    const editingValues = editingTagValues[id] || tags;

    return (
      <div className="flex flex-wrap items-center gap-1 mt-1">
        <TagIcon className="h-3 w-3 text-gray-400" />
        
        {isEditing ? (
          <div className="flex items-center gap-2 flex-1">
            <TagInput
              value={editingValues}
              onChange={(newTags) => setEditingTagValues({ ...editingTagValues, [id]: newTags })}
              placeholder="Edit tags..."
              className="flex-1 min-w-0"
              maxTags={10}
            />
            <div className="flex items-center gap-1">
              <button
                onClick={() => saveDocumentTags(id)}
                className="p-1 text-green-600 hover:text-green-700"
                title="Save tags"
              >
                <CheckIcon className="h-4 w-4" />
              </button>
              <button
                onClick={() => cancelEditingTags(id)}
                className="p-1 text-gray-400 hover:text-gray-600"
                title="Cancel"
              >
                <XMarkIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        ) : (
          <>
            {tags.length > 0 ? (
              <>
                {tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700"
                  >
                    {tag}
                  </span>
                ))}
                {tags.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{tags.length - 3} more
                  </span>
                )}
              </>
            ) : (
              <span className="text-xs text-gray-400 italic">No tags</span>
            )}
            <button
              onClick={() => startEditingTags(id, tags)}
              className="p-1 text-gray-400 hover:text-gray-600"
              title="Edit tags"
            >
              <PencilIcon className="h-3 w-3" />
            </button>
          </>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-white p-4 rounded-lg border">
              <div className="flex items-center space-x-4">
                <div className="w-8 h-8 bg-gray-200 rounded"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/4"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!documents || documents.documents.length === 0) {
    return (
      <div className="text-center py-12">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No documents</h3>
        <p className="mt-1 text-sm text-gray-500">
          Upload your first document to get started.
        </p>
      </div>
    );
  }

  // Show filtered empty state
  if (sortedDocuments.length === 0) {
    return (
      <div className="text-center py-12">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No matching documents</h3>
        <p className="mt-1 text-sm text-gray-500">
          Try adjusting your filters to see more documents.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Bulk Actions Bar */}
      {selectedDocuments.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-blue-900">
                {selectedDocuments.length} document{selectedDocuments.length !== 1 ? 's' : ''} selected
              </span>
              <button
                onClick={clearSelection}
                className="text-sm text-blue-700 hover:text-blue-800"
              >
                Clear selection
              </button>
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowBulkTagSelector(!showBulkTagSelector)}
                className="flex items-center space-x-2 px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
              >
                <TagIcon className="h-4 w-4" />
                <span>Edit Tags</span>
              </button>
              <button
                onClick={handleBulkDelete}
                className="flex items-center space-x-2 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
              >
                <TrashIcon className="h-4 w-4" />
                <span>Delete</span>
              </button>
            </div>
          </div>
          
          {/* Bulk Tag Selector */}
          {showBulkTagSelector && (
            <div className="mt-4 border-t border-blue-200 pt-4">
              <TagSelector
                value={bulkTagsToApply}
                onChange={setBulkTagsToApply}
                placeholder="Select tags to apply to selected documents..."
                mode="dropdown"
                showBulkActions={true}
                selectedDocuments={selectedDocuments}
                onBulkApply={handleBulkTagApply}
                className="max-w-md"
              />
            </div>
          )}
        </div>
      )}

      {/* Filters and Sort */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* Bulk select controls */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedDocuments.length === sortedDocuments.length && sortedDocuments.length > 0}
                onChange={(e) => e.target.checked ? selectAllDocuments() : clearSelection()}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-500">Select all</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <FunnelIcon className="h-5 w-5 text-gray-400" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm"
              >
                <option value="all">All Types</option>
                <option value="pdf">PDF</option>
                <option value="docx">Word</option>
                <option value="txt">Text</option>
                <option value="html">HTML</option>
              </select>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500">Sort by:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'name' | 'date' | 'size')}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm"
              >
                <option value="date">Date</option>
                <option value="name">Name</option>
                <option value="size">Size</option>
              </select>
            </div>
          </div>
          
          <p className="text-sm text-gray-500">
            {sortedDocuments.length} of {documents ? documents.total_count : 0} document{documents && documents.total_count !== 1 ? 's' : ''}
          </p>
        </div>
        
        {/* Tag filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Filter by tags:
          </label>
          <TagInput
            value={tagFilter}
            onChange={setTagFilter}
            placeholder="Filter documents by tags..."
            className="max-w-md"
            maxTags={10}
          />
        </div>
      </div>

      {/* Document Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sortedDocuments.map((doc) => {
          const docId = parseInt(doc.id);
          const isSelected = selectedDocuments.includes(docId);
          
          return (
            <div 
              key={doc.id} 
              className={`bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow ${
                isSelected ? 'ring-2 ring-blue-500 bg-blue-50' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1 min-w-0">
                  {/* Selection checkbox */}
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleDocumentSelection(docId)}
                    className="mt-1 rounded border-gray-300"
                  />
                  
                  {getFileIcon(doc.content_type)}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 truncate" title={doc.filename}>
                      {doc.filename}
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {formatFileSize(doc.file_size)} • {doc.created_at ? format(new Date(doc.created_at), 'MMM d, yyyy') : 'Unknown date'}
                    </p>
                    <div className="mt-2">
                      {getStatusBadge(doc.status)}
                    </div>
                    {renderTags(doc)}
                  </div>
                </div>
              </div>
              
              {/* Actions */}
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleDownload(doc.id, doc.filename)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                    title="Download"
                  >
                    <ArrowDownTrayIcon className="h-4 w-4" />
                  </button>
                  
                  <button
                    onClick={() => handleShare(doc.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                    title="Share"
                  >
                    <ShareIcon className="h-4 w-4" />
                  </button>
                  
                  <button
                    onClick={() => setPreviewDocument(doc)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                    title="Preview"
                  >
                    <EyeIcon className="h-4 w-4" />
                  </button>
                </div>
                
                <button
                  onClick={() => handleDelete(doc.id, doc.filename)}
                  className="p-1 text-gray-400 hover:text-red-600"
                  title="Delete"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {documents.total_count > documents.limit && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-700">
            Showing {documents.skip + 1} to {Math.min(documents.skip + documents.limit, documents.total_count)} of {documents.total_count}
          </p>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={currentPage <= 1}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              Previous
            </button>
            
            <span className="px-3 py-1 text-sm bg-primary-50 text-primary-700 rounded-md">
              {currentPage}
            </span>
            
            <button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={documents.skip + documents.limit >= documents.total_count}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Document Preview Modal */}
      {previewDocument && (
        <DocumentPreview
          document={previewDocument}
          isOpen={!!previewDocument}
          onClose={() => setPreviewDocument(null)}
        />
      )}
    </div>
  );
}