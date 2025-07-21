import { useState, useEffect } from 'react';
import { XMarkIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import { Document } from '@/types';

interface DocumentPreviewProps {
  document: Document;
  isOpen: boolean;
  onClose: () => void;
}

export default function DocumentPreview({ document, isOpen, onClose }: DocumentPreviewProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && document) {
      loadPreview();
    }
    
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
    };
  }, [isOpen, document]);

  const loadPreview = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const blob = await documentsApi.downloadDocument(document.id);
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
    } catch (err: any) {
      setError('Failed to load document preview');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const blob = await documentsApi.downloadDocument(document.id);
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = document.original_filename;
      window.document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      window.document.body.removeChild(a);
    } catch (error) {
      setError('Failed to download document');
    }
  };

  const renderPreview = () => {
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
          <div className="text-center">
            <p className="text-red-600 mb-4">{error}</p>
            <button
              onClick={loadPreview}
              className="btn-primary"
            >
              Retry
            </button>
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
    const fileType = document.file_type.toLowerCase();
    
    if (fileType === 'pdf') {
      return (
        <iframe
          src={previewUrl}
          className="w-full h-96 border-0"
          title="Document Preview"
        />
      );
    }
    
    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(fileType)) {
      return (
        <img
          src={previewUrl}
          alt="Document Preview"
          className="max-w-full h-auto mx-auto"
        />
      );
    }
    
    if (['txt', 'html', 'csv'].includes(fileType)) {
      return (
        <iframe
          src={previewUrl}
          className="w-full h-96 border border-gray-200 rounded"
          title="Document Preview"
        />
      );
    }

    // For unsupported file types, show download option
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-gray-500 mb-4">Preview not available for this file type</p>
          <button
            onClick={handleDownload}
            className="btn-primary"
          >
            Download to view
          </button>
        </div>
      </div>
    );
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
        <div className="inline-block w-full max-w-4xl my-8 overflow-hidden text-left align-middle transition-all transform bg-white shadow-xl rounded-lg">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-medium text-gray-900 truncate">
                {document.original_filename}
              </h3>
              <p className="text-sm text-gray-500">
                {document.file_type.toUpperCase()} • {(document.file_size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={handleDownload}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md"
                title="Download"
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

          {/* Content */}
          <div className="px-6 py-4">
            {renderPreview()}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div>
                Uploaded: {new Date(document.upload_date).toLocaleDateString()}
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