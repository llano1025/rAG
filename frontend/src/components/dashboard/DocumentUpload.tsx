import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, XMarkIcon, FolderIcon } from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import TagInput from '@/components/common/TagInput';
import toast from 'react-hot-toast';

interface UploadFile {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  id: string;
}

interface DocumentUploadProps {
  onUploadComplete?: () => void;
}

export default function DocumentUpload({ onUploadComplete }: DocumentUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [folderPath, setFolderPath] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      progress: 0,
      status: 'pending' as const,
      id: Math.random().toString(36).substr(2, 9),
    }));
    
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/html': ['.html'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const uploadFiles = async () => {
    setIsUploading(true);
    
    try {
      const pendingFiles = files.filter(f => f.status === 'pending');
      
      if (pendingFiles.length === 1) {
        // Single file upload
        const uploadFile = pendingFiles[0];
        setFiles(prev => prev.map(f => 
          f.id === uploadFile.id ? { ...f, status: 'uploading' } : f
        ));

        await documentsApi.uploadDocument(
          { 
            file: uploadFile.file,
            metadata: { folder_path: folderPath || undefined }
          },
          (progress) => {
            setFiles(prev => prev.map(f => 
              f.id === uploadFile.id ? { ...f, progress } : f
            ));
          },
          tags.length > 0 ? tags : undefined
        );

        setFiles(prev => prev.map(f => 
          f.id === uploadFile.id ? { ...f, status: 'completed', progress: 100 } : f
        ));
      } else if (pendingFiles.length > 1) {
        // Batch upload
        const fileList = pendingFiles.map(f => f.file);
        
        pendingFiles.forEach(uploadFile => {
          setFiles(prev => prev.map(f => 
            f.id === uploadFile.id ? { ...f, status: 'uploading' } : f
          ));
        });

        await documentsApi.uploadBatchDocuments(
          fileList,
          (progress) => {
            pendingFiles.forEach(uploadFile => {
              setFiles(prev => prev.map(f => 
                f.id === uploadFile.id ? { ...f, progress } : f
              ));
            });
          },
          tags.length > 0 ? tags : undefined,
          folderPath ? { folder_path: folderPath } : undefined
        );

        pendingFiles.forEach(uploadFile => {
          setFiles(prev => prev.map(f => 
            f.id === uploadFile.id ? { ...f, status: 'completed', progress: 100 } : f
          ));
        });
      }

      toast.success(`Successfully uploaded ${pendingFiles.length} file(s)`);
      onUploadComplete?.();
      
      // Clear completed files after 2 seconds
      setTimeout(() => {
        setFiles(prev => prev.filter(f => f.status !== 'completed'));
      }, 2000);
      
    } catch (error: any) {
      // Handle cancellation gracefully
      if (error.code === 'ERR_CANCELED' || error.name === 'AbortError') {
        // Mark cancelled files as pending so they can be retried
        setFiles(prev => prev.map(f => 
          f.status === 'uploading' ? { ...f, status: 'pending', progress: 0 } : f
        ));
        return;
      }
      
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      toast.error('Upload failed: ' + errorMessage);
      
      // Mark failed files
      setFiles(prev => prev.map(f => 
        f.status === 'uploading' ? { ...f, status: 'failed' } : f
      ));
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">
          {isDragActive
            ? 'Drop files here...'
            : 'Drag & drop files here, or click to browse'}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Supports PDF, Word, Text, HTML, and Images (max 50MB each)
        </p>
      </div>

      {/* Upload options */}
      <div className="space-y-4">
        {/* Tags input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tags (optional)
          </label>
          <TagInput
            value={tags}
            onChange={setTags}
            placeholder="Add tags to organize your documents..."
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Press Enter or comma to add tags. Tags help organize and find your documents.
          </p>
        </div>

        {/* Folder input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Folder (optional)
          </label>
          <div className="relative">
            <FolderIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
              placeholder="e.g., projects/research/ai"
              className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Use forward slashes to create nested folders (e.g., "work/projects/2024").
          </p>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Files to upload:</h4>
          {files.map((uploadFile) => (
            <div key={uploadFile.id} className="flex items-center justify-between p-3 bg-white rounded-lg border">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {uploadFile.file.name}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(uploadFile.file.size)}
                </p>
                
                {/* Progress bar */}
                {uploadFile.status === 'uploading' && (
                  <div className="mt-2">
                    <div className="bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadFile.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{uploadFile.progress}%</p>
                  </div>
                )}
                
                {/* Status indicators */}
                {uploadFile.status === 'completed' && (
                  <p className="text-xs text-green-600 mt-1">✓ Uploaded successfully</p>
                )}
                {uploadFile.status === 'failed' && (
                  <p className="text-xs text-red-600 mt-1">✗ Upload failed</p>
                )}
              </div>
              
              {uploadFile.status === 'pending' && (
                <button
                  onClick={() => removeFile(uploadFile.id)}
                  className="ml-2 p-1 text-gray-400 hover:text-gray-600"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}
          
          {/* Upload button */}
          {files.some(f => f.status === 'pending') && (
            <button
              onClick={uploadFiles}
              disabled={isUploading}
              className="w-full btn-primary disabled:opacity-50"
            >
              {isUploading ? 'Uploading...' : `Upload ${files.filter(f => f.status === 'pending').length} file(s)`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}