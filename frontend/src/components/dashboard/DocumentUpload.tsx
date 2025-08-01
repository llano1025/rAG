import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, XMarkIcon, FolderIcon, CpuChipIcon, EyeIcon } from '@heroicons/react/24/outline';
import { documentsApi } from '@/api/documents';
import { apiClient } from '@/api/client';
import TagInput from '@/components/common/TagInput';
import OCRSettings from '@/components/ocr/OCRSettings';
import OCRPreview from '@/components/ocr/OCRPreview';
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

interface EmbeddingModel {
  id: string;
  name: string;
  display_name: string;
  provider: string;
  description: string;
  embedding_dimension?: number;
  performance_tier?: string;
  quality_score?: number;
}

export default function DocumentUpload({ onUploadComplete }: DocumentUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [folderPath, setFolderPath] = useState<string>('');
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string>('hf-minilm-l6-v2'); // Default to fast model
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  
  // OCR settings
  const [ocrMethod, setOcrMethod] = useState<string>('tesseract');
  const [ocrLanguage, setOcrLanguage] = useState<string>('eng');
  const [showOcrSettings, setShowOcrSettings] = useState(false);
  const [previewFile, setPreviewFile] = useState<File | null>(null);

  // Load embedding models on component mount
  useEffect(() => {
    const loadEmbeddingModels = async () => {
      try {
        const data: any = await apiClient.get('/api/chat/models');
        setEmbeddingModels(data.embedding_models || []);
      } catch (error) {
        console.error('Failed to load embedding models:', error);
        toast.error('Failed to load embedding models');
      }
    };
    
    loadEmbeddingModels();
  }, []);

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
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif'],
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

        // Add OCR parameters for image files
        const uploadParams: any = { 
          file: uploadFile.file,
          metadata: { folder_path: folderPath || undefined },
          embedding_model: selectedEmbeddingModel
        };
        
        // Add OCR settings for image files
        if (uploadFile.file.type.startsWith('image/')) {
          uploadParams.ocr_method = ocrMethod;
          uploadParams.ocr_language = ocrLanguage;
        }

        await documentsApi.uploadDocument(
          uploadParams,
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

        // Check if any files are images to determine if OCR params are needed
        const hasImages = fileList.some(file => file.type.startsWith('image/'));
        
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
          folderPath ? { folder_path: folderPath } : undefined,
          selectedEmbeddingModel,
          hasImages ? ocrMethod : undefined,
          hasImages ? ocrLanguage : undefined
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
          Supports PDF, Word, Text, HTML, and Images (PNG, JPG, GIF, TIFF) - max 50MB each
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

        {/* Advanced options toggle */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            className="flex items-center text-sm text-blue-600 hover:text-blue-800"
          >
            <CpuChipIcon className="h-4 w-4 mr-1" />
            {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
          </button>
        </div>

        {/* Embedding model selector */}
        {showAdvancedOptions && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Embedding Model
            </label>
            <select
              value={selectedEmbeddingModel}
              onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {embeddingModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.display_name || model.name} - {model.provider} 
                  {model.embedding_dimension && ` (${model.embedding_dimension}D)`}
                  {model.performance_tier && ` - ${model.performance_tier}`}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Choose the embedding model for vector search. Different models offer various trade-offs between speed, quality, and resource usage.
            </p>
            {embeddingModels.find(m => m.id === selectedEmbeddingModel) && (
              <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-600">
                {embeddingModels.find(m => m.id === selectedEmbeddingModel)?.description}
              </div>
            )}
          </div>
        )}

        {/* OCR Settings for images */}
        {showAdvancedOptions && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                OCR Settings (for images)
              </label>
              <button
                type="button"
                onClick={() => setShowOcrSettings(!showOcrSettings)}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                {showOcrSettings ? 'Hide' : 'Configure'}
              </button>
            </div>
            
            {showOcrSettings && (
              <OCRSettings
                selectedMethod={ocrMethod}
                selectedLanguage={ocrLanguage}
                onMethodChange={setOcrMethod}
                onLanguageChange={setOcrLanguage}
                showAdvanced={true}
              />
            )}

            {!showOcrSettings && (
              <div className="text-sm text-gray-600 p-3 bg-gray-50 rounded-lg">
                <p>Method: <span className="font-medium">{ocrMethod}</span></p>
                <p>Language: <span className="font-medium">{ocrLanguage}</span></p>
                <p className="text-xs text-gray-500 mt-1">
                  These settings will be applied to uploaded image files for text extraction.
                </p>
              </div>
            )}
          </div>
        )}
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
              
              <div className="flex items-center space-x-2">
                {/* OCR Preview button for images */}
                {uploadFile.status === 'pending' && uploadFile.file.type.startsWith('image/') && (
                  <button
                    onClick={() => setPreviewFile(uploadFile.file)}
                    className="p-1 text-blue-500 hover:text-blue-700 transition-colors"
                    title="Preview OCR"
                  >
                    <EyeIcon className="h-4 w-4" />
                  </button>
                )}
                
                {uploadFile.status === 'pending' && (
                  <button
                    onClick={() => removeFile(uploadFile.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                )}
              </div>
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
      
      {/* OCR Preview Modal */}
      {previewFile && (
        <OCRPreview
          file={previewFile}
          ocrMethod={ocrMethod}
          ocrLanguage={ocrLanguage}
          onClose={() => setPreviewFile(null)}
          onAccept={(result) => {
            // Handle OCR result if needed
            setPreviewFile(null);
          }}
        />
      )}
    </div>
  );
}