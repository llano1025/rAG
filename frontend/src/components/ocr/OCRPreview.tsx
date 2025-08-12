import { useState, useRef, useEffect } from 'react';
import { 
  EyeIcon, 
  ClipboardDocumentIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { apiClient } from '@/api/client';
import toast from 'react-hot-toast';

interface OCRResult {
  extracted_text: string;
  confidence_score?: number;
  method_used: string;
  processing_time_ms: number;
  language_detected?: string;
  metadata: {
    file_name: string;
    file_size: number;
    content_type: string;
  };
}

interface OCRPreviewProps {
  file: File;
  ocrMethod: string;
  ocrLanguage: string;
  visionProvider?: string;
  onClose: () => void;
  onAccept?: (result: OCRResult) => void;
}

export default function OCRPreview({
  file,
  ocrMethod,
  ocrLanguage,
  visionProvider,
  onClose,
  onAccept
}: OCRPreviewProps) {
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate image preview
  useEffect(() => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [file]);

  const processOCR = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', ocrMethod);
      formData.append('language', ocrLanguage);
      if (visionProvider) {
        formData.append('vision_provider', visionProvider);
      }
      formData.append('return_confidence', 'true');

      const result = await apiClient.post('/api/ocr/preview', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setOcrResult(result as OCRResult);
      toast.success('OCR processing completed!');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'OCR processing failed';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (!ocrResult?.extracted_text) return;

    try {
      await navigator.clipboard.writeText(ocrResult.extracted_text);
      toast.success('Text copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy text');
    }
  };

  const getConfidenceColor = (score?: number) => {
    if (!score) return 'text-gray-500';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceLabel = (score?: number) => {
    if (!score) return 'Unknown';
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center">
              <EyeIcon className="h-6 w-6 mr-2" />
              OCR Preview
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="mt-2 text-sm text-gray-600">
            <p>File: <span className="font-medium">{file.name}</span></p>
            <p>Method: <span className="font-medium">{ocrMethod}</span> | Language: <span className="font-medium">{ocrLanguage}</span>
              {visionProvider && ocrMethod === 'vision_llm' && (
                <> | Provider: <span className="font-medium">{visionProvider}</span></>
              )}
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {/* Image Preview */}
          <div className="w-1/2 p-6 border-r border-gray-200">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Original Image</h3>
            {imagePreview ? (
              <div className="bg-gray-50 rounded-lg p-4 h-full overflow-auto">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="max-w-full h-auto mx-auto rounded-lg shadow-sm"
                />
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 h-full flex items-center justify-center">
                <div className="text-center text-gray-500">
                  <svg className="h-12 w-12 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p>Image preview not available</p>
                </div>
              </div>
            )}
          </div>

          {/* OCR Results */}
          <div className="w-1/2 p-6 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Extracted Text</h3>
              {!ocrResult && !loading && (
                <button
                  onClick={processOCR}
                  className="btn-primary flex items-center"
                >
                  <ArrowPathIcon className="h-4 w-4 mr-2" />
                  Process OCR
                </button>
              )}
            </div>

            {loading && (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Processing image...</p>
                  <p className="text-sm text-gray-500 mt-1">This may take a few moments</p>
                </div>
              </div>
            )}

            {error && (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
                  <p className="text-red-600 font-medium">OCR Processing Failed</p>
                  <p className="text-sm text-gray-600 mt-1">{error}</p>
                  <button
                    onClick={processOCR}
                    className="mt-4 btn-secondary flex items-center mx-auto"
                  >
                    <ArrowPathIcon className="h-4 w-4 mr-2" />
                    Try Again
                  </button>
                </div>
              </div>
            )}

            {ocrResult && (
              <div className="flex-1 flex flex-col">
                {/* Results Info */}
                <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Confidence:</span>
                      <span className={`ml-2 font-medium ${getConfidenceColor(ocrResult.confidence_score)}`}>
                        {ocrResult.confidence_score ? 
                          `${Math.round(ocrResult.confidence_score * 100)}% (${getConfidenceLabel(ocrResult.confidence_score)})` : 
                          'N/A'
                        }
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Processing Time:</span>
                      <span className="ml-2 font-medium">
                        {Math.round(ocrResult.processing_time_ms)}ms
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Method:</span>
                      <span className="ml-2 font-medium">{ocrResult.method_used}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Text Length:</span>
                      <span className="ml-2 font-medium">{ocrResult.extracted_text.length} chars</span>
                    </div>
                  </div>
                </div>

                {/* Extracted Text */}
                <div className="flex-1 flex flex-col min-h-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">Extracted Text:</span>
                    <button
                      onClick={copyToClipboard}
                      className="btn-secondary btn-sm flex items-center"
                    >
                      <ClipboardDocumentIcon className="h-4 w-4 mr-1" />
                      Copy
                    </button>
                  </div>
                  
                  <textarea
                    value={ocrResult.extracted_text}
                    readOnly
                    className="flex-1 w-full p-3 border border-gray-300 rounded-lg resize-none font-mono text-sm"
                    placeholder="Extracted text will appear here..."
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Cancel
          </button>
          {ocrResult && onAccept && (
            <button
              onClick={() => onAccept(ocrResult)}
              className="btn-primary flex items-center"
            >
              <CheckCircleIcon className="h-4 w-4 mr-2" />
              Use This Text
            </button>
          )}
        </div>
      </div>
    </div>
  );
}