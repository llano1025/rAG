import { useState, useEffect } from 'react';
import { 
  CpuChipIcon, 
  EyeIcon, 
  LanguageIcon, 
  CogIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { apiClient } from '@/api/client';
import toast from 'react-hot-toast';

interface OCRMethod {
  method: string;
  display_name: string;
  description: string;
  supported_formats: string[];
  requires_api_key: boolean;
  estimated_cost?: string;
  quality_level: string;
}

interface OCRLanguage {
  code: string;
  name: string;
  supported_by: string[];
}

interface OCRConfig {
  ocr_enabled: boolean;
  default_language: string;
  tesseract_available: boolean;
  vision_llm_available: boolean;
  max_file_size_mb: number;
  supported_formats: string[];
  batch_limit: number;
}

interface OCRSettingsProps {
  selectedMethod: string;
  selectedLanguage: string;
  onMethodChange: (method: string) => void;
  onLanguageChange: (language: string) => void;
  showAdvanced?: boolean;
}

export default function OCRSettings({
  selectedMethod,
  selectedLanguage,
  onMethodChange,
  onLanguageChange,
  showAdvanced = false
}: OCRSettingsProps) {
  const [ocrMethods, setOcrMethods] = useState<OCRMethod[]>([]);
  const [ocrLanguages, setOcrLanguages] = useState<OCRLanguage[]>([]);
  const [ocrConfig, setOcrConfig] = useState<OCRConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    loadOCRSettings();
  }, []);

  const loadOCRSettings = async () => {
    try {
      setLoading(true);
      
      // Load OCR methods, languages, and config
      const [methodsResponse, languagesResponse, configResponse] = await Promise.all([
        apiClient.get('/api/ocr/methods'),
        apiClient.get('/api/ocr/languages'),
        apiClient.get('/api/ocr/config')
      ]);
      
      setOcrMethods(methodsResponse);
      setOcrLanguages(languagesResponse);
      setOcrConfig(configResponse);
      
    } catch (error) {
      console.error('Failed to load OCR settings:', error);
      toast.error('Failed to load OCR settings');
    } finally {
      setLoading(false);
    }
  };

  const getQualityColor = (level: string) => {
    switch (level) {
      case 'excellent': return 'text-green-600 bg-green-50';
      case 'good': return 'text-blue-600 bg-blue-50';
      case 'basic': return 'text-gray-600 bg-gray-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getQualityIcon = (level: string) => {
    switch (level) {
      case 'excellent': return '🚀';
      case 'good': return '👍';
      case 'basic': return '📄';
      default: return '❓';
    }
  };

  if (loading) {
    return (
      <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-10 bg-gray-200 rounded"></div>
            <div className="h-10 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* OCR Method Selection */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            <CpuChipIcon className="h-5 w-5 mr-2" />
            OCR Method
          </h3>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-sm text-gray-500 hover:text-gray-700 flex items-center"
          >
            <InformationCircleIcon className="h-4 w-4 mr-1" />
            Details
          </button>
        </div>

        <div className="space-y-3">
          {ocrMethods.map((method) => (
            <div key={method.method} className="relative">
              <label className="flex items-start p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                <input
                  type="radio"
                  name="ocrMethod"
                  value={method.method}
                  checked={selectedMethod === method.method}
                  onChange={(e) => onMethodChange(e.target.value)}
                  className="mt-1 mr-3"
                  disabled={method.requires_api_key && !ocrConfig?.vision_llm_available}
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-gray-900 flex items-center">
                      {method.display_name}
                      <span className={`ml-2 px-2 py-1 text-xs rounded-full ${getQualityColor(method.quality_level)}`}>
                        {getQualityIcon(method.quality_level)} {method.quality_level}
                      </span>
                    </h4>
                    {method.estimated_cost && (
                      <span className="text-sm text-gray-500">{method.estimated_cost}</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{method.description}</p>
                  
                  {showDetails && (
                    <div className="mt-2 text-xs text-gray-500">
                      <p>Supported formats: {method.supported_formats.join(', ')}</p>
                      {method.requires_api_key && (
                        <p className="text-amber-600 mt-1">
                          ⚠️ Requires API key configuration
                          {!ocrConfig?.vision_llm_available && ' (not available)'}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* Language Selection */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 flex items-center mb-4">
          <LanguageIcon className="h-5 w-5 mr-2" />
          OCR Language
        </h3>

        <select
          value={selectedLanguage}
          onChange={(e) => onLanguageChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {ocrLanguages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} ({lang.code})
            </option>
          ))}
        </select>

        <p className="text-sm text-gray-500 mt-2">
          Select the primary language of your documents for better OCR accuracy.
        </p>
      </div>

      {/* Advanced Settings */}
      {showAdvanced && ocrConfig && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-lg font-medium text-gray-900 flex items-center mb-4">
            <CogIcon className="h-5 w-5 mr-2" />
            Advanced Settings
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max File Size
              </label>
              <div className="text-sm text-gray-600">
                {ocrConfig.max_file_size_mb} MB per file
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch Upload Limit
              </label>
              <div className="text-sm text-gray-600">
                {ocrConfig.batch_limit} files maximum
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Supported Formats
              </label>
              <div className="flex flex-wrap gap-2">
                {ocrConfig.supported_formats.map((format) => (
                  <span
                    key={format}
                    className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                  >
                    {format}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-start">
              <InformationCircleIcon className="h-5 w-5 text-blue-400 mt-0.5 mr-2 flex-shrink-0" />
              <div className="text-sm text-blue-700">
                <p className="font-medium mb-1">OCR Processing Tips:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li>Higher resolution images provide better OCR accuracy</li>
                  <li>Vision LLM works better with complex layouts and handwriting</li>
                  <li>Tesseract is faster and free for simple printed text</li>
                  <li>Ensure images are well-lit and text is clearly visible</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Current Status */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-2">Current Settings</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p>Method: <span className="font-medium">{selectedMethod}</span></p>
          <p>Language: <span className="font-medium">
            {ocrLanguages.find(l => l.code === selectedLanguage)?.name || selectedLanguage}
          </span></p>
          {ocrConfig && (
            <p>Status: <span className={`font-medium ${ocrConfig.ocr_enabled ? 'text-green-600' : 'text-red-600'}`}>
              {ocrConfig.ocr_enabled ? 'Enabled' : 'Disabled'}
            </span></p>
          )}
        </div>
      </div>
    </div>
  );
}