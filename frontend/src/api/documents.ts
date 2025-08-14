import { apiClient } from './client';
import { Document, DocumentUpload, PaginatedResponse } from '@/types';

// Backend response structure for documents (after API client transformation)
interface DocumentsResponse {
  documents: Document[];
  total_count: number;
  skip: number;
  limit: number;
  filters: Record<string, any>;
  // Pagination metadata from StandardResponse
  current_page?: number;
  page_size?: number;
  total_items?: number;
  total_pages?: number;
  has_next?: boolean;
  has_previous?: boolean;
  next_page?: number | null;
  previous_page?: number | null;
}

export const documentsApi = {
  getDocuments: async (page = 1, perPage = 20): Promise<DocumentsResponse> => {
    return apiClient.get(`/api/documents?page=${page}&per_page=${perPage}`);
  },

  getDocument: async (id: string): Promise<Document> => {
    return apiClient.get(`/api/documents/${id}`);
  },

  uploadDocument: async (
    upload: DocumentUpload,
    onProgress?: (progress: number) => void,
    tags?: string[]
  ): Promise<Document> => {
    const formData = new FormData();
    formData.append('file', upload.file);
    
    if (upload.metadata) {
      formData.append('metadata', JSON.stringify(upload.metadata));
    }

    if (tags && tags.length > 0) {
      formData.append('tags', JSON.stringify(tags));
    }

    if (upload.embedding_model) {
      formData.append('embedding_model', upload.embedding_model);
    }

    // Add OCR parameters for image files
    if ((upload as any).ocr_method) {
      formData.append('ocr_method', (upload as any).ocr_method);
    }

    if ((upload as any).ocr_language) {
      formData.append('ocr_language', (upload as any).ocr_language);
    }

    if ((upload as any).vision_provider) {
      formData.append('vision_provider', (upload as any).vision_provider);
    }

    return apiClient.upload('/api/documents/upload', formData, onProgress);
  },

  uploadBatchDocuments: async (
    files: File[],
    onProgress?: (progress: number) => void,
    tags?: string[],
    metadata?: Record<string, any>,
    embeddingModel?: string,
    ocrMethod?: string,
    ocrLanguage?: string,
    visionProvider?: string
  ): Promise<Document[]> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    if (tags && tags.length > 0) {
      formData.append('tags', JSON.stringify(tags));
    }

    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    if (embeddingModel) {
      formData.append('embedding_model', embeddingModel);
    }

    // Add OCR parameters for image files
    if (ocrMethod) {
      formData.append('ocr_method', ocrMethod);
    }

    if (ocrLanguage) {
      formData.append('ocr_language', ocrLanguage);
    }

    if (visionProvider) {
      formData.append('vision_provider', visionProvider);
    }

    return apiClient.upload('/api/documents/batch-upload', formData, onProgress);
  },

  updateDocument: async (id: string, updates: Partial<Document>): Promise<Document> => {
    return apiClient.put(`/api/documents/${id}`, updates);
  },

  deleteDocument: async (id: string): Promise<void> => {
    return apiClient.delete(`/api/documents/${id}`);
  },

  downloadDocument: async (id: string): Promise<Blob> => {
    return apiClient.downloadBlob(`/api/documents/${id}/download`);
  },

  getDocumentContent: async (id: string): Promise<{
    document_id: number;
    filename: string;
    extracted_text: string;
    text_length: number;
    chunks_count: number;
    language: string;
    content_type: string;
  }> => {
    return apiClient.get(`/api/documents/${id}/content`);
  },

  downloadExtractedText: async (id: string): Promise<Blob> => {
    return apiClient.downloadBlob(`/api/documents/${id}/download?format=text`);
  },

  shareDocument: async (id: string, permissions: string[]): Promise<void> => {
    return apiClient.post(`/api/documents/${id}/share`, { permissions });
  },

  getSharedDocuments: async (): Promise<Document[]> => {
    return apiClient.get('/api/documents/shared');
  },
};