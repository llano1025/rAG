import { apiClient } from './client';
import { Document, DocumentUpload, PaginatedResponse } from '@/types';

// Backend response structure for documents
interface DocumentsResponse {
  documents: Document[];
  total_count: number;
  skip: number;
  limit: number;
  filters: Record<string, any>;
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

    return apiClient.upload('/api/documents/upload', formData, onProgress);
  },

  uploadBatchDocuments: async (
    files: File[],
    onProgress?: (progress: number) => void,
    tags?: string[],
    metadata?: Record<string, any>,
    embeddingModel?: string,
    ocrMethod?: string,
    ocrLanguage?: string
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

    return apiClient.upload('/api/documents/batch-upload', formData, onProgress);
  },

  updateDocument: async (id: string, updates: Partial<Document>): Promise<Document> => {
    return apiClient.put(`/api/documents/${id}`, updates);
  },

  deleteDocument: async (id: string): Promise<void> => {
    return apiClient.delete(`/api/documents/${id}`);
  },

  downloadDocument: async (id: string): Promise<Blob> => {
    return apiClient.get(`/api/documents/${id}/download`, {
      responseType: 'blob',
    });
  },

  shareDocument: async (id: string, permissions: string[]): Promise<void> => {
    return apiClient.post(`/api/documents/${id}/share`, { permissions });
  },

  getSharedDocuments: async (): Promise<Document[]> => {
    return apiClient.get('/api/documents/shared');
  },
};