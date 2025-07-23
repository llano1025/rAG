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
    return apiClient.get(`/documents?page=${page}&per_page=${perPage}`);
  },

  getDocument: async (id: string): Promise<Document> => {
    return apiClient.get(`/documents/${id}`);
  },

  uploadDocument: async (
    upload: DocumentUpload,
    onProgress?: (progress: number) => void
  ): Promise<Document> => {
    const formData = new FormData();
    formData.append('file', upload.file);
    
    if (upload.metadata) {
      formData.append('metadata', JSON.stringify(upload.metadata));
    }

    return apiClient.upload('/documents/upload', formData, onProgress);
  },

  uploadBatchDocuments: async (
    files: File[],
    onProgress?: (progress: number) => void
  ): Promise<Document[]> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    return apiClient.upload('/documents/batch-upload', formData, onProgress);
  },

  updateDocument: async (id: string, updates: Partial<Document>): Promise<Document> => {
    return apiClient.put(`/documents/${id}`, updates);
  },

  deleteDocument: async (id: string): Promise<void> => {
    return apiClient.delete(`/documents/${id}`);
  },

  downloadDocument: async (id: string): Promise<Blob> => {
    return apiClient.get(`/documents/${id}/download`, {
      responseType: 'blob',
    });
  },

  shareDocument: async (id: string, permissions: string[]): Promise<void> => {
    return apiClient.post(`/documents/${id}/share`, { permissions });
  },

  getSharedDocuments: async (): Promise<Document[]> => {
    return apiClient.get('/documents/shared');
  },
};