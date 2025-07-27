import { apiClient } from './client';

export interface Tag {
  tag_name: string;
  document_count: number;
  description?: string;
  color?: string;
}

export interface TagCreate {
  tag_name: string;
  description?: string;
  color?: string;
}

export interface Folder {
  folder_path: string;
  document_count: number;
  last_updated: string | null;
}

export interface LibraryStats {
  total_documents: number;
  total_folders: number;
  total_tags: number;
  folder_breakdown: Record<string, number>;
  tag_breakdown: Record<string, number>;
}

export interface BulkTagResult {
  successful: number;
  failed: number;
  errors: string[];
  message: string;
}

export const libraryApi = {
  // Tag Management
  getTags: async (): Promise<Tag[]> => {
    return apiClient.get('/api/library/tags');
  },

  createTag: async (tagData: TagCreate): Promise<Tag> => {
    return apiClient.post('/api/library/tags', tagData);
  },

  updateTag: async (tagId: string, tagData: Partial<TagCreate>): Promise<Tag> => {
    return apiClient.put(`/api/library/tags/${tagId}`, tagData);
  },

  deleteTag: async (tagId: string): Promise<void> => {
    return apiClient.delete(`/api/library/tags/${tagId}`);
  },

  // Bulk Tag Operations
  bulkAddTags: async (documentIds: number[], tags: string[]): Promise<BulkTagResult> => {
    return apiClient.post('/api/library/tags/bulk-add', { document_ids: documentIds, tags });
  },

  bulkRemoveTags: async (documentIds: number[], tags: string[]): Promise<BulkTagResult> => {
    return apiClient.post('/api/library/tags/bulk-remove', { document_ids: documentIds, tags });
  },

  applyTagsToDocuments: async (documentIds: number[], tags: string[]): Promise<BulkTagResult> => {
    return apiClient.post('/api/library/documents/tags/apply', { document_ids: documentIds, tags });
  },

  // Individual Document Tag Operations (legacy support)
  addTagsToDocuments: async (documentIds: number[], tags: string[]): Promise<void> => {
    return apiClient.post('/api/library/tags/add', { document_ids: documentIds, tags });
  },

  removeTagsFromDocuments: async (documentIds: number[], tags: string[]): Promise<void> => {
    return apiClient.post('/api/library/tags/remove', { document_ids: documentIds, tags });
  },

  addTagToDocument: async (documentId: number, tagId: string): Promise<void> => {
    return apiClient.post(`/api/library/documents/${documentId}/tags/${tagId}`);
  },

  removeTagFromDocument: async (documentId: number, tagId: string): Promise<void> => {
    return apiClient.delete(`/api/library/documents/${documentId}/tags/${tagId}`);
  },

  // Folder Management
  getFolders: async (parentPath?: string): Promise<Folder[]> => {
    const params = parentPath ? { parent_path: parentPath } : {};
    return apiClient.get('/api/library/folders', { params });
  },

  createFolder: async (folderPath: string, description?: string): Promise<Folder> => {
    return apiClient.post('/api/library/folders', { folder_path: folderPath, description });
  },

  deleteFolder: async (folderPath: string): Promise<void> => {
    return apiClient.delete('/api/library/folders', { data: { folder_path: folderPath } });
  },

  moveDocumentsToFolder: async (documentIds: string[], folderPath: string): Promise<void> => {
    return apiClient.post('/api/library/folders/move', { document_ids: documentIds, folder_path: folderPath });
  },

  // Library Statistics
  getLibraryStats: async (): Promise<LibraryStats> => {
    return apiClient.get('/api/library/stats');
  },
};