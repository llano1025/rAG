import { apiClient } from './client';

export interface Tag {
  tag_name: string;
  document_count: number;
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

export const libraryApi = {
  // Tag Management
  getTags: async (): Promise<Tag[]> => {
    return apiClient.get('/api/library/tags');
  },

  createTag: async (tagName: string): Promise<Tag> => {
    return apiClient.post('/api/library/tags', { tag_name: tagName });
  },

  addTagsToDocuments: async (documentIds: string[], tags: string[]): Promise<void> => {
    return apiClient.post('/api/library/tags/add', { document_ids: documentIds, tags });
  },

  removeTagsFromDocuments: async (documentIds: string[], tags: string[]): Promise<void> => {
    return apiClient.post('/api/library/tags/remove', { document_ids: documentIds, tags });
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