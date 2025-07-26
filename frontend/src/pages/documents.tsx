import { useState, Suspense } from 'react';
import dynamic from 'next/dynamic';
import Layout from '@/components/common/Layout';
import ErrorBoundary from '@/components/common/ErrorBoundary';
import { useAuth } from '@/hooks/useAuth';

// Lazy load document components to prevent blocking
const DocumentUpload = dynamic(() => import('@/components/dashboard/DocumentUpload'), {
  loading: () => (
    <div className="animate-pulse">
      <div className="h-32 bg-gray-200 rounded-lg"></div>
    </div>
  ),
});

const DocumentList = dynamic(() => import('@/components/dashboard/DocumentList'), {
  loading: () => (
    <div className="animate-pulse space-y-4">
      {[...Array(3)].map((_, i) => (
        <div key={i} className="h-16 bg-gray-200 rounded-lg"></div>
      ))}
    </div>
  ),
});

export default function Documents() {
  const { user } = useAuth();
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadComplete = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Document Management
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Upload, organize, and manage your documents for search and analysis.
            </p>
          </div>

          {/* Upload Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Upload Documents
            </h2>
            <ErrorBoundary>
              <Suspense fallback={
                <div className="animate-pulse">
                  <div className="h-32 bg-gray-200 rounded-lg"></div>
                </div>
              }>
                <DocumentUpload onUploadComplete={handleUploadComplete} />
              </Suspense>
            </ErrorBoundary>
          </div>

          {/* Documents Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-gray-900">
                Your Documents
              </h2>
            </div>
            <ErrorBoundary>
              <Suspense fallback={
                <div className="animate-pulse space-y-4">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="h-16 bg-gray-200 rounded-lg"></div>
                  ))}
                </div>
              }>
                <DocumentList refreshTrigger={refreshTrigger} />
              </Suspense>
            </ErrorBoundary>
          </div>
        </div>
      </div>
    </Layout>
  );
}