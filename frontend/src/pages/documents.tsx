import { useState, Suspense } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import Layout from '@/components/common/Layout';
import ErrorBoundary from '@/components/common/ErrorBoundary';
import { useAuth } from '@/hooks/useAuth';
import { PlusIcon } from '@heroicons/react/24/outline';

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

  const handleRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Document Library
              </h1>
              <p className="mt-1 text-sm text-gray-500">
                Manage, organize, and search through your document collection.
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Link 
                href="/dashboard"
                className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
              >
                Back to Dashboard
              </Link>
              <Link 
                href="/dashboard"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
              >
                <PlusIcon className="h-4 w-4 mr-2" />
                Upload Documents
              </Link>
            </div>
          </div>

          {/* Documents Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-gray-900">
                Your Documents
              </h2>
              <button
                onClick={handleRefresh}
                className="text-sm text-blue-600 hover:text-blue-700 transition-colors"
              >
                Refresh
              </button>
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