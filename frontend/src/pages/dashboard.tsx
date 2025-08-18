import { useState, Suspense } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import Layout from '@/components/common/Layout';
import ErrorBoundary from '@/components/common/ErrorBoundary';
import { useAuth } from '@/hooks/useAuth';
import { DocumentTextIcon, ArrowRightIcon, CloudArrowUpIcon, ChartBarIcon } from '@heroicons/react/24/outline';

// Lazy load document upload component
const DocumentUpload = dynamic(() => import('@/components/dashboard/DocumentUpload'), {
  loading: () => (
    <div className="animate-pulse">
      <div className="h-32 bg-gray-200 rounded-lg"></div>
    </div>
  ),
});

export default function Dashboard() {
  const { user } = useAuth();
  const [uploadCount, setUploadCount] = useState(0);

  const handleUploadComplete = () => {
    setUploadCount(prev => prev + 1);
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Welcome back, {user?.username}!
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Upload new documents and get quick access to your document library.
            </p>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white shadow-md rounded-lg p-6">
              <div className="flex items-center">
                <CloudArrowUpIcon className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Upload Documents</h3>
                  <p className="text-sm text-gray-500">Add new files to your library</p>
                </div>
              </div>
            </div>
            
            <Link href="/documents" className="bg-white shadow-md rounded-lg p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <DocumentTextIcon className="h-8 w-8 text-green-600" />
                  <div className="ml-4">
                    <h3 className="text-lg font-medium text-gray-900">Manage Documents</h3>
                    <p className="text-sm text-gray-500">View and organize your files</p>
                  </div>
                </div>
                <ArrowRightIcon className="h-5 w-5 text-gray-400" />
              </div>
            </Link>

            <Link href="/analytics" className="bg-white shadow-md rounded-lg p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <ChartBarIcon className="h-8 w-8 text-purple-600" />
                  <div className="ml-4">
                    <h3 className="text-lg font-medium text-gray-900">View Analytics</h3>
                    <p className="text-sm text-gray-500">Track usage and insights</p>
                  </div>
                </div>
                <ArrowRightIcon className="h-5 w-5 text-gray-400" />
              </div>
            </Link>
          </div>

          {/* Upload Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-lg font-medium text-gray-900">
                  Upload New Documents
                </h2>
                <p className="text-sm text-gray-500">
                  Drag and drop files or browse to add them to your library
                </p>
              </div>
              <div className="flex items-center space-x-3">
                <Link 
                  href="/documents"
                  className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                >
                  Manage Documents
                  <ArrowRightIcon className="ml-2 h-4 w-4" />
                </Link>
                {uploadCount > 0 && (
                  <Link 
                    href="/documents"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 transition-colors"
                  >
                    View Uploaded ({uploadCount})
                    <ArrowRightIcon className="ml-2 h-4 w-4" />
                  </Link>
                )}
              </div>
            </div>
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
        </div>
      </div>
    </Layout>
  );
}