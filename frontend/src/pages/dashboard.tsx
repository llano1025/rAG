import { useState } from 'react';
import Layout from '@/components/common/Layout';
import DocumentUpload from '@/components/dashboard/DocumentUpload';
import DocumentList from '@/components/dashboard/DocumentList';
import { useAuth } from '@/hooks/useAuth';

export default function Dashboard() {
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
              Welcome back, {user?.full_name || user?.username}!
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Manage your documents and search through your knowledge base.
            </p>
          </div>

          {/* Upload Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Upload Documents
            </h2>
            <DocumentUpload onUploadComplete={handleUploadComplete} />
          </div>

          {/* Documents Section */}
          <div className="bg-white shadow-md rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-gray-900">
                Your Documents
              </h2>
            </div>
            <DocumentList refreshTrigger={refreshTrigger} />
          </div>
        </div>
      </div>
    </Layout>
  );
}