// frontend/src/pages/chat.tsx

import React from 'react';
import { GetServerSideProps } from 'next';
import Layout from '../components/common/Layout';
import ProtectedRoute from '../components/common/ProtectedRoute';
import ChatInterface from '../components/chat/ChatInterface';

const ChatPage: React.FC = () => {
  return (
    <ProtectedRoute>
      <Layout>
        <div className="h-full">
          <ChatInterface />
        </div>
      </Layout>
    </ProtectedRoute>
  );
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  return {
    props: {}
  };
};

export default ChatPage;