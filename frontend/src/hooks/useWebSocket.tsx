import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import Cookies from 'js-cookie';
import toast from 'react-hot-toast';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface UseWebSocketOptions {
  onDocumentUpdate?: (data: any) => void;
  onDocumentProgress?: (data: any) => void;
  onSystemAlert?: (data: any) => void;
  onUserUpdate?: (data: any) => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const socketRef = useRef<Socket | null>(null);

  const connect = useCallback(() => {
    const token = Cookies.get('access_token');
    if (!token) return;

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
    
    socketRef.current = io(wsUrl, {
      auth: {
        token: token,
      },
      transports: ['websocket', 'polling'],
    });

    const socket = socketRef.current;

    socket.on('connect', () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    });

    socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
    });

    // Document events
    socket.on('document_uploaded', (data) => {
      setLastMessage({ type: 'document_uploaded', data, timestamp: new Date().toISOString() });
      options.onDocumentUpdate?.(data);
      toast.success(`Document "${data.filename}" uploaded successfully`);
    });

    socket.on('document_processed', (data) => {
      setLastMessage({ type: 'document_processed', data, timestamp: new Date().toISOString() });
      options.onDocumentUpdate?.(data);
      toast.success(`Document "${data.filename}" processed successfully`);
    });

    socket.on('document_failed', (data) => {
      setLastMessage({ type: 'document_failed', data, timestamp: new Date().toISOString() });
      options.onDocumentUpdate?.(data);
      toast.error(`Document processing failed: ${data.error}`);
    });

    // System events
    socket.on('system_alert', (data) => {
      setLastMessage({ type: 'system_alert', data, timestamp: new Date().toISOString() });
      options.onSystemAlert?.(data);
      
      if (data.level === 'error') {
        toast.error(`System Alert: ${data.message}`);
      } else if (data.level === 'warning') {
        toast((t) => (
          <div className="flex items-center">
            <span>⚠️ {data.message}</span>
          </div>
        ));
      }
    });

    // User events
    socket.on('user_updated', (data) => {
      setLastMessage({ type: 'user_updated', data, timestamp: new Date().toISOString() });
      options.onUserUpdate?.(data);
    });

    // Search events
    socket.on('search_completed', (data) => {
      setLastMessage({ type: 'search_completed', data, timestamp: new Date().toISOString() });
    });

    // Document progress events
    socket.on('document_processing_progress', (data) => {
      setLastMessage({ type: 'document_processing_progress', data, timestamp: new Date().toISOString() });
      options.onDocumentProgress?.(data);
    });

    socket.on('document_processing_complete', (data) => {
      setLastMessage({ type: 'document_processing_complete', data, timestamp: new Date().toISOString() });
      options.onDocumentProgress?.(data);
    });

    return socket;
  }, [options]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const sendMessage = useCallback((type: string, data: any) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit(type, data);
    }
  }, [isConnected]);

  useEffect(() => {
    const token = Cookies.get('access_token');
    if (token) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
  };
};