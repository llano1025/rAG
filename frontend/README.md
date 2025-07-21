# RAG System Frontend

A modern React/Next.js frontend for the RAG (Retrieval-Augmented Generation) system.

## Features

- **Authentication**: Login/register with JWT token authentication
- **Document Management**: Upload, view, and manage documents with drag-and-drop support
- **Search Interface**: Advanced search with filters, semantic search, and hybrid search capabilities
- **Analytics Dashboard**: Usage statistics and system health monitoring
- **Admin Interface**: User management and system administration
- **Real-time Updates**: WebSocket integration for live updates
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **React Hook Form** - Form management
- **React Query** - Server state management
- **Axios** - HTTP client
- **Socket.IO** - Real-time communication
- **React Dropzone** - File upload interface
- **React Hot Toast** - Notifications

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Running RAG backend API

### Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp .env.example .env.local
```

3. Update environment variables in `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Project Structure

```
src/
├── api/          # API client and endpoints
├── components/   # React components
│   ├── auth/     # Authentication components
│   ├── dashboard/# Dashboard components
│   ├── search/   # Search interface components
│   ├── admin/    # Admin interface components
│   └── common/   # Shared components
├── hooks/        # Custom React hooks
├── pages/        # Next.js pages
├── styles/       # CSS styles
├── types/        # TypeScript type definitions
└── utils/        # Utility functions
```

## Key Components

### Authentication
- JWT token-based authentication
- Protected routes with automatic redirection
- User session management with cookies

### Document Management
- Drag-and-drop file upload
- Batch upload support
- Document preview and management
- File type validation and size limits

### Search Interface
- Basic, semantic, and hybrid search modes
- Advanced filters (file type, date range, owner)
- Search history and saved searches
- Real-time search results with highlighting

### Analytics
- Usage statistics dashboard
- System health monitoring
- Performance metrics
- Data visualization

### Admin Interface
- User management (CRUD operations)
- Role-based access control
- System settings management
- API key management

## API Integration

The frontend integrates with the RAG backend API through:
- RESTful API endpoints
- JWT authentication headers
- File upload with progress tracking
- WebSocket connections for real-time updates

## Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for type safety
3. Add proper error handling and loading states
4. Test components and features thoroughly
5. Update documentation as needed

## Production Deployment

1. Build the application:
```bash
npm run build
```

2. Start the production server:
```bash
npm start
```

For deployment to platforms like Vercel, Netlify, or Docker:
- Configure environment variables
- Set up proper API endpoints
- Configure static file serving
- Set up monitoring and error tracking