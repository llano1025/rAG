import { ReactNode, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import {
  DocumentTextIcon,
  MagnifyingGlassIcon,
  UserGroupIcon,
  ChartBarIcon,
  ArrowRightOnRectangleIcon,
  ChatBubbleLeftRightIcon,
  CpuChipIcon,
  HomeIcon,
  PlusIcon,
  EyeIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '@/hooks/useAuth';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'Search', href: '/search', icon: MagnifyingGlassIcon },
  { name: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
];

const adminNavigation = [
  { name: 'Users', href: '/admin/users', icon: UserGroupIcon },
  { 
    name: 'Models', 
    icon: CpuChipIcon,
    expandable: true,
    children: [
      { name: 'Manage Models', href: '/admin/models/manage', icon: CpuChipIcon },
      { name: 'Discover Models', href: '/admin/models/discover', icon: MagnifyingGlassIcon },
      { name: 'Register Model', href: '/admin/models/register', icon: PlusIcon },
    ]
  },
];

export default function Layout({ children }: LayoutProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [expandedMenus, setExpandedMenus] = useState<string[]>([]);

  const isAdmin = user?.roles?.includes('admin') || user?.role === 'admin';

  const toggleMenu = (menuName: string) => {
    setExpandedMenus(prev => 
      prev.includes(menuName) 
        ? prev.filter(name => name !== menuName)
        : [...prev, menuName]
    );
  };

  const isMenuExpanded = (menuName: string) => expandedMenus.includes(menuName);

  const isActiveRoute = (item: any) => {
    if (item.expandable) {
      return item.children?.some((child: any) => router.pathname === child.href);
    }
    return router.pathname === item.href;
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="hidden md:flex md:w-64 md:flex-col">
        <div className="flex flex-col flex-grow pt-5 bg-white overflow-y-auto border-r border-gray-200">
          <div className="flex items-center flex-shrink-0 px-4">
            <h1 className="text-xl font-bold text-gray-900">RAG System</h1>
          </div>
          
          <div className="mt-8 flex-grow flex flex-col">
            <nav className="flex-1 px-2 space-y-1">
              {navigation.map((item) => {
                const isActive = router.pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                      isActive
                        ? 'bg-primary-100 text-primary-900'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                  >
                    <item.icon
                      className={`mr-3 h-6 w-6 ${
                        isActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'
                      }`}
                    />
                    {item.name}
                  </Link>
                );
              })}
              
              {isAdmin && (
                <>
                  <div className="pt-4 mt-4 border-t border-gray-200">
                    <p className="px-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                      Admin
                    </p>
                  </div>
                  {adminNavigation.map((item) => {
                    const isActive = isActiveRoute(item);
                    const isExpanded = isMenuExpanded(item.name);
                    
                    if (item.expandable) {
                      return (
                        <div key={item.name}>
                          <button
                            onClick={() => toggleMenu(item.name)}
                            className={`group w-full flex items-center justify-between px-2 py-2 text-sm font-medium rounded-md ${
                              isActive
                                ? 'bg-primary-100 text-primary-900'
                                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                            }`}
                          >
                            <div className="flex items-center">
                              <item.icon
                                className={`mr-3 h-6 w-6 ${
                                  isActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'
                                }`}
                              />
                              {item.name}
                            </div>
                            {isExpanded ? (
                              <ChevronDownIcon className="h-4 w-4" />
                            ) : (
                              <ChevronRightIcon className="h-4 w-4" />
                            )}
                          </button>
                          {isExpanded && item.children && (
                            <div className="ml-6 mt-1 space-y-1">
                              {item.children.map((child) => {
                                const isChildActive = router.pathname === child.href;
                                return (
                                  <Link
                                    key={child.name}
                                    href={child.href}
                                    className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                                      isChildActive
                                        ? 'bg-primary-100 text-primary-900'
                                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                                    }`}
                                  >
                                    <child.icon
                                      className={`mr-3 h-5 w-5 ${
                                        isChildActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'
                                      }`}
                                    />
                                    {child.name}
                                  </Link>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    }
                    
                    return (
                      <Link
                        key={item.name}
                        href={item.href || '#'}
                        className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                          isActive
                            ? 'bg-primary-100 text-primary-900'
                            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                        }`}
                      >
                        <item.icon
                          className={`mr-3 h-6 w-6 ${
                            isActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'
                          }`}
                        />
                        {item.name}
                      </Link>
                    );
                  })}
                </>
              )}
            </nav>
          </div>
          
          {/* User info and logout */}
          <div className="flex-shrink-0 flex border-t border-gray-200 p-4">
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 rounded-full bg-primary-600 flex items-center justify-center">
                    <span className="text-sm font-medium text-white">
                      {user?.username?.charAt(0).toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-700">{user?.username}</p>
                  <p className="text-xs text-gray-500">{user?.role}</p>
                </div>
              </div>
              <button
                onClick={logout}
                className="ml-3 p-1 rounded-full text-gray-400 hover:text-gray-500"
                title="Logout"
              >
                <ArrowRightOnRectangleIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1">
        <main className="flex-1">
          {children}
        </main>
      </div>
    </div>
  );
}