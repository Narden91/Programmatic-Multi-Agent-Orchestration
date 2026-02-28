import { Menu } from 'lucide-react'

export default function Layout({
  children,
  sidebar,
  sidebarOpen,
  onToggleSidebar,
}) {
  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      {/* Mobile sidebar toggle */}
      <button
        onClick={onToggleSidebar}
        className="fixed top-4 left-4 z-50 lg:hidden p-2 glass rounded-lg text-text-secondary hover:text-text-primary transition-colors"
      >
        <Menu size={20} />
      </button>

      {/* Sidebar */}
      <div
        className={`
        fixed lg:relative z-40 h-full transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}
      >
        {sidebar}
      </div>

      {/* Backdrop for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/25 lg:hidden"
          onClick={onToggleSidebar}
        />
      )}

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden min-w-0">
        {children}
      </main>
    </div>
  )
}
