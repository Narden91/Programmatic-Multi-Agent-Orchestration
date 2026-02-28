import { useState } from 'react'
import {
  Key,
  Cpu,
  Users,
  Zap,
  Trash2,
  ChevronDown,
  Code2,
  Brain,
  Palette,
  BarChart3,
  MessageSquare,
  BookOpen,
  Home,
  X,
  Shield,
} from 'lucide-react'

const EXPERT_INFO = [
  {
    name: 'Technical',
    icon: Code2,
    color: 'text-accent-teal',
    desc: 'Code, tech, science',
  },
  {
    name: 'Creative',
    icon: Palette,
    color: 'text-accent-rose',
    desc: 'Stories, ideas, content',
  },
  {
    name: 'Analytical',
    icon: BarChart3,
    color: 'text-accent-blue',
    desc: 'Data, logic, analysis',
  },
  {
    name: 'General',
    icon: MessageSquare,
    color: 'text-accent-violet',
    desc: 'Conversations, facts',
  },
]

export default function Sidebar({
  config,
  models,
  stats,
  onSetApiKey,
  onSetModel,
  onClear,
  isOpen,
  onToggle,
  currentPage,
  onNavigate,
}) {
  const [showKey, setShowKey] = useState(false)

  return (
    <aside className="w-72 h-full bg-bg-surface/95 backdrop-blur-md border-r border-border flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-accent-purple/20 flex items-center justify-center">
              <Brain className="w-5 h-5 text-accent-purple" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-text-primary tracking-tight">
                Programmatic MoE
              </h1>
              <p className="text-[10px] text-text-muted font-mono">
                Code-as-Orchestration · v0.5.0
              </p>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="lg:hidden p-1 text-text-muted hover:text-text-primary"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* API Key */}
        <section>
          <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
            <Key size={12} />
            API Key
          </label>
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={config.apiKey}
              onChange={(e) => onSetApiKey(e.target.value)}
              placeholder={
                config.hasEnvKey ? '••• from .env' : 'Enter Groq API key'
              }
              className="w-full bg-bg/60 border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-purple/50 transition-colors"
            />
            <button
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary text-[10px]"
            >
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
          {config.hasEnvKey && !config.apiKey && (
            <p className="mt-1 text-[10px] text-accent-green">
              ✓ Using key from environment
            </p>
          )}
        </section>

        {/* Model Selector */}
        <section>
          <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
            <Cpu size={12} />
            Model
          </label>
          <div className="relative">
            <select
              value={config.model}
              onChange={(e) => onSetModel(e.target.value)}
              className="w-full bg-bg/60 border border-border rounded-lg px-3 py-2 text-xs font-mono text-text-primary appearance-none focus:outline-none focus:border-accent-purple/50 transition-colors cursor-pointer"
            >
              {(models.length > 0 ? models : [config.model]).map((m) => (
                <option key={m} value={m} className="bg-bg-surface">
                  {m.split('/').pop()}
                </option>
              ))}
            </select>
            <ChevronDown
              size={12}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
            />
          </div>
        </section>

        {/* Experts */}
        <section>
          <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
            <Users size={12} />
            Expert Pool
          </label>
          <div className="space-y-1.5">
            {EXPERT_INFO.map(({ name, icon: Icon, color, desc }) => (
              <div
                key={name}
                className="flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg bg-bg/40 border border-border/50"
              >
                <Icon size={13} className={color} />
                <div>
                  <span className="text-xs font-medium text-text-primary">
                    {name}
                  </span>
                  <span className="text-[10px] text-text-muted ml-1.5">
                    {desc}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
            <Shield size={12} />
            Why this is novel
          </label>
          <div className="rounded-xl border border-accent-purple/25 bg-accent-purple/5 p-3 space-y-2">
            <p className="text-[11px] text-text-secondary leading-relaxed">
              The orchestrator writes a fresh async Python plan for each query,
              instead of routing through a fixed DAG.
            </p>
            <ul className="space-y-1 text-[10px] text-text-secondary">
              <li>• Dynamic tool calling with transient expert functions</li>
              <li>• AST-validated sandbox execution with timeout controls</li>
              <li>• Intermediate reasoning stays in sandbox variables</li>
            </ul>
          </div>
        </section>

        {/* Session Stats */}
        {stats.queriesProcessed > 0 && (
          <section>
            <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
              <Zap size={12} />
              Session
            </label>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center p-2 rounded-lg bg-bg/40 border border-border/50">
                <p className="text-lg font-bold text-accent-purple">
                  {stats.queriesProcessed}
                </p>
                <p className="text-[9px] text-text-muted">Queries</p>
              </div>
              <div className="text-center p-2 rounded-lg bg-bg/40 border border-border/50">
                <p className="text-lg font-bold text-accent-amber">
                  {stats.totalTokens.toLocaleString()}
                </p>
                <p className="text-[9px] text-text-muted">Tokens</p>
              </div>
              <div className="text-center p-2 rounded-lg bg-bg/40 border border-border/50">
                <p className="text-lg font-bold text-accent-teal">
                  {stats.totalExperts}
                </p>
                <p className="text-[9px] text-text-muted">Experts</p>
              </div>
            </div>
          </section>
        )}

        {/* Navigation */}
        <section className="space-y-1">
          <label className="flex items-center gap-2 text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">
            Navigation
          </label>
          <button
            onClick={() => onNavigate?.('chat')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              currentPage === 'chat'
                ? 'bg-accent-purple/15 text-accent-purple border border-accent-purple/30'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover/70'
            }`}
          >
            <Home size={13} />
            Chat
          </button>
          <button
            onClick={() => onNavigate?.('about')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              currentPage === 'about'
                ? 'bg-accent-purple/15 text-accent-purple border border-accent-purple/30'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover/70'
            }`}
          >
            <BookOpen size={13} />
            Architecture &amp; About
          </button>
        </section>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <button
          onClick={onClear}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-bg-hover/70 border border-border text-xs text-text-secondary hover:text-accent-rose hover:border-accent-rose/30 transition-all"
        >
          <Trash2 size={12} />
          Clear Chat
        </button>
      </div>
    </aside>
  )
}
