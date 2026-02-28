import { useState, lazy, Suspense } from 'react'
import { motion } from 'framer-motion'
import { User, Brain, ChevronDown, Code2 } from 'lucide-react'
import Markdown from './Markdown'

// Lazy-load MissionControl (heavy: recharts, syntax-highlighter, SVG flow graph)
const MissionControl = lazy(() => import('./MissionControl'))

const EXPERT_COLORS = {
  technical: {
    bg: 'bg-accent-teal/15',
    text: 'text-accent-teal',
    border: 'border-accent-teal/30',
  },
  creative: {
    bg: 'bg-accent-rose/15',
    text: 'text-accent-rose',
    border: 'border-accent-rose/30',
  },
  analytical: {
    bg: 'bg-accent-blue/15',
    text: 'text-accent-blue',
    border: 'border-accent-blue/30',
  },
  general: {
    bg: 'bg-accent-violet/15',
    text: 'text-accent-violet',
    border: 'border-accent-violet/30',
  },
}

export default function ChatMessage({ message, index }) {
  const [showDetails, setShowDetails] = useState(false)
  const isUser = message.role === 'user'
  const hasMetadata = !isUser && message.generated_code

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center ${
          isUser ? 'bg-accent-blue/20' : 'bg-accent-purple/20'
        }`}
      >
        {isUser ? (
          <User size={16} className="text-accent-blue" />
        ) : (
          <Brain size={16} className="text-accent-purple" />
        )}
      </div>

      {/* Content */}
      <div
        className={`flex-1 min-w-0 ${isUser ? 'flex flex-col items-end' : ''}`}
      >
        {isUser ? (
          <div className="glass-card p-4 max-w-[80%]" style={{ borderColor: 'rgba(96, 165, 250, 0.1)' }}>
            <p className="text-sm text-text-primary">{message.content}</p>
          </div>
        ) : (
          <div className="space-y-3 max-w-full">
            <div className="glass-card p-4" style={{ borderColor: 'rgba(167, 139, 250, 0.1)' }}>
              <Markdown>{message.final_answer || message.content}</Markdown>

              {/* Expert badges */}
              {message.selected_experts?.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-border/50">
                  {message.selected_experts.map((expert) => {
                    const colors =
                      EXPERT_COLORS[expert] || EXPERT_COLORS.general
                    return (
                      <span
                        key={expert}
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${colors.bg} ${colors.text} border ${colors.border}`}
                      >
                        {expert}
                      </span>
                    )
                  })}
                  {message.code_execution_iterations > 1 && (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent-amber/15 text-accent-amber border border-accent-amber/30">
                      {message.code_execution_iterations} iterations
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Mission Control toggle */}
            {hasMetadata && (
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px] font-medium text-text-muted hover:text-accent-purple hover:bg-accent-purple/5 transition-all"
              >
                <Code2 size={12} />
                Mission Control
                <ChevronDown
                  size={12}
                  className={`transition-transform duration-200 ${showDetails ? 'rotate-180' : ''}`}
                />
              </button>
            )}

            {/* Mission Control Dashboard */}
            {showDetails && hasMetadata && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                transition={{ duration: 0.3 }}
              >
                <Suspense fallback={<div className="glass-card p-4 text-center text-text-muted text-xs">Loading Mission Control...</div>}>
                  <MissionControl data={message} id={index} />
                </Suspense>
              </motion.div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
