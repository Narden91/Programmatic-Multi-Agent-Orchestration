import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  ChevronDown,
  Code2,
  Palette,
  BarChart3,
  MessageSquare,
} from 'lucide-react'

const EXPERT_CONFIG = {
  technical: {
    icon: Code2,
    color: '#2dd4bf',
    bg: 'rgba(45, 212, 191, 0.1)',
    border: 'rgba(45, 212, 191, 0.2)',
  },
  creative: {
    icon: Palette,
    color: '#fb7185',
    bg: 'rgba(251, 113, 133, 0.1)',
    border: 'rgba(251, 113, 133, 0.2)',
  },
  analytical: {
    icon: BarChart3,
    color: '#60a5fa',
    bg: 'rgba(96, 165, 250, 0.1)',
    border: 'rgba(96, 165, 250, 0.2)',
  },
  general: {
    icon: MessageSquare,
    color: '#c084fc',
    bg: 'rgba(192, 132, 252, 0.1)',
    border: 'rgba(192, 132, 252, 0.2)',
  },
}

function ExpertCard({ name, response, delay }) {
  const [expanded, setExpanded] = useState(false)
  const config = EXPERT_CONFIG[name] || EXPERT_CONFIG.general
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-xl overflow-hidden"
      style={{
        backgroundColor: config.bg,
        border: `1px solid ${config.border}`,
      }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-3 text-left cursor-pointer"
      >
        <div
          className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${config.color}20` }}
        >
          <Icon size={14} style={{ color: config.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-semibold capitalize"
            style={{ color: config.color }}
          >
            {name} Expert
          </p>
          {!expanded && (
            <p className="text-[11px] text-text-muted truncate mt-0.5">
              {response?.slice(0, 100)}...
            </p>
          )}
        </div>
        <ChevronDown
          size={14}
          className={`text-text-muted transition-transform flex-shrink-0 ${expanded ? 'rotate-180' : ''}`}
        />
      </button>

      {expanded && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="px-3 pb-3"
        >
          <div className="p-3 rounded-lg bg-bg/40 text-xs text-text-secondary leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
            {response}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default function ExpertCards({ expertResponses, selectedExperts }) {
  if (!expertResponses || Object.keys(expertResponses).length === 0) {
    return (
      <div className="text-center py-8 text-text-muted text-xs">
        No expert outputs captured.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {Object.entries(expertResponses).map(([name, response], i) => (
        <ExpertCard
          key={name}
          name={name}
          response={response}
          delay={i * 0.1}
        />
      ))}
    </div>
  )
}
