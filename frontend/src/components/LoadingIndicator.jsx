import { motion } from 'framer-motion'
import { Brain, Code2, Zap, Users, Sparkles } from 'lucide-react'

const stages = [
  {
    icon: Brain,
    label: 'Orchestrator analyzing query...',
    color: '#a78bfa',
  },
  {
    icon: Code2,
    label: 'Generating orchestration script...',
    color: '#c084fc',
  },
  { icon: Zap, label: 'Executing in sandbox...', color: '#fbbf24' },
  { icon: Users, label: 'Querying experts...', color: '#2dd4bf' },
  {
    icon: Sparkles,
    label: 'Synthesizing answer...',
    color: '#fb923c',
  },
]

export default function LoadingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-4"
    >
      <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-accent-purple/20 flex items-center justify-center">
        <Brain size={16} className="text-accent-purple" />
      </div>
      <div className="glass-card p-4 flex-1 max-w-lg">
        <div className="space-y-2.5">
          {stages.map((stage, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0.2 }}
              animate={{ opacity: [0.2, 1, 0.2] }}
              transition={{
                delay: i * 1.2,
                duration: 1.2,
                repeat: Infinity,
                repeatDelay: stages.length * 1.2 - 1.2,
              }}
              className="flex items-center gap-2"
            >
              <stage.icon size={12} style={{ color: stage.color }} />
              <span className="text-xs text-text-secondary">
                {stage.label}
              </span>
            </motion.div>
          ))}
        </div>
        <div className="flex gap-1.5 mt-3 pt-3 border-t border-border/30">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-accent-purple"
              animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
              transition={{
                delay: i * 0.15,
                duration: 0.6,
                repeat: Infinity,
              }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}
