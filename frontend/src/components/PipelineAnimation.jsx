import { motion } from 'framer-motion'

const nodes = [
  {
    id: 'query',
    label: 'Query',
    icon: '💬',
    color: '#60a5fa',
    glow: 'rgba(96, 165, 250, 0.4)',
  },
  {
    id: 'orchestrator',
    label: 'Orchestrator',
    icon: '🧠',
    color: '#a78bfa',
    glow: 'rgba(167, 139, 250, 0.4)',
  },
  {
    id: 'codegen',
    label: 'Code Gen',
    icon: '🐍',
    color: '#c084fc',
    glow: 'rgba(192, 132, 252, 0.4)',
  },
  {
    id: 'sandbox',
    label: 'Sandbox',
    icon: '⚡',
    color: '#fbbf24',
    glow: 'rgba(251, 191, 36, 0.4)',
  },
  {
    id: 'experts',
    label: 'Experts',
    icon: '👥',
    color: '#2dd4bf',
    glow: 'rgba(45, 212, 191, 0.4)',
  },
  {
    id: 'answer',
    label: 'Answer',
    icon: '✨',
    color: '#fb923c',
    glow: 'rgba(251, 147, 60, 0.4)',
  },
]

export default function PipelineAnimation() {
  return (
    <div className="glass-card p-8 overflow-hidden">
      <div className="flex items-center justify-between gap-2 relative">
        {/* Connection lines (SVG) */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          preserveAspectRatio="none"
          style={{ overflow: 'visible' }}
        >
          <defs>
            <linearGradient
              id="lineGrad"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.5" />
            </linearGradient>
          </defs>
          {nodes.slice(0, -1).map((node, i) => {
            const totalNodes = nodes.length
            const startPct = ((i + 0.5) / totalNodes) * 100
            const endPct = ((i + 1.5) / totalNodes) * 100
            return (
              <motion.line
                key={`line-${i}`}
                x1={`${startPct}%`}
                y1="50%"
                x2={`${endPct}%`}
                y2="50%"
                stroke="url(#lineGrad)"
                strokeWidth="2"
                strokeDasharray="6 4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 + i * 0.15, duration: 0.5 }}
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="20"
                  to="0"
                  dur="1.5s"
                  repeatCount="indefinite"
                />
              </motion.line>
            )
          })}
        </svg>

        {/* Nodes */}
        {nodes.map((node, i) => (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              delay: i * 0.12,
              duration: 0.4,
              type: 'spring',
              stiffness: 200,
            }}
            className="flex flex-col items-center gap-2 z-10 flex-1"
          >
            <motion.div
              animate={{
                boxShadow: [
                  `0 0 0px ${node.glow}`,
                  `0 0 24px ${node.glow}`,
                  `0 0 0px ${node.glow}`,
                ],
              }}
              transition={{
                delay: 1 + i * 0.3,
                duration: 2,
                repeat: Infinity,
                repeatDelay: 1,
              }}
              className="w-14 h-14 md:w-16 md:h-16 rounded-2xl flex items-center justify-center text-2xl"
              style={{
                backgroundColor: `${node.color}15`,
                border: `1.5px solid ${node.color}30`,
              }}
            >
              {node.icon}
            </motion.div>
            <span className="text-[11px] font-medium text-text-secondary whitespace-nowrap">
              {node.label}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Caption */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 0.8 }}
        className="text-center text-[11px] text-text-muted mt-6 font-mono tracking-wide"
      >
        Query → Orchestrator writes async Python → Sandbox executes →
        Experts called programmatically → Answer
      </motion.p>
    </div>
  )
}
