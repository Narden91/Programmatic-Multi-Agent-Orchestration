import { motion } from 'framer-motion'
import PipelineAnimation from './PipelineAnimation'
import PromptCards from './PromptCards'
import { Sparkles, GitBranch, Code2, Zap } from 'lucide-react'

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12 },
  },
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
}

const stats = [
  {
    label: 'Orchestration',
    value: 'Dynamic Python',
    detail: 'unique script per query',
    icon: Code2,
    color: '#a78bfa',
    bgColor: 'rgba(167, 139, 250, 0.1)',
  },
  {
    label: 'Execution',
    value: 'Sandboxed',
    detail: 'AST-validated + timeout',
    icon: Zap,
    color: '#fbbf24',
    bgColor: 'rgba(251, 191, 36, 0.1)',
  },
  {
    label: 'Experts',
    value: 'Parallel Async',
    detail: 'context-compressed output',
    icon: Sparkles,
    color: '#2dd4bf',
    bgColor: 'rgba(45, 212, 191, 0.1)',
  },
]

export default function Hero({ onSelectPrompt }) {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="max-w-5xl mx-auto px-6 py-12 space-y-12"
    >
      {/* Title */}
      <motion.div variants={item} className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass border border-accent-purple/20 text-xs text-accent-purple font-medium">
          <Sparkles size={12} />
          Programmatic Multi-Agent Orchestration
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight leading-tight">
          <span className="gradient-text">Code-as-Orchestration</span>
        </h1>
        <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
          The AI writes its own async Python orchestration script — then
          executes it in a sandbox. No static DAGs. No boilerplate. Just code.
        </p>
      </motion.div>

      {/* Pipeline Animation */}
      <motion.div variants={item}>
        <PipelineAnimation />
      </motion.div>

      {/* Stats */}
      <motion.div
        variants={item}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        {stats.map(({ label, value, detail, icon: Icon, color, bgColor }) => (
          <div
            key={label}
            className="glass-card p-5 text-center group hover:border-white/10 transition-all duration-300"
          >
            <div
              className="inline-flex items-center justify-center w-10 h-10 rounded-xl mb-3"
              style={{ backgroundColor: bgColor }}
            >
              <Icon size={20} style={{ color }} />
            </div>
            <p className="text-xs text-text-muted uppercase tracking-wider mb-1">
              {label}
            </p>
            <p className="text-xl font-bold" style={{ color }}>
              {value}
            </p>
            <p className="text-[11px] text-text-muted mt-1">{detail}</p>
          </div>
        ))}
      </motion.div>

      {/* Comparison */}
      <motion.div variants={item}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Traditional */}
          <div className="glass-card p-6" style={{ borderColor: 'rgba(251, 113, 133, 0.1)' }}>
            <div className="flex items-center gap-2 mb-4">
              <GitBranch size={16} className="text-accent-rose" />
              <h3 className="text-sm font-semibold text-accent-rose">
                Traditional DAG Routing
              </h3>
            </div>
            <ul className="space-y-2 text-xs text-text-secondary">
              {[
                'Static, developer-defined graphs',
                'Complex state dictionaries bloat context',
                'Same routing for every query type',
                'Massive boilerplate DAG routing code',
              ].map((text) => (
                <li key={text} className="flex items-start gap-2">
                  <span className="text-accent-rose mt-0.5">✗</span>
                  <span>{text}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Programmatic */}
          <div className="glass-card p-6 gradient-border border-transparent">
            <div className="flex items-center gap-2 mb-4">
              <Code2 size={16} className="text-accent-purple" />
              <h3 className="text-sm font-semibold text-accent-purple">
                Code-as-Orchestration
              </h3>
            </div>
            <ul className="space-y-2 text-xs text-text-secondary">
              {[
                [
                  'AI writes a ',
                  'unique script per query',
                ],
                [
                  'Variables stay in sandbox — ',
                  '80%+ token savings',
                ],
                [
                  'Dynamic ',
                  'parallel/sequential expert calls',
                ],
                [
                  'Zero boilerplate — ',
                  "the AI writes its own graph",
                ],
              ].map(([prefix, bold], i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-accent-green mt-0.5">✓</span>
                  <span>
                    {prefix}
                    <strong className="text-text-primary">{bold}</strong>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Prompt Cards */}
      <motion.div variants={item}>
        <PromptCards onSelect={onSelectPrompt} />
      </motion.div>
    </motion.div>
  )
}
