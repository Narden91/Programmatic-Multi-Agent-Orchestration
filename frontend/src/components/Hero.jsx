import { motion } from 'framer-motion'
import PromptCards from './PromptCards'
import { Sparkles, ArrowRight, Code2, Shield, Workflow } from 'lucide-react'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
}
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: 'easeOut' } },
}

export default function Hero({ onSelectPrompt, onLearnMore }) {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="max-w-5xl mx-auto px-6 pt-16 pb-12 space-y-10"
    >
      {/* Title */}
      <motion.div variants={item} className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-purple/25 bg-accent-purple/10 text-xs font-medium text-accent-purple">
          <Sparkles size={12} />
          Programmatic Multi-Agent Orchestration
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight">
          <span className="gradient-text">Code-as-Orchestration</span>
        </h1>
        <p className="text-base text-text-secondary max-w-3xl mx-auto leading-relaxed">
          Instead of fixed DAG routing, each request gets a fresh
          <strong className="text-text-primary"> async orchestration script</strong>.
          That script runs in a hardened sandbox, calls expert tools
          programmatically, and returns only the final synthesis.
        </p>
        <motion.button
          variants={item}
          onClick={onLearnMore}
          className="inline-flex items-center gap-1.5 text-xs text-accent-purple hover:text-accent-blue transition-colors mt-1"
        >
          Learn the architecture details
          <ArrowRight size={12} />
        </motion.button>
      </motion.div>

      <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          {
            title: 'Novelty',
            icon: Code2,
            text: 'The orchestrator writes executable Python plans per query, not static graph edges.',
            color: 'text-accent-purple',
            bg: 'bg-accent-purple/10',
          },
          {
            title: 'How it is achieved',
            icon: Workflow,
            text: 'Generated code awaits expert tool functions, loops over data, and branches with real control flow.',
            color: 'text-accent-blue',
            bg: 'bg-accent-blue/10',
          },
          {
            title: 'Why it matters',
            icon: Shield,
            text: 'Execution is AST-validated and sandboxed, while intermediate reasoning stays out of chat context.',
            color: 'text-accent-teal',
            bg: 'bg-accent-teal/10',
          },
        ].map(({ title, icon: Icon, text, color, bg }) => (
          <div key={title} className="glass-card p-4">
            <div className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center mb-2.5`}>
              <Icon size={15} className={color} />
            </div>
            <h3 className="text-sm font-semibold text-text-primary mb-1.5">
              {title}
            </h3>
            <p className="text-xs text-text-secondary leading-relaxed">{text}</p>
          </div>
        ))}
      </motion.div>

      {/* Prompt Cards */}
      <motion.div variants={item}>
        <PromptCards onSelect={onSelectPrompt} />
      </motion.div>
    </motion.div>
  )
}
