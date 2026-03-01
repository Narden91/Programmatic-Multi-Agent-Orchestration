import { motion } from 'framer-motion'
import PromptCards from './PromptCards'
import { Sparkles, ArrowRight } from 'lucide-react'

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
      className="max-w-4xl mx-auto px-6 pt-24 pb-12 space-y-12"
    >
      {/* Title */}
      <motion.div variants={item} className="text-center space-y-6">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-bg-surface text-xs font-semibold text-text-secondary shadow-sm">
          <Sparkles size={14} />
          Programmatic Multi-Agent Orchestration
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-text-primary leading-tight">
          Code-as-Orchestration
        </h1>
        <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
          Instead of fixed routing, each request gets a fresh async script.
          The script runs in a sandbox, calls expert tools programmatically,
          and returns the final result.
        </p>
        <motion.button
          variants={item}
          onClick={onLearnMore}
          className="inline-flex items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-text-primary transition-colors mt-2"
        >
          Learn how it works
          <ArrowRight size={14} />
        </motion.button>
      </motion.div>

      {/* Prompt Cards */}
      <motion.div variants={item} className="pt-8">
        <PromptCards onSelect={onSelectPrompt} />
      </motion.div>
    </motion.div>
  )
}
