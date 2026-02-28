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
      className="max-w-3xl mx-auto px-6 pt-20 pb-12 space-y-10"
    >
      {/* Title */}
      <motion.div variants={item} className="text-center space-y-4">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight">
          <span className="gradient-text">Code-as-Orchestration</span>
        </h1>
        <p className="text-base text-text-secondary max-w-xl mx-auto leading-relaxed">
          Ask anything. The AI writes a dynamic Python script to orchestrate
          specialized experts — then executes it in a sandbox.
        </p>
        <motion.button
          variants={item}
          onClick={onLearnMore}
          className="inline-flex items-center gap-1.5 text-xs text-accent-purple hover:text-accent-blue transition-colors mt-1"
        >
          <Sparkles size={12} />
          Learn how it works
          <ArrowRight size={12} />
        </motion.button>
      </motion.div>

      {/* Prompt Cards */}
      <motion.div variants={item}>
        <PromptCards onSelect={onSelectPrompt} />
      </motion.div>
    </motion.div>
  )
}
