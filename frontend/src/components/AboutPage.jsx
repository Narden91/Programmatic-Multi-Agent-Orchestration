import { motion } from 'framer-motion'
import PipelineAnimation from './PipelineAnimation'
import {
  Sparkles,
  Code2,
  Zap,
  Brain,
  Shield,
  ArrowLeft,
  Users,
  Layers,
  Workflow,
} from 'lucide-react'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
}
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

export default function AboutPage({ onBack }) {
  return (
    <div className="flex-1 overflow-y-auto">
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="max-w-4xl mx-auto px-6 py-10 space-y-10"
      >
        {/* Back button */}
        <motion.div variants={item}>
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-xs text-text-muted hover:text-accent-primary transition-colors"
          >
            <ArrowLeft size={14} />
            Back to Chat
          </button>
        </motion.div>

        {/* Header */}
        <motion.div variants={item} className="space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-bg-surface text-xs text-text-secondary font-medium shadow-sm">
            <Sparkles size={12} />
            How It Works
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-text-primary">
            A Smarter Way to Coordinate AI
          </h1>
          <p className="text-base text-text-secondary max-w-3xl leading-relaxed">
            Instead of relying on fixed pathways, our system dynamically generates a custom script for every request. This script orchestrates specialized AI agents to solve your exact problem efficiently.
          </p>
        </motion.div>

        <motion.div variants={item} className="glass-card p-5">
          <h2 className="text-sm font-semibold text-text-primary mb-2">
            In Simple Terms
          </h2>
          <p className="text-sm text-text-secondary leading-relaxed">
            Think of the orchestrator as a project manager. When you ask a question, it doesn't just guess the answer—it writes a plan, creates a team of expert agents on the fly, and manages their work until the job is done perfectly.
          </p>
        </motion.div>

        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <Zap size={14} className="text-accent-primary" />
            Practical Examples
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              {
                title: 'Parallel Analysis',
                desc: "When you ask the system to compare two long documents, it writes a script to launch two 'reader' agents simultaneously. They read the documents at the same time, and their notes are merged at the end.",
                icon: Users,
              },
              {
                title: 'Iterative Coding',
                desc: "When asked to build an application, it assigns a coding agent to write the code and a separate testing agent to verify it. The script loops this process until the tests pass automatically.",
                icon: Code2,
              }
            ].map(({ title, desc, icon: Icon }) => (
              <div key={title} className="glass-card p-5">
                <div className="w-10 h-10 rounded-lg bg-bg-hover flex items-center justify-center mb-3">
                  <Icon size={18} className="text-accent-primary" />
                </div>
                <p className="text-sm font-semibold text-text-primary mb-2">
                  {title}
                </p>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Pipeline */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <Workflow size={14} className="text-accent-primary" />
            Execution Pipeline
          </h2>
          <PipelineAnimation />
        </motion.div>

        {/* How it works steps */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
            <Layers size={14} className="text-accent-primary" />
            The Process
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {[
              {
                n: 1,
                icon: Brain,
                title: 'Orchestrator',
                desc: 'Analyzes your request and determines the best approach.',
              },
              {
                n: 2,
                icon: Code2,
                title: 'Code Builder',
                desc: 'Writes a custom script to solve the specific request.',
              },
              {
                n: 3,
                icon: Shield,
                title: 'Sandbox',
                desc: 'Safely runs the script in an isolated environment.',
              },
              {
                n: 4,
                icon: Users,
                title: 'Experts',
                desc: 'Spawns specialized AI agents to handle specific sub-tasks.',
              },
              {
                n: 5,
                icon: Sparkles,
                title: 'Final Answer',
                desc: 'Combines the results into a clear, comprehensive response.',
              },
            ].map(({ n, icon: Icon, title, desc }) => (
              <div key={n} className="glass-card p-4 text-center border border-border group hover:border-accent-primary transition-colors">
                <div className="inline-flex items-center justify-center w-9 h-9 rounded-lg mb-2 bg-bg-hover group-hover:bg-accent-light transition-colors">
                  <Icon size={16} className="text-accent-secondary" />
                </div>
                <p className="text-xs font-semibold text-text-primary mb-1">
                  {title}
                </p>
                <p className="text-[10px] text-text-muted leading-snug">
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </motion.div>

        <div className="h-8" />
      </motion.div>
    </div>
  )
}
