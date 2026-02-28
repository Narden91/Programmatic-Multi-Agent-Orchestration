import { motion } from 'framer-motion'
import PipelineAnimation from './PipelineAnimation'
import {
  Sparkles,
  GitBranch,
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
            className="flex items-center gap-2 text-xs text-text-muted hover:text-accent-purple transition-colors"
          >
            <ArrowLeft size={14} />
            Back to Chat
          </button>
        </motion.div>

        {/* Header */}
        <motion.div variants={item} className="space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass border border-accent-purple/20 text-xs text-accent-purple font-medium">
            <Sparkles size={12} />
            Architecture Deep-Dive
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight">
            <span className="gradient-text">How It Works</span>
          </h1>
          <p className="text-base text-text-secondary max-w-2xl leading-relaxed">
            A new paradigm for multi-agent AI systems: the orchestrator writes
            dynamic async Python instead of following a static graph.
          </p>
        </motion.div>

        {/* Pipeline */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <Workflow size={14} className="text-accent-purple" />
            Execution Pipeline
          </h2>
          <PipelineAnimation />
        </motion.div>

        {/* How it works steps */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
            <Layers size={14} className="text-accent-amber" />
            Step by Step
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {[
              {
                n: 1,
                icon: Brain,
                color: '#a78bfa',
                title: 'Orchestrator',
                desc: 'Analyzes your query and decides the strategy',
              },
              {
                n: 2,
                icon: Code2,
                color: '#c084fc',
                title: 'Code Gen',
                desc: 'Writes a tailored async Python script',
              },
              {
                n: 3,
                icon: Shield,
                color: '#fbbf24',
                title: 'Sandbox',
                desc: 'Executes the script in an AST-validated sandbox',
              },
              {
                n: 4,
                icon: Users,
                color: '#2dd4bf',
                title: 'Experts',
                desc: 'Specialized micro-agents are invoked programmatically',
              },
              {
                n: 5,
                icon: Sparkles,
                color: '#fb923c',
                title: 'Answer',
                desc: 'Synthesized result returned with full transparency',
              },
            ].map(({ n, icon: Icon, color, title, desc }) => (
              <div
                key={n}
                className="glass-card p-4 text-center"
                style={{ borderColor: `${color}15` }}
              >
                <div
                  className="inline-flex items-center justify-center w-9 h-9 rounded-lg mb-2"
                  style={{ backgroundColor: `${color}15` }}
                >
                  <Icon size={16} style={{ color }} />
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

        {/* Stats */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <Zap size={14} className="text-accent-teal" />
            Key Capabilities
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                label: 'Orchestration',
                value: 'Dynamic Python',
                detail: 'Unique script generated for every query',
                icon: Code2,
                color: '#a78bfa',
              },
              {
                label: 'Execution',
                value: 'Sandboxed',
                detail: 'AST-validated with timeout protection',
                icon: Shield,
                color: '#fbbf24',
              },
              {
                label: 'Experts',
                value: 'Parallel Async',
                detail: 'Context-compressed, transient micro-agents',
                icon: Users,
                color: '#2dd4bf',
              },
            ].map(({ label, value, detail, icon: Icon, color }) => (
              <div key={label} className="glass-card p-5 text-center">
                <div
                  className="inline-flex items-center justify-center w-10 h-10 rounded-xl mb-3"
                  style={{ backgroundColor: `${color}15` }}
                >
                  <Icon size={18} style={{ color }} />
                </div>
                <p className="text-xs text-text-muted uppercase tracking-wider mb-1">
                  {label}
                </p>
                <p className="text-lg font-bold" style={{ color }}>
                  {value}
                </p>
                <p className="text-[11px] text-text-muted mt-1">{detail}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Comparison */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <GitBranch size={14} className="text-accent-rose" />
            Paradigm Comparison
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className="glass-card p-6"
              style={{ borderColor: 'rgba(251, 113, 133, 0.15)' }}
            >
              <div className="flex items-center gap-2 mb-4">
                <GitBranch size={16} className="text-accent-rose" />
                <h3 className="text-sm font-semibold text-accent-rose">
                  Traditional DAG Routing
                </h3>
              </div>
              <ul className="space-y-2.5 text-xs text-text-secondary">
                {[
                  'Static, developer-defined graphs',
                  'Complex state dictionaries bloat context window',
                  'Same routing for every query type',
                  'Massive boilerplate DAG routing code',
                  'Intermediate outputs fill context → hallucinations',
                ].map((text) => (
                  <li key={text} className="flex items-start gap-2">
                    <span className="text-accent-rose mt-0.5 flex-shrink-0">
                      ✗
                    </span>
                    <span>{text}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass-card p-6 gradient-border border-transparent">
              <div className="flex items-center gap-2 mb-4">
                <Code2 size={16} className="text-accent-purple" />
                <h3 className="text-sm font-semibold text-accent-purple">
                  Code-as-Orchestration
                </h3>
              </div>
              <ul className="space-y-2.5 text-xs text-text-secondary">
                {[
                  ['AI writes a ', 'unique script per query'],
                  ['Variables stay in sandbox — ', '80%+ token savings'],
                  ['Dynamic ', 'parallel/sequential expert calls'],
                  ['Zero boilerplate — ', 'the AI writes its own graph'],
                  ['Only final result enters context — ', 'no hallucination loops'],
                ].map(([prefix, bold], i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-accent-green mt-0.5 flex-shrink-0">
                      ✓
                    </span>
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

        {/* Research angle */}
        <motion.div variants={item}>
          <div className="glass-card p-6 gradient-border border-transparent">
            <h2 className="text-sm font-semibold text-accent-purple mb-3 flex items-center gap-2">
              <Sparkles size={14} />
              Research Angle
            </h2>
            <div className="space-y-3 text-xs text-text-secondary leading-relaxed">
              <p>
                <strong className="text-text-primary">
                  Code-as-Orchestration vs Graph-as-Orchestration:
                </strong>{' '}
                Benchmarking how an LLM writing ephemeral async Python compares
                to standard graph-based routing (LangGraph/AutoGen).
              </p>
              <p>
                <strong className="text-text-primary">
                  Solving the Context Window Problem:
                </strong>{' '}
                Keeping intermediate agent dialogue inside sandbox variables instead
                of LLM context windows saves 80%+ on token costs and eliminates
                hallucination loops.
              </p>
              <p>
                <strong className="text-text-primary">
                  Developer Fatigue:
                </strong>{' '}
                No more massive boilerplate DAG routing code. The AI writes its own
                LangGraph on the fly.
              </p>
            </div>
          </div>
        </motion.div>

        <div className="h-8" />
      </motion.div>
    </div>
  )
}
