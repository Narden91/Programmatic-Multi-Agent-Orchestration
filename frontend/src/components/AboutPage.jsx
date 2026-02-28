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
  Database,
  Timer,
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
            Architecture Deep Dive
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight">
            <span className="gradient-text">What Makes This Different</span>
          </h1>
          <p className="text-base text-text-secondary max-w-3xl leading-relaxed">
            This system shifts orchestration from fixed graph wiring to generated
            program logic. For each query, the orchestrator writes async Python,
            executes it in a sandbox, and programmatically coordinates expert
            agents.
          </p>
        </motion.div>

        <motion.div variants={item} className="glass-card p-5">
          <h2 className="text-sm font-semibold text-text-primary mb-2">
            Novelty in one sentence
          </h2>
          <p className="text-sm text-text-secondary leading-relaxed">
            The AI no longer chooses from a prebuilt route map; it writes and runs
            a purpose-built orchestration script that can use loops, conditionals,
            and parallel `asyncio.gather` calls to solve the query.
          </p>
        </motion.div>

        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3 flex items-center gap-2">
            <Sparkles size={14} className="text-accent-purple" />
            Core Differentiators
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              {
                title: 'Dynamic Tool Calling',
                desc: 'The orchestrator writes a fresh async Python plan per query and invokes transient expert functions at runtime.',
                color: 'text-accent-purple',
                bg: 'bg-accent-purple/10',
              },
              {
                title: 'Validated Sandbox Execution',
                desc: 'Generated code is AST-validated and executed with restricted builtins and explicit timeout controls.',
                color: 'text-accent-amber',
                bg: 'bg-accent-amber/10',
              },
              {
                title: 'Context-Efficient Reasoning',
                desc: 'Intermediate reasoning stays in sandbox variables instead of bloating the shared chat transcript.',
                color: 'text-accent-teal',
                bg: 'bg-accent-teal/10',
              },
            ].map(({ title, desc, color, bg }) => (
              <div key={title} className="glass-card p-4">
                <div
                  className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center mb-2.5`}
                >
                  <Sparkles size={14} className={color} />
                </div>
                <p className="text-sm font-semibold text-text-primary mb-1.5">
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
            <Workflow size={14} className="text-accent-purple" />
            Execution Pipeline
          </h2>
          <PipelineAnimation />
        </motion.div>

        {/* How it works steps */}
        <motion.div variants={item}>
          <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
            <Layers size={14} className="text-accent-amber" />
            How It Is Achieved
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {[
              {
                n: 1,
                icon: Brain,
                color: '#a78bfa',
                title: 'Orchestrator',
                desc: 'Chooses a strategy and drafts a query-specific orchestration program',
              },
              {
                n: 2,
                icon: Code2,
                color: '#c084fc',
                title: 'Code Gen',
                desc: 'Generates async Python with tool calls, loops, and conditional routing',
              },
              {
                n: 3,
                icon: Shield,
                color: '#fbbf24',
                title: 'Sandbox',
                desc: 'Validates AST, restricts execution, and enforces runtime limits',
              },
              {
                n: 4,
                icon: Users,
                color: '#2dd4bf',
                title: 'Experts',
                desc: 'Spawns transient specialist agents via async tool functions',
              },
              {
                n: 5,
                icon: Sparkles,
                color: '#fb923c',
                title: 'Answer',
                desc: 'Returns synthesis, generated code, and execution metadata',
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
            Practical Advantages
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                label: 'Adaptive Logic',
                value: 'Per-query script',
                detail: 'No one-size-fits-all graph path',
                icon: Code2,
                color: '#a78bfa',
              },
              {
                label: 'Safety',
                value: 'Sandbox controls',
                detail: 'AST checks, restricted builtins, timeout enforcement',
                icon: Shield,
                color: '#fbbf24',
              },
              {
                label: 'Context Efficiency',
                value: 'Compressed traces',
                detail: 'Intermediate steps stay in runtime variables',
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
            Routing Paradigms
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className="glass-card p-6"
              style={{ borderColor: 'rgba(251, 113, 133, 0.15)' }}
            >
              <div className="flex items-center gap-2 mb-4">
                <GitBranch size={16} className="text-accent-rose" />
                <h3 className="text-sm font-semibold text-accent-rose">
                  Static Graph Routing
                </h3>
              </div>
              <ul className="space-y-2.5 text-xs text-text-secondary">
                {[
                  'Developer-maintained edge logic and static node wiring',
                  'Low flexibility for query-specific control flow',
                  'Intermediates often accumulate in shared context state',
                  'High orchestration boilerplate and maintenance overhead',
                  'Parallel/sequential strategy changes require manual graph edits',
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
                  ['AI generates a ', 'fresh async orchestration script per request'],
                  ['Tool functions enable ', 'dynamic expert invocation at runtime'],
                  ['Native Python control flow handles ', 'branching and iteration naturally'],
                  ['Execution remains inside a ', 'validated and bounded sandbox'],
                  ['Only the synthesized output and metadata ', 'return to graph state'],
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
              Research and Measurement
            </h2>
            <div className="space-y-3 text-xs text-text-secondary leading-relaxed">
              <p>
                <strong className="text-text-primary">Benchmark axis 1:</strong>{' '}
                generated-code orchestration versus fixed graph routing on quality,
                flexibility, and implementation overhead.
              </p>
              <p>
                <strong className="text-text-primary">Benchmark axis 2:</strong>{' '}
                context growth and token usage when intermediate deliberation is kept
                inside sandbox variables instead of the chat transcript.
              </p>
              <p>
                <strong className="text-text-primary">Benchmark axis 3:</strong>{' '}
                developer effort reduction by replacing manual DAG boilerplate with
                model-authored orchestration code.
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div variants={item}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              {
                icon: Database,
                title: 'Context Compression',
                desc: 'Intermediate reasoning remains in runtime variables, not full transcript memory.',
                color: 'text-accent-blue',
                bg: 'bg-accent-blue/10',
              },
              {
                icon: Timer,
                title: 'Execution Guardrails',
                desc: 'AST checks and bounded runtime enforce predictable, safer code execution.',
                color: 'text-accent-amber',
                bg: 'bg-accent-amber/10',
              },
              {
                icon: Users,
                title: 'Transient Experts',
                desc: 'Specialist agents are spawned on demand and discarded after response synthesis.',
                color: 'text-accent-teal',
                bg: 'bg-accent-teal/10',
              },
            ].map(({ icon: Icon, title, desc, color, bg }) => (
              <div key={title} className="glass-card p-4">
                <div className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center mb-2.5`}>
                  <Icon size={14} className={color} />
                </div>
                <p className="text-sm font-semibold text-text-primary mb-1.5">
                  {title}
                </p>
                <p className="text-xs text-text-secondary leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        <div className="h-8" />
      </motion.div>
    </div>
  )
}
