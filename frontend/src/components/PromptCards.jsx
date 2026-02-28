import { motion } from 'framer-motion'
import {
  Code2,
  Palette,
  BarChart3,
  MessageSquare,
  Cpu,
  Lightbulb,
} from 'lucide-react'

const prompts = [
  {
    icon: Code2,
    text: 'Explain quantum computing in simple terms',
    category: 'Technical',
    color: '#2dd4bf',
    complexity: 1,
    showcase: 'Single expert routing',
  },
  {
    icon: Palette,
    text: 'Write a short story about AI discovering emotions',
    category: 'Creative',
    color: '#fb7185',
    complexity: 1,
    showcase: 'Creative expert invocation',
  },
  {
    icon: BarChart3,
    text: 'Compare the pros and cons of remote work vs office work',
    category: 'Analytical',
    color: '#60a5fa',
    complexity: 2,
    showcase: 'Multi-expert parallel analysis',
  },
  {
    icon: Cpu,
    text: 'How do I build a REST API with FastAPI?',
    category: 'Technical',
    color: '#2dd4bf',
    complexity: 2,
    showcase: 'Code generation + explanation',
  },
  {
    icon: MessageSquare,
    text: 'What are the main causes of climate change?',
    category: 'General',
    color: '#c084fc',
    complexity: 1,
    showcase: 'Knowledge synthesis',
  },
  {
    icon: Lightbulb,
    text: 'Generate creative marketing slogans for eco-friendly products',
    category: 'Creative',
    color: '#fb7185',
    complexity: 2,
    showcase: 'Creative + analytical blend',
  },
]

export default function PromptCards({ onSelect }) {
  return (
    <div>
      <h2 className="text-sm font-semibold text-text-secondary mb-4 flex items-center gap-2">
        <Lightbulb size={14} className="text-accent-amber" />
        Try these showcase prompts
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {prompts.map((prompt, i) => (
          <motion.button
            key={i}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSelect(prompt.text)}
            className="glass-card p-4 text-left group hover:border-white/10 transition-all duration-300 cursor-pointer"
          >
            <div className="flex items-start gap-3">
              <div
                className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: `${prompt.color}15` }}
              >
                <prompt.icon size={15} style={{ color: prompt.color }} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-text-primary leading-relaxed group-hover:text-white transition-colors">
                  {prompt.text}
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <span
                    className="text-[9px] font-medium px-1.5 py-0.5 rounded-full"
                    style={{
                      backgroundColor: `${prompt.color}15`,
                      color: prompt.color,
                    }}
                  >
                    {prompt.category}
                  </span>
                  <span className="text-[9px] text-text-muted">
                    {'⚡'.repeat(prompt.complexity)}
                  </span>
                  <span className="text-[9px] text-text-muted ml-auto hidden sm:inline">
                    {prompt.showcase}
                  </span>
                </div>
              </div>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  )
}
