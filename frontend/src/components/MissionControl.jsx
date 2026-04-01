import { lazy, Suspense, useState } from 'react'
import { Code2, Workflow, BarChart3, Users } from 'lucide-react'
import CodePanel from './CodePanel'
import FlowGraph from './FlowGraph'
import ExpertCards from './ExpertCards'
import VisualDNA from './VisualDNA'

const TokenChart = lazy(() => import('./TokenChart'))

const tabs = [
  { id: 'code', label: 'Code', icon: Code2 },
  { id: 'flow', label: 'Agent Flow', icon: Workflow },
  { id: 'dna', label: 'Visual DNA', icon: Code2 },
  { id: 'tokens', label: 'Tokens', icon: BarChart3 },
  { id: 'experts', label: 'Experts', icon: Users },
]

export default function MissionControl({ data, id }) {
  const [activeTab, setActiveTab] = useState('code')

  return (
    <div className="glass-card overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-border/50">
        {tabs.map(({ id: tabId, label, icon: Icon }) => (
          <button
            key={tabId}
            onClick={() => setActiveTab(tabId)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-[11px] font-medium transition-all relative ${activeTab === tabId
                ? 'text-accent-purple'
                : 'text-text-muted hover:text-text-secondary'
              }`}
          >
            <Icon size={12} />
            {label}
            {activeTab === tabId && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-purple rounded-full" />
            )}
          </button>
        ))}

        {/* Badges */}
        <div className="ml-auto flex items-center gap-2 px-3">
          {data.code_execution_iterations > 1 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-amber/15 text-accent-amber border border-accent-amber/30">
              {data.code_execution_iterations} iterations
            </span>
          )}
          {data.token_usage?.total_tokens > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-teal/15 text-accent-teal border border-accent-teal/30">
              {data.token_usage.total_tokens.toLocaleString()} tokens
            </span>
          )}
        </div>
      </div>

      {/* Tab content */}
      <div className="p-4">
        {activeTab === 'code' && (
          <CodePanel
            code={data.generated_code}
            error={data.code_execution_error}
            iterations={data.code_execution_iterations}
          />
        )}
        {activeTab === 'flow' && (
          <FlowGraph
            experts={data.selected_experts}
          />
        )}
        {activeTab === 'tokens' && (
          <Suspense fallback={<div className="text-center py-8 text-text-muted text-xs">Loading token chart...</div>}>
            <TokenChart tokenUsage={data.token_usage} />
          </Suspense>
        )}
        {activeTab === 'experts' && (
          <ExpertCards
            expertResponses={data.expert_responses}
            selectedExperts={data.selected_experts}
          />
        )}
        {activeTab === 'dna' && (
          <VisualDNA trace={data.trace_dna} />
        )}
      </div>
    </div>
  )
}
