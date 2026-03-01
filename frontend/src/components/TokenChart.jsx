import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

const COLORS = {
  orchestrator: '#a78bfa',
  technical: '#2dd4bf',
  creative: '#fb7185',
  analytical: '#60a5fa',
  general: '#c084fc',
  sandbox: '#fbbf24',
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card p-2.5 text-xs shadow-xl">
      <p className="text-text-primary font-medium mb-1">
        {payload[0].payload.name}
      </p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }} className="text-[11px]">
          {entry.name}: {entry.value.toLocaleString()}
        </p>
      ))}
    </div>
  )
}

export default function TokenChart({ tokenUsage }) {
  if (
    !tokenUsage ||
    !tokenUsage.by_agent ||
    Object.keys(tokenUsage.by_agent).length === 0
  ) {
    return (
      <div className="text-center py-8 text-text-muted text-xs">
        No token usage data available.
      </div>
    )
  }

  const byAgent = tokenUsage.by_agent
  const totalTokens = tokenUsage.total_tokens || 0
  const cost = tokenUsage.estimated_cost_usd || 0

  const normalizeName = (n) => n.replace(/^(agent_|expert_)/i, '').toLowerCase()

  const barData = Object.entries(byAgent).map(([name, data]) => {
    const rawName = normalizeName(name)
    return {
      name: rawName.charAt(0).toUpperCase() + rawName.slice(1),
      input: data.input,
      output: data.output,
      fill: COLORS[rawName] || '#8b949e',
    }
  })

  const pieData = Object.entries(byAgent).map(([name, data]) => {
    const rawName = normalizeName(name)
    return {
      name: rawName.charAt(0).toUpperCase() + rawName.slice(1),
      value: data.input + data.output,
      fill: COLORS[rawName] || '#8b949e',
    }
  })

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-4 rounded-xl bg-bg/60 border border-border/50 text-center flex flex-col items-center justify-center">
          <p className="text-3xl font-bold text-accent-blue mb-1">
            {totalTokens.toLocaleString()}
          </p>
          <p className="text-xs text-text-muted">Total Tokens</p>
        </div>
        <div className="p-4 rounded-xl bg-bg/60 border border-border/50 text-center flex flex-col items-center justify-center">
          <p className="text-3xl font-bold text-text-primary mb-1">
            ${cost.toFixed(4)}
          </p>
          <p className="text-xs text-text-muted">Estimated Cost</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Donut */}
        <div className="p-3 rounded-xl bg-bg/60 border border-border/50">
          <p className="text-[10px] font-medium text-text-muted uppercase tracking-wider mb-2">
            Distribution
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={75}
                paddingAngle={3}
                dataKey="value"
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend
                formatter={(value) => (
                  <span className="text-text-secondary text-[10px]">
                    {value}
                  </span>
                )}
                iconSize={8}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar chart */}
        <div className="p-3 rounded-xl bg-bg/60 border border-border/50">
          <p className="text-[10px] font-medium text-text-muted uppercase tracking-wider mb-2">
            Input vs Output
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={barData}
              layout="vertical"
              margin={{ left: 0, right: 10 }}
            >
              <XAxis
                type="number"
                tick={{ fontSize: 10, fill: '#8b949e' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 10, fill: '#8b949e' }}
                axisLine={false}
                tickLine={false}
                width={80}
              />
              <Tooltip content={<CustomTooltip />} />
              {/* We no longer hardcode the stack fill here. We use Cell to dynamically color each bar segment */}
              <Bar dataKey="input" stackId="a" name="Input">
                {barData.map((entry, index) => (
                  <Cell key={`cell-in-${index}`} fill={entry.fill} />
                ))}
              </Bar>
              <Bar dataKey="output" stackId="a" radius={[0, 4, 4, 0]} name="Output">
                {barData.map((entry, index) => (
                  <Cell key={`cell-out-${index}`} fill={`${entry.fill}80`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
