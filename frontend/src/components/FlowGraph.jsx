import { motion } from 'framer-motion'

const NODE_COLORS = {
  query: { bg: '#60a5fa', glow: 'rgba(96, 165, 250, 0.3)' },
  orchestrator: { bg: '#a78bfa', glow: 'rgba(167, 139, 250, 0.3)' },
  sandbox: { bg: '#fbbf24', glow: 'rgba(251, 191, 36, 0.3)' },
  technical: { bg: '#2dd4bf', glow: 'rgba(45, 212, 191, 0.3)' },
  creative: { bg: '#fb7185', glow: 'rgba(251, 113, 133, 0.3)' },
  analytical: { bg: '#60a5fa', glow: 'rgba(96, 165, 250, 0.3)' },
  general: { bg: '#c084fc', glow: 'rgba(192, 132, 252, 0.3)' },
  answer: { bg: '#fb923c', glow: 'rgba(251, 147, 60, 0.3)' },
}

function FlowNode({ x, y, label, colorKey, delay = 0 }) {
  const { bg, glow } = NODE_COLORS[colorKey] || NODE_COLORS.answer
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.4, type: 'spring' }}
      style={{ transformOrigin: `${x}px ${y}px` }}
    >
      {/* Glow */}
      <circle cx={x} cy={y} r={28} fill={glow} opacity={0.3}>
        <animate
          attributeName="r"
          values="28;32;28"
          dur="3s"
          repeatCount="indefinite"
        />
        <animate
          attributeName="opacity"
          values="0.3;0.5;0.3"
          dur="3s"
          repeatCount="indefinite"
        />
      </circle>
      {/* Node circle */}
      <circle
        cx={x}
        cy={y}
        r={22}
        fill={`${bg}20`}
        stroke={bg}
        strokeWidth={1.5}
      />
      {/* Label */}
      <text
        x={x}
        y={y + 38}
        textAnchor="middle"
        fill="#8b949e"
        fontSize={10}
        fontFamily="Plus Jakarta Sans, sans-serif"
        fontWeight="500"
      >
        {label}
      </text>
      {/* Icon letter */}
      <text
        x={x}
        y={y + 5}
        textAnchor="middle"
        fill={bg}
        fontSize={13}
        fontFamily="Plus Jakarta Sans, sans-serif"
        fontWeight="700"
      >
        {label.charAt(0).toUpperCase()}
      </text>
    </motion.g>
  )
}

function FlowEdge({ x1, y1, x2, y2, delay = 0 }) {
  return (
    <motion.line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="url(#edgeGrad)"
      strokeWidth={1.5}
      strokeDasharray="6 3"
      initial={{ opacity: 0 }}
      animate={{ opacity: 0.6 }}
      transition={{ delay }}
    >
      <animate
        attributeName="stroke-dashoffset"
        from="20"
        to="0"
        dur="1.5s"
        repeatCount="indefinite"
      />
    </motion.line>
  )
}

function calculateEdgePoints(x1, y1, x2, y2, radius) {
  const angle = Math.atan2(y2 - y1, x2 - x1)
  return {
    startX: x1 + radius * Math.cos(angle),
    startY: y1 + radius * Math.sin(angle),
    endX: x2 - radius * Math.cos(angle),
    endY: y2 - radius * Math.sin(angle),
  }
}

export default function FlowGraph({ experts = [] }) {
  if (!experts.length) {
    return (
      <div className="text-center py-8 text-text-muted text-xs">
        No routing data available for this response.
      </div>
    )
  }

  const padding = 60
  const nodeRadius = 22
  const expertCount = experts.length
  const width = 620
  const expertSpread = Math.min(expertCount * 70, 220)
  const height = 140 + (expertCount > 1 ? expertSpread : 0)
  const centerY = height / 2

  const positions = {
    query: { x: padding, y: centerY },
    orchestrator: { x: padding + 120, y: centerY },
    sandbox: { x: padding + 240, y: centerY },
    answer: { x: width - padding, y: centerY },
  }

  const expertPositions = experts.map((_, i) => ({
    x: padding + 370,
    y:
      expertCount === 1
        ? centerY
        : centerY -
        expertSpread / 2 +
        (expertSpread / (expertCount - 1)) * i,
  }))

  return (
    <div className="overflow-x-auto py-2">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="mx-auto"
      >
        <defs>
          <linearGradient
            id="edgeGrad"
            x1="0"
            y1="0"
            x2={width}
            y2="0"
            gradientUnits="userSpaceOnUse"
          >
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.5" />
          </linearGradient>
        </defs>

        {/* Edges: Query → Orchestrator → Sandbox */}
        <FlowEdge
          x1={positions.query.x + nodeRadius}
          y1={positions.query.y}
          x2={positions.orchestrator.x - nodeRadius}
          y2={positions.orchestrator.y}
          delay={0.2}
        />
        <FlowEdge
          x1={positions.orchestrator.x + nodeRadius}
          y1={positions.orchestrator.y}
          x2={positions.sandbox.x - nodeRadius}
          y2={positions.sandbox.y}
          delay={0.4}
        />

        {/* Edges: Sandbox → Experts */}
        {expertPositions.map((pos, i) => {
          const edge = calculateEdgePoints(
            positions.sandbox.x,
            positions.sandbox.y,
            pos.x,
            pos.y,
            nodeRadius
          )
          return (
            <FlowEdge
              key={`s-e-${i}`}
              x1={edge.startX}
              y1={edge.startY}
              x2={edge.endX}
              y2={edge.endY}
              delay={0.6 + i * 0.1}
            />
          )
        })}

        {/* Edges: Experts → Answer */}
        {expertPositions.map((pos, i) => {
          const edge = calculateEdgePoints(
            pos.x,
            pos.y,
            positions.answer.x,
            positions.answer.y,
            nodeRadius
          )
          return (
            <FlowEdge
              key={`e-a-${i}`}
              x1={edge.startX}
              y1={edge.startY}
              x2={edge.endX}
              y2={edge.endY}
              delay={0.8 + i * 0.1}
            />
          )
        })}

        {/* Nodes */}
        <FlowNode
          x={positions.query.x}
          y={positions.query.y}
          label="Query"
          colorKey="query"
          delay={0}
        />
        <FlowNode
          x={positions.orchestrator.x}
          y={positions.orchestrator.y}
          label="Orchestrator"
          colorKey="orchestrator"
          delay={0.15}
        />
        <FlowNode
          x={positions.sandbox.x}
          y={positions.sandbox.y}
          label="Sandbox"
          colorKey="sandbox"
          delay={0.3}
        />
        {experts.map((expert, i) => (
          <FlowNode
            key={expert}
            x={expertPositions[i].x}
            y={expertPositions[i].y}
            label={expert}
            colorKey={expert}
            delay={0.45 + i * 0.1}
          />
        ))}
        <FlowNode
          x={positions.answer.x}
          y={positions.answer.y}
          label="Answer"
          colorKey="answer"
          delay={0.6 + experts.length * 0.1}
        />
      </svg>
    </div>
  )
}
