import { motion } from 'framer-motion'
import { Brain, Database, Server, CheckCircle2, XCircle, AlertCircle, Clock } from 'lucide-react'

const getIconForType = (type) => {
    if (type === 'agent') return <Brain size={14} className="text-accent-purple" />
    if (type === 'memory') return <Database size={14} className="text-accent-teal" />
    return <Server size={14} className="text-text-muted" />
}

const formatValue = (val) => {
    if (typeof val === 'object') return JSON.stringify(val)
    return String(val)
}

export default function VisualDNA({ trace }) {
    if (!trace || trace.length === 0) {
        return (
            <div className="text-center p-8 text-text-muted text-xs">
                No trace DNA available for this execution.
            </div>
        )
    }

    // Find overall duration
    const start = trace[0]?.start_time || 0
    const end = trace[trace.length - 1]?.end_time || start
    const totalDuration = end > start ? (end - start) * 1000 : 0

    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center text-xs text-text-muted mb-2">
                <h4 className="font-semibold text-text-secondary">Execution Trace</h4>
                <span className="flex items-center gap-1"><Clock size={12} /> {totalDuration > 0 ? totalDuration.toFixed(1) : '< 1'} ms</span>
            </div>

            <div className="relative border-l border-border/50 ml-3 pl-5 space-y-6">
                {trace.map((span, i) => (
                    <motion.div
                        key={span.id || i}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className="relative"
                    >
                        {/* Timeline dot */}
                        <div className="absolute -left-[27px] top-1 w-4 h-4 rounded-full bg-bg-card border-2 border-border/50 flex items-center justify-center shrink-0">
                            {span.error ? (
                                <XCircle size={10} className="text-accent-rose" />
                            ) : (
                                <CheckCircle2 size={10} className="text-accent-teal" />
                            )}
                        </div>

                        {/* Content card */}
                        <div className={`p-3 rounded-xl border ${span.error ? 'border-accent-rose/30 bg-accent-rose/5' : 'border-border/50 bg-bg-card/40'}`}>
                            <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <div className="p-1.5 rounded-lg bg-bg-card border border-border/50">
                                        {getIconForType(span.type)}
                                    </div>
                                    <div>
                                        <span className="text-xs font-semibold text-text-primary capitalize">{span.name}</span>
                                        <span className="block text-[10px] text-text-muted">{span.type} span</span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <span className="text-[10px] font-mono text-text-secondary">{span.durationMs ? span.durationMs.toFixed(1) : 0} ms</span>
                                    {span.metrics?.tokens && (
                                        <span className="block text-[10px] text-accent-teal">{span.metrics.tokens} tokens</span>
                                    )}
                                </div>
                            </div>

                            {/* Inputs/Outputs */}
                            <div className="mt-2 space-y-1.5">
                                {span.inputs && Object.keys(span.inputs).length > 0 && (
                                    <div className="text-[10px]">
                                        <span className="text-text-muted mr-2">INPUTS:</span>
                                        <span className="font-mono text-text-secondary break-all">
                                            {Object.entries(span.inputs).map(([k, v]) => `${k}=${formatValue(v)}`).join(', ')}
                                        </span>
                                    </div>
                                )}
                                {span.error ? (
                                    <div className="text-[10px] text-accent-rose mt-1 flex items-start gap-1">
                                        <AlertCircle size={12} className="shrink-0 mt-0.5" />
                                        <span className="break-all">{span.error}</span>
                                    </div>
                                ) : span.outputs && Object.keys(span.outputs).length > 0 ? (
                                    <div className="text-[10px]">
                                        <span className="text-text-muted mr-2">OUTPUTS:</span>
                                        <span className="font-mono text-text-secondary break-all">
                                            {Object.entries(span.outputs).map(([k, v]) => `${k}=${formatValue(v)}`).join(', ')}
                                        </span>
                                    </div>
                                ) : null}
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    )
}
