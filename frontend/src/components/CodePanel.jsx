import { useState } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import codeTheme from '../styles/codeTheme'
import { Copy, Check, AlertTriangle, RefreshCw } from 'lucide-react'

export default function CodePanel({ code, error, iterations }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!code) {
    return (
      <div className="text-center py-8 text-text-muted text-xs">
        No orchestration code was generated for this response.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-text-muted uppercase tracking-wider">
            Generated Orchestration Script
          </span>
          {iterations > 1 && (
            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-accent-amber/15 text-accent-amber">
              <RefreshCw size={9} />
              Retried {iterations - 1}x
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] text-text-muted hover:text-text-primary hover:bg-bg-hover/70 transition-all"
        >
          {copied ? (
            <Check size={11} className="text-accent-green" />
          ) : (
            <Copy size={11} />
          )}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>

      {/* Code */}
      <div className="code-panel">
        <SyntaxHighlighter
          language="python"
          style={codeTheme}
          showLineNumbers
          wrapLongLines
          lineNumberStyle={{
            color: '#484f58',
            fontSize: '10px',
            paddingRight: '16px',
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-accent-rose/10 border border-accent-rose/20">
          <AlertTriangle
            size={14}
            className="text-accent-rose flex-shrink-0 mt-0.5"
          />
          <div>
            <p className="text-[11px] font-medium text-accent-rose">
              Execution Error
            </p>
            <p className="text-[11px] text-text-secondary mt-0.5 font-mono">
              {error}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
