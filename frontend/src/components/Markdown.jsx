import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import codeTheme from '../styles/codeTheme'

export default function Markdown({ children }) {
  if (!children) return null

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '')
          const isBlock = node?.position?.start?.line !== node?.position?.end?.line || match
          if (isBlock && match) {
            return (
              <SyntaxHighlighter
                style={codeTheme}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            )
          }
          return (
            <code
              className="px-1.5 py-0.5 rounded-md bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-[0.85em] font-mono"
              {...props}
            >
              {children}
            </code>
          )
        },
        p({ children }) {
          return <p className="mb-3 leading-relaxed last:mb-0">{children}</p>
        },
        h1({ children }) {
          return (
            <h1 className="text-xl font-bold text-text-primary mt-5 mb-3">
              {children}
            </h1>
          )
        },
        h2({ children }) {
          return (
            <h2 className="text-lg font-bold text-text-primary mt-4 mb-2">
              {children}
            </h2>
          )
        },
        h3({ children }) {
          return (
            <h3 className="text-base font-semibold text-text-primary mt-3 mb-2">
              {children}
            </h3>
          )
        },
        ul({ children }) {
          return (
            <ul className="list-disc list-inside mb-3 space-y-1 text-text-secondary">
              {children}
            </ul>
          )
        },
        ol({ children }) {
          return (
            <ol className="list-decimal list-inside mb-3 space-y-1 text-text-secondary">
              {children}
            </ol>
          )
        },
        li({ children }) {
          return <li className="text-text-primary leading-relaxed">{children}</li>
        },
        a({ href, children }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-purple hover:text-accent-blue transition-colors underline underline-offset-2"
            >
              {children}
            </a>
          )
        },
        blockquote({ children }) {
          return (
            <blockquote className="border-l-2 border-accent-purple/40 pl-4 my-3 text-text-secondary italic">
              {children}
            </blockquote>
          )
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-3">
              <table className="w-full text-sm border border-border rounded-lg overflow-hidden">
                {children}
              </table>
            </div>
          )
        },
        thead({ children }) {
          return <thead className="bg-bg-card text-text-secondary">{children}</thead>
        },
        th({ children }) {
          return <th className="px-3 py-2 text-left text-xs font-semibold">{children}</th>
        },
        td({ children }) {
          return <td className="px-3 py-2 border-t border-border/50">{children}</td>
        },
        strong({ children }) {
          return <strong className="font-semibold text-text-primary">{children}</strong>
        },
        hr() {
          return <hr className="border-border my-4" />
        },
      }}
    >
      {children}
    </ReactMarkdown>
  )
}
