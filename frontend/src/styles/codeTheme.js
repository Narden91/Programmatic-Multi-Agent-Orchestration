/**
 * Shared syntax-highlighter style used by Markdown.jsx and CodePanel.jsx.
 * Single source of truth — avoids duplicating the oneDark override.
 */
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

const codeTheme = {
  ...oneDark,
  'pre[class*="language-"]': {
    ...oneDark['pre[class*="language-"]'],
    background: 'rgba(6, 8, 15, 0.8)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.06)',
    fontSize: '12px',
    lineHeight: '1.7',
    margin: 0,
  },
  'code[class*="language-"]': {
    ...oneDark['code[class*="language-"]'],
    fontSize: '12px',
    fontFamily: '"JetBrains Mono", "Fira Code", monospace',
  },
}

export default codeTheme
