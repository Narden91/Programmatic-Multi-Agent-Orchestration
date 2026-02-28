/**
 * Shared syntax-highlighter style used by Markdown.jsx and CodePanel.jsx.
 * Single source of truth — avoids duplicating the oneDark override.
 */
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'

const codeTheme = {
  ...oneLight,
  'pre[class*="language-"]': {
    ...oneLight['pre[class*="language-"]'],
    background: 'rgba(244, 247, 255, 0.9)',
    borderRadius: '12px',
    border: '1px solid rgba(199, 212, 233, 0.7)',
    fontSize: '12px',
    lineHeight: '1.7',
    margin: 0,
  },
  'code[class*="language-"]': {
    ...oneLight['code[class*="language-"]'],
    fontSize: '12px',
    fontFamily: '"JetBrains Mono", "Fira Code", monospace',
  },
}

export default codeTheme
