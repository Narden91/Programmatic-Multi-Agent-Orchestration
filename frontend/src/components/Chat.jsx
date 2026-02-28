import { useRef, useEffect } from 'react'
import ChatMessage from './ChatMessage'
import LoadingIndicator from './LoadingIndicator'

export default function Chat({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  return (
    <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} index={i} />
        ))}
        {isLoading && <LoadingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
