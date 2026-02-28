import { useState, useReducer, useEffect, useCallback } from 'react'
import Layout from './components/Layout'
import Sidebar from './components/Sidebar'
import Hero from './components/Hero'
import Chat from './components/Chat'
import ChatInput from './components/ChatInput'
import { sendQuery, getModels, getConfig } from './api/client'

const initialState = {
  messages: [],
  config: {
    apiKey: '',
    model: 'meta-llama/llama-4-maverick-17b-128e-instruct',
    hasEnvKey: false,
  },
  isLoading: false,
  error: null,
  sessionStats: {
    queriesProcessed: 0,
    totalTokens: 0,
    totalExperts: 0,
  },
  availableModels: [],
  sidebarOpen: false,
}

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_USER_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, { role: 'user', content: action.payload }],
        isLoading: true,
        error: null,
      }
    case 'ADD_ASSISTANT_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, { role: 'assistant', ...action.payload }],
        isLoading: false,
        sessionStats: {
          queriesProcessed: state.sessionStats.queriesProcessed + 1,
          totalTokens:
            state.sessionStats.totalTokens +
            (action.payload.token_usage?.total_tokens || 0),
          totalExperts:
            state.sessionStats.totalExperts +
            (action.payload.selected_experts?.length || 0),
        },
      }
    case 'SET_ERROR':
      return { ...state, isLoading: false, error: action.payload }
    case 'SET_CONFIG':
      return { ...state, config: { ...state.config, ...action.payload } }
    case 'SET_MODELS':
      return { ...state, availableModels: action.payload }
    case 'CLEAR_MESSAGES':
      return {
        ...state,
        messages: [],
        sessionStats: { queriesProcessed: 0, totalTokens: 0, totalExperts: 0 },
      }
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen }
    default:
      return state
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    const saved = localStorage.getItem('moe_api_key')
    if (saved) dispatch({ type: 'SET_CONFIG', payload: { apiKey: saved } })

    getConfig()
      .then((cfg) => {
        dispatch({
          type: 'SET_CONFIG',
          payload: { hasEnvKey: cfg.has_env_api_key },
        })
      })
      .catch(() => {})

    getModels()
      .then((data) => {
        dispatch({ type: 'SET_MODELS', payload: data.models })
      })
      .catch(() => {})
  }, [])

  const handleSend = useCallback(
    async (query) => {
      dispatch({ type: 'ADD_USER_MESSAGE', payload: query })
      try {
        const result = await sendQuery(
          query,
          state.config.apiKey,
          state.config.model,
        )
        dispatch({ type: 'ADD_ASSISTANT_MESSAGE', payload: result })
      } catch (err) {
        dispatch({ type: 'SET_ERROR', payload: err.message })
        dispatch({
          type: 'ADD_ASSISTANT_MESSAGE',
          payload: {
            content: `Error: ${err.message}`,
            final_answer: `Error: ${err.message}`,
            generated_code: '',
            selected_experts: [],
            expert_responses: {},
            token_usage: {},
            code_execution_iterations: 0,
            code_execution_error: err.message,
            execution_plan: {},
          },
        })
      }
    },
    [state.config.apiKey, state.config.model],
  )

  const handleSetApiKey = useCallback((key) => {
    localStorage.setItem('moe_api_key', key)
    dispatch({ type: 'SET_CONFIG', payload: { apiKey: key } })
  }, [])

  const handleSetModel = useCallback((model) => {
    dispatch({ type: 'SET_CONFIG', payload: { model } })
  }, [])

  const hasMessages = state.messages.length > 0
  const needsApiKey = !state.config.apiKey && !state.config.hasEnvKey

  return (
    <Layout
      sidebarOpen={state.sidebarOpen}
      onToggleSidebar={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
      sidebar={
        <Sidebar
          config={state.config}
          models={state.availableModels}
          stats={state.sessionStats}
          onSetApiKey={handleSetApiKey}
          onSetModel={handleSetModel}
          onClear={() => dispatch({ type: 'CLEAR_MESSAGES' })}
          isOpen={state.sidebarOpen}
          onToggle={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
        />
      }
    >
      <div className="flex-1 flex flex-col overflow-hidden">
        {!hasMessages ? (
          <div className="flex-1 overflow-y-auto">
            <Hero onSelectPrompt={handleSend} />
          </div>
        ) : (
          <Chat messages={state.messages} isLoading={state.isLoading} />
        )}
        <ChatInput
          onSend={handleSend}
          isLoading={state.isLoading}
          disabled={needsApiKey}
          placeholder={
            needsApiKey
              ? 'Add your Groq API key in the sidebar to get started...'
              : 'Ask anything — the AI writes its own orchestration code...'
          }
        />
      </div>
    </Layout>
  )
}
