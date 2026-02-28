import typography from '@tailwindcss/typography'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: '#f5f7fc',
          surface: '#f1f4ff',
          card: '#f4efff',
          hover: '#eef2fb',
        },
        border: {
          DEFAULT: '#d7dfed',
          light: '#c4d0e3',
        },
        text: {
          primary: '#1f2940',
          secondary: '#4d5c78',
          muted: '#7c89a3',
        },
        accent: {
          purple: '#8f81f7',
          amber: '#e5ba67',
          teal: '#5eb7ab',
          rose: '#df88a6',
          blue: '#7da7e7',
          violet: '#ac8de8',
          orange: '#e6a480',
          green: '#70bf99',
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        glow: 'glow 2s ease-in-out infinite alternate',
        flow: 'flow 2s linear infinite',
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(167, 139, 250, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(167, 139, 250, 0.6)' },
        },
        flow: {
          '0%': { strokeDashoffset: '20' },
          '100%': { strokeDashoffset: '0' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [typography],
}
