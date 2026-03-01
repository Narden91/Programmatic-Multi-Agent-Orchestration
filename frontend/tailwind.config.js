import typography from '@tailwindcss/typography'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: '#F2F5FB',
          surface: '#ffffff',
          card: '#ffffff',
          hover: '#E1E8F4',
        },
        border: {
          DEFAULT: '#E1E8F4',
          light: '#F2F5FB',
        },
        text: {
          primary: '#16365D',
          secondary: '#4075AD',
          muted: '#8da6c4',
        },
        accent: {
          primary: '#4075AD',
          secondary: '#16365D',
          light: '#E1E8F4',
          purple: '#4075AD',
          amber: '#16365D',
          teal: '#4075AD',
          rose: '#16365D',
          blue: '#4075AD',
          violet: '#16365D',
          orange: '#4075AD',
          green: '#16365D',
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
