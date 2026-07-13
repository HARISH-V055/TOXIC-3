/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // MolXAI Brand Colors
        primary: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
          950: '#083344',
        },
        accent: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        surface: {
          900: '#020617',
          850: '#0a1020',
          800: '#0f172a',
          750: '#131e33',
          700: '#1e293b',
          600: '#253352',
          500: '#334155',
        },
        toxic: {
          light: '#fca5a5',
          DEFAULT: '#ef4444',
          dark: '#b91c1c',
        },
        safe: {
          light: '#86efac',
          DEFAULT: '#22c55e',
          dark: '#15803d',
        },
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
        'gradient-dark': 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%)',
        'gradient-card': 'linear-gradient(135deg, rgba(6,182,212,0.05) 0%, rgba(59,130,246,0.05) 100%)',
        'gradient-glow': 'radial-gradient(ellipse at top, rgba(6,182,212,0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        'glow-cyan': '0 0 30px rgba(6, 182, 212, 0.3)',
        'glow-blue': '0 0 30px rgba(59, 130, 246, 0.3)',
        'glow-sm': '0 0 10px rgba(6, 182, 212, 0.2)',
        'card': '0 4px 24px rgba(0, 0, 0, 0.4)',
        'card-hover': '0 8px 40px rgba(6, 182, 212, 0.15)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'slide-up': 'slideUp 0.5s ease-out forwards',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'spin-slow': 'spin 4s linear infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(6, 182, 212, 0.2)' },
          '50%': { boxShadow: '0 0 30px rgba(6, 182, 212, 0.5)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
    },
  },
  plugins: [],
};
