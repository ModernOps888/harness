/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        harness: {
          dark: '#0a0d14',
          card: '#111726',
          border: '#1e293b',
          accent: '#10b981',
          cyan: '#06b6d4',
          amber: '#f59e0b',
        },
      },
    },
  },
  plugins: [],
};
