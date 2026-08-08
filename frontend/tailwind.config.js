/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        surface: '#171717',
        border: 'rgba(255,255,255,0.1)',
        accent: '#5e6ad2',
      }
    },
  },
  plugins: [],
}
