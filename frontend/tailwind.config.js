/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f4ff',
          100: '#dbe4ff',
          500: '#3b5bdb',
          600: '#364fc7',
          700: '#2f44ad',
          800: '#1e3a8a',
          900: '#1a3073',
        },
        saffron: {
          400: '#ff9933',
          500: '#f97316',
          600: '#ea6c00',
        },
      },
    },
  },
  plugins: [],
}
