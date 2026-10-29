/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background) / <alpha-value>)",
        foreground: "hsl(var(--foreground) / <alpha-value>)",
        card: "hsl(var(--card) / <alpha-value>)",
        muted: "hsl(var(--muted) / <alpha-value>)",
        "muted-foreground": "hsl(var(--muted-foreground) / <alpha-value>)",
        border: "hsl(var(--border) / <alpha-value>)",
        primary: "hsl(var(--primary) / <alpha-value>)",
        accent: "hsl(var(--accent) / <alpha-value>)",
        "primary-foreground": "hsl(var(--primary-foreground) / <alpha-value>)",
        violation: "hsl(var(--violation) / <alpha-value>)",
        caution: "hsl(var(--caution) / <alpha-value>)",
        legal: "hsl(var(--legal) / <alpha-value>)",
      },
      borderRadius: { lg: "0.75rem", md: "0.5rem", sm: "0.375rem" },
      boxShadow: {
        lift: "0 10px 30px -12px hsl(var(--shadow) / 0.45)",
        glow: "0 0 0 1px hsl(var(--violation) / 0.5), 0 0 28px -4px hsl(var(--violation) / 0.55)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        pulseLight: { "0%,100%": { opacity: "1", transform: "scale(1)" }, "50%": { opacity: "0.55", transform: "scale(0.94)" } },
      },
      animation: { shimmer: "shimmer 1.6s infinite", pulseLight: "pulseLight 1.4s ease-in-out infinite" },
    },
  },
  plugins: [],
};
