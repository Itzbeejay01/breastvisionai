/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{jsx,js,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Verdict semantics (never decorative)
        malignant: "#ba1a1a",
        benign: "#006c44",
        moderate: "#eab308",
        // Material 3 teal palette (light)
        primary: "#005f6c",
        "primary-container": "#daf8ff",
        "on-primary": "#ffffff",
        "on-primary-container": "#004e59",
        "primary-fixed": "#a0efff",
        "primary-fixed-dim": "#7ad4e5",
        secondary: "#006c44",
        "secondary-container": "#8df8bd",
        "on-secondary": "#ffffff",
        "on-secondary-container": "#007349",
        tertiary: "#a62127",
        error: "#ba1a1a",
        "error-container": "#ffdad6",
        "on-error": "#ffffff",
        background: "#f8fafb",
        surface: "#f8fafb",
        "surface-bright": "#f8fafb",
        "surface-dim": "#d8dadb",
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f2f4f5",
        "surface-container": "#eceeef",
        "surface-container-high": "#e6e8e9",
        "surface-container-highest": "#e1e3e4",
        "on-surface": "#191c1d",
        "on-surface-variant": "#3e494b",
        "surface-variant": "#e1e3e4",
        outline: "#6e797b",
        "outline-variant": "#bdc8cb",
        "inverse-surface": "#2e3132",
        "inverse-on-surface": "#eff1f2",
        "inverse-primary": "#7ad4e5",
        "on-background": "#191c1d",
      },
      fontFamily: {
        sans: ["Plus Jakarta Sans", "system-ui", "sans-serif"],
        display: ["Space Grotesk", "Plus Jakarta Sans", "sans-serif"],
        headline: ["Space Grotesk", "Plus Jakarta Sans", "sans-serif"],
      },
      fontSize: {
        "label-caps": ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "700" }],
        "body-sm": ["15px", { lineHeight: "22px", fontWeight: "400" }],
        "body-md": ["17px", { lineHeight: "27px", fontWeight: "400" }],
        "body-lg": ["19px", { lineHeight: "30px", fontWeight: "400" }],
        "data-num": ["16px", { lineHeight: "24px", fontWeight: "500" }],
        "headline-sm": ["20px", { lineHeight: "30px", fontWeight: "600" }],
        "headline-md": ["24px", { lineHeight: "34px", fontWeight: "600" }],
        "headline-lg": ["32px", { lineHeight: "42px", letterSpacing: "-0.02em", fontWeight: "600" }],
        "display-lg": ["clamp(2.5rem, 5vw, 4.5rem)", { lineHeight: "1.05", letterSpacing: "-0.045em", fontWeight: "700" }],
      },
      spacing: {
        gutter: "24px",
        unit: "8px",
      },
      maxWidth: {
        "container-max": "1440px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(25,28,29,0.04), 0 4px 12px rgba(25,28,29,0.06)",
        lifted: "0 2px 4px rgba(25,28,29,0.05), 0 12px 28px rgba(25,28,29,0.12)",
      },
      keyframes: {
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        scan: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(200%)" },
        },
      },
      animation: {
        "slide-up": "slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "pulse-soft": "pulseSoft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        scan: "scan 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};
