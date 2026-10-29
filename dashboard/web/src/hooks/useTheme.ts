import { useCallback, useState } from "react";

export type Theme = "dark" | "light";

function read(): Theme {
  const attr = document.documentElement.dataset.theme;
  return attr === "light" ? "light" : "dark";
}

/** Dark by default; toggling animates colours briefly and persists the choice. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(read);

  const toggle = useCallback(() => {
    const next: Theme = read() === "dark" ? "light" : "dark";
    const root = document.documentElement;
    root.classList.add("theme-anim");
    root.dataset.theme = next;
    try { localStorage.setItem("tvd_theme", next); } catch { /* storage blocked */ }
    setTheme(next);
    window.setTimeout(() => root.classList.remove("theme-anim"), 450);
  }, []);

  return { theme, toggle };
}
