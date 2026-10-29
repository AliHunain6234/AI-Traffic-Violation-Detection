import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Gauge, KeyRound, ListChecks, Moon, PenTool, Sun, Video, WifiOff } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useLive } from "@/hooks/useLiveSocket";
import { useTheme } from "@/hooks/useTheme";
import { getToken, setToken } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

const NAV = [
  { to: "/", label: "Overview", icon: Gauge, end: true },
  { to: "/events", label: "Events", icon: ListChecks, end: false },
  { to: "/live", label: "Live monitor", icon: Video, end: false },
  { to: "/setup", label: "Setup", icon: PenTool, end: false },
  { to: "/analytics", label: "Analytics", icon: BarChart3, end: false },
];

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <Button variant="ghost" size="icon" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={theme} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.18 }}>
          {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </motion.span>
      </AnimatePresence>
    </Button>
  );
}

/** Appears when the API answers 401: lets the user paste the access token. */
function TokenPrompt() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState(getToken());
  useEffect(() => {
    const show = () => setOpen(true);
    window.addEventListener("tvd-unauthorized", show);
    return () => window.removeEventListener("tvd-unauthorized", show);
  }, []);
  return (
    <>
      <Button variant="ghost" size="icon" aria-label="Access token" onClick={() => setOpen(true)}><KeyRound className="h-5 w-5" /></Button>
      <AnimatePresence>
        {open && (
          <motion.div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <motion.form
              initial={{ y: 20, scale: 0.97 }} animate={{ y: 0, scale: 1 }} exit={{ y: 10, opacity: 0 }}
              onSubmit={(e) => { e.preventDefault(); setToken(value.trim()); window.location.reload(); }}
              className="w-full max-w-sm space-y-4 rounded-lg border border-border bg-card p-6 shadow-2xl"
            >
              <h2 className="text-lg font-semibold">Access token</h2>
              <p className="text-sm text-muted-foreground">This dashboard is protected. Enter the token configured as <code>dashboard.api_token</code>.</p>
              <Input type="password" autoFocus value={value} onChange={(e) => setValue(e.target.value)} placeholder="Token" aria-label="Access token" />
              <div className="flex justify-end gap-2">
                <Button type="button" variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
                <Button type="submit">Save</Button>
              </div>
            </motion.form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { connected } = useLive();
  const location = useLocation();
  return (
    <div className="min-h-screen lg:pl-[17rem]">
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <aside className="glass fixed inset-y-3 left-3 z-30 hidden w-[15.5rem] flex-col rounded-2xl lg:flex">
        <div className="flex items-center gap-3 px-5 py-5">
          <motion.div
            className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent text-primary-foreground shadow-lg"
            whileHover={{ rotate: 12, scale: 1.08 }}
            transition={{ type: "spring", stiffness: 300, damping: 14 }}
          >
            <span className="absolute -inset-1 -z-10 rounded-xl bg-gradient-to-br from-primary to-accent opacity-50 blur-md" />
            <Gauge className="h-5 w-5" />
          </motion.div>
          <div><p className="text-sm font-bold leading-tight">Traffic Control</p><p className="text-xs text-muted-foreground">Violation monitoring</p></div>
        </div>
        <nav className="flex-1 space-y-1 px-3" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => cn("group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors", isActive ? "text-foreground" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground")}>
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span layoutId="nav-active" className="absolute inset-0 rounded-lg border border-primary/30 bg-gradient-to-r from-primary/25 to-accent/10" transition={{ type: "spring", stiffness: 400, damping: 34 }}>
                      <span className="absolute bottom-2 left-0 top-2 w-[3px] rounded-full bg-gradient-to-b from-primary to-accent" />
                    </motion.span>
                  )}
                  <Icon className={cn("relative h-4 w-4 transition-transform duration-200 group-hover:scale-125", isActive && "text-primary")} aria-hidden /><span className="relative">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <p className="px-5 pb-5 text-[11px] leading-snug text-muted-foreground">Automated detections for human review. Not a legal determination.</p>
      </aside>

      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border/50 bg-background/90 px-4 py-3 lg:px-8">
        <div className="flex items-center gap-2 lg:hidden"><Gauge className="h-5 w-5 text-primary" /><span className="text-sm font-bold">Traffic Control</span></div>
        <div className={cn("hidden items-center gap-2.5 rounded-full border px-3 py-1.5 text-xs font-medium lg:flex", connected ? "border-legal/40 bg-legal/10 text-legal" : "border-border bg-muted/50 text-muted-foreground")}>
          {connected ? <span className="live-dot" /> : <WifiOff className="h-3.5 w-3.5" />}{connected ? "Connected to live updates" : "Offline, retrying"}
        </div>
        <div className="flex items-center gap-1"><TokenPrompt /><ThemeToggle /></div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pb-24 pt-6 lg:px-8 lg:pb-10" key={location.pathname}>{children}</main>

      <nav className="glass fixed inset-x-3 bottom-3 z-30 grid grid-cols-5 rounded-2xl lg:hidden" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => cn("flex flex-col items-center gap-1 py-2 text-[11px] font-medium", isActive ? "text-primary" : "text-muted-foreground")}>
            <Icon className="h-5 w-5" aria-hidden />{label.split(" ")[0]}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
