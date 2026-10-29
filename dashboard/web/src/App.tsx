import { AnimatePresence, MotionConfig } from "framer-motion";
import { Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { LiveProvider } from "@/hooks/useLiveSocket";
import Analytics from "@/pages/Analytics";
import Events from "@/pages/Events";
import LiveMonitor from "@/pages/LiveMonitor";
import Overview from "@/pages/Overview";
import Setup from "@/pages/Setup";

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait" initial={false}>
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Overview />} />
        <Route path="/events" element={<Events />} />
        <Route path="/live" element={<LiveMonitor />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    // reducedMotion="user": Framer Motion skips transform/layout animations when the OS asks for less motion.
    <MotionConfig reducedMotion="user">
      <LiveProvider>
        <Layout>
          <AnimatedRoutes />
        </Layout>
      </LiveProvider>
    </MotionConfig>
  );
}
