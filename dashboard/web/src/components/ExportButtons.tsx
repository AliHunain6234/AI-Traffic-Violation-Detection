import { FileDown, FileText } from "lucide-react";
import { exportUrl } from "@/lib/api";
import type { Filters } from "@/lib/types";
import { Button } from "./ui/button";

/** Download the currently filtered events as CSV, or a violation report as PDF. */
export function ExportButtons({ filters }: { filters: Filters }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Button asChild variant="outline" size="sm">
        <a href={exportUrl.csv(filters)} download="events.csv"><FileDown className="h-4 w-4" />CSV</a>
      </Button>
      <Button asChild variant="outline" size="sm">
        <a href={exportUrl.pdf(filters)} download="violation_report.pdf"><FileText className="h-4 w-4" />PDF report</a>
      </Button>
    </div>
  );
}
