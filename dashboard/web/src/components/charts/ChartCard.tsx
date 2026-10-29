import type { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Skeleton } from "../ui/skeleton";

interface Props {
  title: string;
  description?: string;
  loading?: boolean;
  empty?: boolean;
  emptyText?: string;
  height?: number;
  action?: ReactNode;
  children: ReactNode;
}

/** Shared chart frame: title, skeleton while loading and an empty state (never a blank box). */
export function ChartCard({ title, description, loading, empty, emptyText = "No data yet", height = 260, action, children }: Props) {
  return (
    <Card className="transition-shadow hover:shadow-lift">
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </div>
        {action}
      </CardHeader>
      <CardContent>
        <div style={{ height }} className="w-full">
          {loading ? <Skeleton className="h-full w-full" /> : empty ? (
            <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">{emptyText}</div>
          ) : children}
        </div>
      </CardContent>
    </Card>
  );
}
