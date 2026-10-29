import * as React from "react";
import { cn } from "@/lib/utils";

const field =
  "h-10 w-full rounded-md border border-border bg-card px-3 text-sm placeholder:text-muted-foreground " +
  "transition-shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => <input ref={ref} className={cn(field, className)} {...props} />,
);
Input.displayName = "Input";

export const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => <select ref={ref} className={cn(field, "pr-8", className)} {...props} />,
);
Select.displayName = "Select";
