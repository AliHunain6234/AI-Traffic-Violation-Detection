import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold", {
  variants: {
    variant: {
      default: "border-border bg-muted text-foreground",
      violation: "border-violation/40 bg-violation/15 text-violation",
      caution: "border-caution/40 bg-caution/15 text-caution",
      legal: "border-legal/40 bg-legal/15 text-legal",
      primary: "border-primary/40 bg-primary/15 text-primary",
      speed: "border-fuchsia-500/40 bg-fuchsia-500/15 text-fuchsia-400",
    },
  },
  defaultVariants: { variant: "default" },
});

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
