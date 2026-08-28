"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  LineChart,
  Users,
  Sparkles,
  Briefcase,
  AudioLines,
  FlaskConical,
  Cpu,
  Microscope,
  type LucideIcon,
} from "lucide-react";

import { NAV_ITEMS } from "@/lib/nav";
import { cn } from "@/lib/utils";

const ICONS: Record<string, LucideIcon> = {
  LayoutDashboard,
  LineChart,
  Users,
  Sparkles,
  Briefcase,
  AudioLines,
  FlaskConical,
  Cpu,
  Microscope,
};

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Huvudnavigering"
      className="hidden w-60 shrink-0 flex-col gap-1 border-r border-border p-3 md:flex"
    >
      {NAV_ITEMS.map((item) => {
        const Icon = ICONS[item.icon];
        const active = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              active
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
            )}
          >
            {Icon ? <Icon className="size-4 shrink-0" /> : null}
            <span className="truncate">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

/** Horizontally scrollable navigation used below the header on small screens. */
export function MobileNavigation() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Mobilnavigering"
      className="flex shrink-0 gap-1 overflow-x-auto border-b border-border px-2 py-2 md:hidden"
    >
      {NAV_ITEMS.map((item) => {
        const Icon = ICONS[item.icon];
        const active = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
              active
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
            )}
          >
            {Icon ? <Icon className="size-3.5 shrink-0" /> : null}
            <span className="truncate">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
