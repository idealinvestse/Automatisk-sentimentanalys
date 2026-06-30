"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  LineChart,
  Users,
  Sparkles,
  AudioLines,
  FlaskConical,
  type LucideIcon,
} from "lucide-react";

import { NAV_ITEMS } from "@/lib/nav";
import { cn } from "@/lib/utils";

const ICONS: Record<string, LucideIcon> = {
  LayoutDashboard,
  LineChart,
  Users,
  Sparkles,
  AudioLines,
  FlaskConical,
};

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <nav className="hidden w-60 shrink-0 flex-col gap-1 border-r border-border p-3 md:flex">
      {NAV_ITEMS.map((item) => {
        const Icon = ICONS[item.icon];
        const active = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
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
