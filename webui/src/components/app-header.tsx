"use client";

import { useTheme } from "next-themes";
import { Moon, Sun, PhoneCall, RefreshCw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { isApiConnected, useHealth } from "@/hooks/use-health";
import { useIsMounted } from "@/hooks/use-is-mounted";

function healthBadge(status: ReturnType<typeof useHealth>["data"]) {
  if (!status || status.status === "offline") {
    return { variant: "warning" as const, label: "API ej tillgänglig" };
  }
  if (status.status === "unauthorized") {
    return { variant: "destructive" as const, label: "API 401 – fel nyckel" };
  }
  if (status.status === "auth_required") {
    return { variant: "destructive" as const, label: "API kräver nyckel" };
  }
  return { variant: "success" as const, label: "API ansluten" };
}

export function AppHeader() {
  const { resolvedTheme, setTheme } = useTheme();
  const { data: status, isFetching, refetch } = useHealth();
  const mounted = useIsMounted();
  const badge = healthBadge(status);
  const connected = isApiConnected(status);

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card/60 px-4 backdrop-blur supports-[backdrop-filter]:bg-card/40">
      <div className="flex items-center gap-2">
        <PhoneCall className="size-5 text-primary" />
        <span className="text-sm font-semibold sm:text-base">
          Svensk Call Center – Samtalsintelligens
        </span>
        <Badge variant="outline" className="ml-1 hidden sm:inline-flex">
          Produktion
        </Badge>
      </div>

      <div className="flex min-w-0 items-center gap-2">
        {!connected && status?.detail ? (
          <span
            className="hidden max-w-[14rem] truncate text-xs text-muted-foreground lg:inline"
            title={status.detail}
          >
            {status.detail}
          </span>
        ) : null}
        <Badge variant={badge.variant}>{badge.label}</Badge>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => refetch()}
          aria-label="Ladda om anslutningsstatus"
          className={isFetching ? "animate-spin" : undefined}
        >
          <RefreshCw className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Växla ljust/mörkt tema"
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
        >
          {mounted && resolvedTheme === "dark" ? (
            <Sun className="size-4" />
          ) : (
            <Moon className="size-4" />
          )}
        </Button>
      </div>
    </header>
  );
}
