"use client";

import { useTheme } from "next-themes";
import { Moon, Sun, PhoneCall, RefreshCw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/hooks/use-health";
import { useIsMounted } from "@/hooks/use-is-mounted";

export function AppHeader() {
  const { resolvedTheme, setTheme } = useTheme();
  const { data: connected, isFetching, refetch } = useHealth();
  // next-themes only knows the real theme after mount (it reads from
  // localStorage/system on the client) - render a stable icon until then
  // to avoid a server/client hydration mismatch.
  const mounted = useIsMounted();

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

      <div className="flex items-center gap-2">
        <Badge variant={connected ? "success" : "warning"}>
          {connected ? "API ansluten" : "API ej tillgänglig"}
        </Badge>
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
