"use client";

import { Check, Wallet, Scale, Brain, Heart, Route, Search, Shield, List, Flag, Gauge, Sparkles, GraduationCap } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  type AnalysisProfileMenuItem,
  costQualityLabel,
  formatUsd,
} from "@/lib/analysis-profiles";

const ICONS: Record<string, typeof Scale> = {
  wallet: Wallet,
  gauge: Gauge,
  heart: Heart,
  route: Route,
  search: Search,
  graduation: GraduationCap,
  shield: Shield,
  list: List,
  flag: Flag,
  brain: Brain,
  scale: Scale,
  sparkles: Sparkles,
};

export interface AnalysisPerspectivePickerProps {
  items: AnalysisProfileMenuItem[];
  value: string | null;
  onChange: (item: AnalysisProfileMenuItem) => void;
  disabled?: boolean;
  loading?: boolean;
}

/**
 * Simple selectable grid of analysis perspectives with recommended paid model + cost.
 */
export function AnalysisPerspectivePicker({
  items,
  value,
  onChange,
  disabled,
  loading,
}: AnalysisPerspectivePickerProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Analysperspektiv (paid modeller)</CardTitle>
        <CardDescription>
          Välj perspektiv — systemet föreslår lämplig <strong>betald</strong> modell med{" "}
          <strong>kostnad som optimeringsfaktor</strong>. Skickas som{" "}
          <code>analysis_perspective</code> + <code>llm_model</code> till pipelinen.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <p className="text-sm text-muted-foreground">Hämtar rekommendationer från katalog…</p>
        ) : items.length === 0 ? (
          <p className="text-sm text-muted-foreground">Inga profiler tillgängliga.</p>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((item) => {
              const Icon = ICONS[item.icon || "scale"] || Scale;
              const selected = value === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  disabled={disabled}
                  aria-pressed={selected}
                  onClick={() => onChange(item)}
                  className={cn(
                    "flex flex-col gap-1.5 rounded-lg border p-3 text-left transition-colors",
                    "hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    "disabled:cursor-not-allowed disabled:opacity-50",
                    selected ? "border-primary bg-primary/5" : "border-border",
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Icon className="size-4 shrink-0 text-muted-foreground" />
                      <span className="text-sm font-semibold leading-tight">{item.label}</span>
                    </div>
                    {selected ? <Check className="size-4 shrink-0 text-primary" /> : null}
                  </div>
                  <p className="line-clamp-2 text-xs text-muted-foreground">{item.description}</p>
                  {item.use_when ? (
                    <p className="text-[11px] text-muted-foreground">
                      <span className="font-medium">När:</span> {item.use_when}
                    </p>
                  ) : null}
                  <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
                    <Badge variant="secondary" className="text-[10px]">
                      {costQualityLabel(item.cost_priority, item.quality_priority)}
                    </Badge>
                    <Badge variant="outline" className="text-[10px] tabular-nums">
                      {formatUsd(item.blended_usd_per_m, 2)}/M
                    </Badge>
                    <Badge variant="outline" className="text-[10px] tabular-nums">
                      ~{formatUsd(item.est_cost_per_call_usd, 4)}/samtal
                    </Badge>
                  </div>
                  <code className="truncate text-[10px] font-mono text-muted-foreground">
                    {(item.provider || "?") + " · " + (item.model || "ingen modell")}
                  </code>
                </button>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
