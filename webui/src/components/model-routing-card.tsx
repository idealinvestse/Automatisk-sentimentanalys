"use client";

import { Zap, Scale, Brain, Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { TIER_LIST, type RoutingTier, type TierInfo } from "@/lib/routing-tier";

const TIER_ICON: Record<RoutingTier, typeof Zap> = {
  fast: Zap,
  balanced: Scale,
  deep: Brain,
};

const TIER_TONE: Record<RoutingTier, string> = {
  fast: "border-emerald-500/50 bg-emerald-500/5",
  balanced: "border-blue-500/50 bg-blue-500/5",
  deep: "border-purple-500/50 bg-purple-500/5",
};

export interface ModelRoutingCardProps {
  tier: RoutingTier;
  onTierChange: (tier: RoutingTier) => void;
  /** If provided, show the effective tier after segment-count overrides. */
  effectiveTier?: RoutingTier;
  disabled?: boolean;
}

/**
 * Card with three tier buttons (FAST / BALANCED / DEEP) for choosing
 * cost/quality model routing. Mirrors src/llm/routing.py tiers.
 */
export function ModelRoutingCard({
  tier,
  onTierChange,
  effectiveTier,
  disabled,
}: ModelRoutingCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Model routing</CardTitle>
        <CardDescription>
          Välj kostnad/kvalitet-tier för LLM-analys. Mappar till <code>llm_model</code> i
          pipeline-requesten (speglar <code>src/llm/routing.py</code>).
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {TIER_LIST.map((info) => (
            <TierButton
              key={info.id}
              info={info}
              selected={tier === info.id}
              effective={effectiveTier === info.id && effectiveTier !== tier}
              disabled={disabled}
              onClick={() => onTierChange(info.id)}
            />
          ))}
        </div>
        {effectiveTier && effectiveTier !== tier ? (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Obs:</span> Tier justerades automatiskt till{" "}
            <code className="font-mono">{effectiveTier.toUpperCase()}</code> baserat på
            segmentantal/djupanalys-flaggan (samma logik som <code>select_model()</code>).
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

function TierButton({
  info,
  selected,
  effective,
  disabled,
  onClick,
}: {
  info: TierInfo;
  selected: boolean;
  effective: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  const Icon = TIER_ICON[info.id];
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3 text-left transition-colors",
        "hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        "disabled:cursor-not-allowed disabled:opacity-50",
        selected ? TIER_TONE[info.id] : "border-border",
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-muted-foreground" />
          <span className="text-sm font-semibold">{info.label}</span>
        </div>
        {selected ? (
          <Check className="size-4 text-primary" />
        ) : effective ? (
          <Badge variant="secondary" className="text-[10px]">auto</Badge>
        ) : null}
      </div>
      <p className="text-xs text-muted-foreground">{info.description}</p>
      <div className="flex items-center gap-3 text-xs">
        <span className="tabular-nums text-muted-foreground">
          ${info.costPerMTokens.toFixed(2)}/M tok
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="capitalize text-muted-foreground">{info.latency} latency</span>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">{"★".repeat(info.quality)}{"☆".repeat(5 - info.quality)}</span>
      </div>
      <code className="truncate text-[10px] font-mono text-muted-foreground">{info.model}</code>
    </button>
  );
}
