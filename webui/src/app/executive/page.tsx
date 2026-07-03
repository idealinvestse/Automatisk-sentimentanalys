"use client";

import {
  Briefcase,
  WifiOff,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  ShieldAlert,
  Gauge,
  DollarSign,
  Award,
  Activity,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { useExecutiveSummary } from "@/hooks/use-executive-summary";

const RISK_TONE: Record<string, "destructive" | "default" | "secondary" | "success"> = {
  critical: "destructive",
  high: "destructive",
  medium: "secondary",
  low: "success",
};

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function KpiCard({
  icon: Icon,
  label,
  value,
  hint,
  tone = "default",
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "success" | "destructive";
}) {
  const toneClass =
    tone === "success"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "destructive"
        ? "text-red-600 dark:text-red-400"
        : "text-foreground";
  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">{label}</span>
          <Icon className="size-4 text-muted-foreground" />
        </div>
        <span className={`text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</span>
        {hint ? <span className="text-xs text-muted-foreground">{hint}</span> : null}
      </CardContent>
    </Card>
  );
}

export default function ExecutivePage() {
  const { summary, isLoading, isError } = useExecutiveSummary();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Executive Insights</h1>
        <p className="text-sm text-muted-foreground">
          Aggregerade KPI:er över alla demo-samtal — risker, QA, agentprestanda och LLM-kostnad.
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      ) : isError ? (
        <EmptyState
          icon={WifiOff}
          title="Kunde inte hämta executive data"
          hint="Kontrollera att backend-API:et körs och är nåbart från webui."
        />
      ) : !summary ? (
        <EmptyState
          icon={Briefcase}
          title="Ingen data tillgänglig"
          hint="Kör pipeline-analysen först för att generera underlag för executive insights."
        />
      ) : (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            <KpiCard
              icon={Activity}
              label="Totalt samtal"
              value={String(summary.kpis.totalCalls)}
              hint="analysade i pipeline"
            />
            <KpiCard
              icon={Gauge}
              label="Snitt QA"
              value={summary.kpis.avgQaScore.toFixed(1)}
              hint={`${pct(summary.kpis.qaPassRate)} godkända`}
              tone={summary.kpis.avgQaScore >= 70 ? "success" : "destructive"}
            />
            <KpiCard
              icon={TrendingUp}
              label="Snitt sentiment"
              value={summary.kpis.avgSentiment.toFixed(2)}
              hint={summary.kpis.avgSentiment >= 0 ? "positiv lutning" : "negativ lutning"}
              tone={summary.kpis.avgSentiment >= 0 ? "success" : "destructive"}
            />
            <KpiCard
              icon={AlertTriangle}
              label="Totala larm"
              value={String(summary.kpis.totalAlerts)}
              hint={`${summary.kpis.criticalCalls} kritiska samtal`}
              tone={summary.kpis.totalAlerts > 0 ? "destructive" : "success"}
            />
            <KpiCard
              icon={DollarSign}
              label="LLM-kostnad"
              value={`$${summary.kpis.totalLlmCostUsd.toFixed(4)}`}
              hint="uppskattad från llm_judge"
            />
          </div>

          {/* Risk overview */}
          <div className="grid gap-4 lg:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <ShieldAlert className="size-4" />
                  Risköversikt
                </CardTitle>
                <CardDescription>Aggregerade riskmetrik från pipeline-rapporter.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <RiskBar label="Churn-risk" value={summary.kpis.avgChurnRisk} />
                <RiskBar label="Eskaleringsrisk" value={summary.kpis.avgEscalationRisk} />
                <RiskBar
                  label="Tillfredsställelse"
                  value={summary.kpis.avgSatisfaction}
                  positive
                />
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(summary.riskDistribution).map(([level, count]) => (
                    <Badge key={level} variant={RISK_TONE[level] ?? "default"} className="capitalize">
                      {level}: {count}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top alert rules */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <AlertTriangle className="size-4" />
                  Topp larmregler
                </CardTitle>
                <CardDescription>Mest utlösande regler denna period.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {summary.topAlertRules.length === 0 ? (
                  <span className="text-sm text-muted-foreground">Inga larm utlösta.</span>
                ) : (
                  summary.topAlertRules.map((rule) => (
                    <div
                      key={rule.ruleId}
                      className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                    >
                      <code className="text-xs">{rule.ruleId}</code>
                      <Badge variant="secondary">{rule.count}×</Badge>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            {/* Category breakdown */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Activity className="size-4" />
                  Kategorifördelning
                </CardTitle>
                <CardDescription>Samtal och sentiment per kategori.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {summary.categoryBreakdown.map((cat) => (
                  <div
                    key={cat.category}
                    className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                  >
                    <div className="flex flex-col">
                      <span className="text-sm font-medium capitalize">{cat.category}</span>
                      <span className="text-xs text-muted-foreground">{cat.calls} samtal</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs tabular-nums text-muted-foreground">
                        QA {cat.avgQa.toFixed(0)}
                      </span>
                      <Badge
                        variant={cat.avgSentiment >= 0 ? "success" : "destructive"}
                        className="tabular-nums"
                      >
                        {cat.avgSentiment >= 0 ? "+" : ""}
                        {cat.avgSentiment.toFixed(2)}
                      </Badge>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Agent benchmark table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Award className="size-4" />
                Agentbenchmark
              </CardTitle>
              <CardDescription>
                Rankad efter snitt-QA. Empathy från <code>agent_performance</code> om tillgängligt.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs text-muted-foreground">
                      <th className="py-2 pr-4 font-medium">Agent</th>
                      <th className="py-2 pr-4 font-medium">Samtal</th>
                      <th className="py-2 pr-4 font-medium">QA</th>
                      <th className="py-2 pr-4 font-medium">Sentiment</th>
                      <th className="py-2 pr-4 font-medium">Empathy</th>
                      <th className="py-2 pr-4 font-medium">Larm</th>
                      <th className="py-2 pr-4 font-medium">Churn-risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.agentBenchmarks.map((agent, idx) => (
                      <tr key={agent.agent} className="border-b border-border/50 last:border-0">
                        <td className="py-2 pr-4">
                          <div className="flex items-center gap-2">
                            {idx === 0 ? (
                              <Award className="size-3.5 text-amber-500" />
                            ) : null}
                            <span className="font-medium">{agent.agent}</span>
                          </div>
                        </td>
                        <td className="py-2 pr-4 tabular-nums">{agent.calls}</td>
                        <td className="py-2 pr-4 tabular-nums">{agent.avgQaScore.toFixed(1)}</td>
                        <td className="py-2 pr-4 tabular-nums">
                          <SentimentBadge value={agent.avgSentiment} />
                        </td>
                        <td className="py-2 pr-4 tabular-nums">
                          {agent.avgEmpathy !== null ? agent.avgEmpathy.toFixed(2) : "—"}
                        </td>
                        <td className="py-2 pr-4 tabular-nums">{agent.alertCount}</td>
                        <td className="py-2 pr-4 tabular-nums">
                          <RiskBadge value={agent.avgChurnRisk} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function RiskBar({ label, value, positive = false }: { label: string; value: number; positive?: boolean }) {
  const pctVal = Math.round(value * 100);
  const tone = positive
    ? value >= 0.7
      ? "bg-emerald-500"
      : value >= 0.4
        ? "bg-amber-500"
        : "bg-red-500"
    : value >= 0.6
      ? "bg-red-500"
      : value >= 0.3
        ? "bg-amber-500"
        : "bg-emerald-500";
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums font-medium">{pctVal}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${pctVal}%` }} />
      </div>
    </div>
  );
}

function SentimentBadge({ value }: { value: number }) {
  const Icon = value >= 0.05 ? TrendingUp : value <= -0.05 ? TrendingDown : Minus;
  const tone = value >= 0.05 ? "success" : value <= -0.05 ? "destructive" : "secondary";
  return (
    <Badge variant={tone} className="gap-1 tabular-nums">
      <Icon className="size-3" />
      {value >= 0 ? "+" : ""}
      {value.toFixed(2)}
    </Badge>
  );
}

function RiskBadge({ value }: { value: number }) {
  const tone = value >= 0.6 ? "destructive" : value >= 0.3 ? "secondary" : "success";
  return <Badge variant={tone} className="tabular-nums">{Math.round(value * 100)}%</Badge>;
}
