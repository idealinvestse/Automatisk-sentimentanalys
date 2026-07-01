import { Badge } from "@/components/ui/badge";
import type { RiskLevel, SentimentLabel } from "@/lib/mock-data";

const SENTIMENT_LABEL: Record<SentimentLabel, string> = {
  positive: "Positiv",
  neutral: "Neutral",
  negative: "Negativ",
};

const SENTIMENT_VARIANT: Record<SentimentLabel, "success" | "secondary" | "destructive"> = {
  positive: "success",
  neutral: "secondary",
  negative: "destructive",
};

export function SentimentBadge({ value }: { value: SentimentLabel }) {
  return <Badge variant={SENTIMENT_VARIANT[value]}>{SENTIMENT_LABEL[value]}</Badge>;
}

const RISK_LABEL: Record<RiskLevel, string> = {
  low: "Låg",
  medium: "Medel",
  high: "Hög",
  critical: "Kritisk",
};

const RISK_VARIANT: Record<RiskLevel, "secondary" | "warning" | "destructive"> = {
  low: "secondary",
  medium: "warning",
  high: "destructive",
  critical: "destructive",
};

export function RiskBadge({ value }: { value: RiskLevel }) {
  return <Badge variant={RISK_VARIANT[value]}>{RISK_LABEL[value]}</Badge>;
}

export function QaBadge({ passed, score }: { passed: boolean | null; score: number | null }) {
  if (passed === null || score === null) {
    return <Badge variant="outline">—</Badge>;
  }
  return (
    <Badge variant={passed ? "success" : "destructive"}>
      {score} {passed ? "· Godkänd" : "· Underkänd"}
    </Badge>
  );
}
