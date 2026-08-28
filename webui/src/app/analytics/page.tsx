"use client";

import { WifiOff } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/empty-state";
import { CategorySentimentChart } from "@/components/charts/category-sentiment-chart";
import { VolumeByCategoryChart } from "@/components/charts/volume-by-category-chart";
import { useDemoReports } from "@/hooks/use-demo-reports";
import { summarizeCategories } from "@/lib/mock-data";

const LEGEND_COLORS = [
  "bg-primary",
  "bg-success",
  "bg-warning",
  "bg-destructive",
  "bg-secondary-foreground",
  "bg-muted-foreground",
];

export default function AnalyticsPage() {
  const { calls, isLoading, isError, usingLiveData } = useDemoReports();
  const categories = summarizeCategories(calls);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Analys & Trender</h1>
        <p className="text-sm text-muted-foreground">
          Sentiment och volym per samtalskategori, beräknat av den riktiga pipelinen från{" "}
          {usingLiveData ? "sparade samtal." : "demosamtalen."}
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Skeleton className="h-72 w-full" />
          <Skeleton className="h-72 w-full" />
        </div>
      ) : isError ? (
        <EmptyState
          icon={WifiOff}
          title="Kunde inte hämta analysdata"
          hint="Kontrollera att backend-API:et körs och är nåbart från webui."
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Snitt-sentiment per kategori</CardTitle>
              <CardDescription>Andel positivt sentiment, 0–100%.</CardDescription>
            </CardHeader>
            <CardContent>
              <CategorySentimentChart data={categories} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Samtalsvolym per kategori</CardTitle>
              <CardDescription>Antal samtal fördelat på kategori.</CardDescription>
            </CardHeader>
            <CardContent>
              <VolumeByCategoryChart data={categories} />
              <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
                {categories.map((c, i) => (
                  <li key={c.category} className="flex items-center gap-1.5">
                    <span
                      className={`size-2.5 rounded-full ${LEGEND_COLORS[i % LEGEND_COLORS.length]}`}
                    />
                    <span className="capitalize">{c.category}</span>
                    <span className="font-medium text-foreground">{c.calls}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
