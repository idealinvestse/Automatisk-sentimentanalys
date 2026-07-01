"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { CategorySentimentChart } from "@/components/charts/category-sentiment-chart";
import { VolumeByCategoryChart } from "@/components/charts/volume-by-category-chart";
import { MOCK_CALLS, summarizeCategories } from "@/lib/mock-data";

const LEGEND_COLORS = [
  "bg-primary",
  "bg-success",
  "bg-warning",
  "bg-destructive",
  "bg-secondary-foreground",
  "bg-muted-foreground",
];

export default function AnalyticsPage() {
  const categories = summarizeCategories(MOCK_CALLS);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Analys & Trender</h1>
        <p className="text-sm text-muted-foreground">
          Sentiment och volym per samtalskategori (demo-data).
        </p>
      </div>

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
    </div>
  );
}
