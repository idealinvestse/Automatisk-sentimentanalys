import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ComingSoon({ title }: { title: string; legacyTab?: string }) {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      <Card>
        <CardHeader>
          <CardTitle>Kommer snart</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Denna vy är ännu inte färdig. Se{" "}
          <code className="text-xs">docs/ROADMAP.md</code> för aktuella prioriteringar.
        </CardContent>
      </Card>
    </div>
  );
}
