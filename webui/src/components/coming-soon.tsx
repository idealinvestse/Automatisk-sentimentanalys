import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ComingSoon({ title, legacyTab }: { title: string; legacyTab: string }) {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      <Card>
        <CardHeader>
          <CardTitle>Migreras enligt plan</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Denna vy motsvarar fliken &ldquo;{legacyTab}&rdquo; i den befintliga NiceGUI-dashboarden
          och migreras i en senare fas, se{" "}
          <code className="text-xs">docs/WEBUI_MODERNIZATION_PLAN.md</code>. Fram tills den är
          klar, använd NiceGUI-dashboarden för denna funktionalitet.
        </CardContent>
      </Card>
    </div>
  );
}
