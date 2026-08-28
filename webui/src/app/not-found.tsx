import Link from "next/link";

import { Button } from "@/components/ui/button";

/** Swedish fallback for stale, deleted, or unknown call deep links. */
export default function NotFound() {
  return (
    <section className="mx-auto flex max-w-lg flex-col items-start gap-4 py-16">
      <p className="text-sm font-medium text-muted-foreground">404</p>
      <h1 className="text-2xl font-semibold tracking-tight">Sidan kunde inte hittas</h1>
      <p className="text-sm text-muted-foreground">
        Samtalet kan ha tagits bort eller inte längre vara tillgängligt från backend.
      </p>
      <Button asChild>
        <Link href="/">Till översikten</Link>
      </Button>
    </section>
  );
}
