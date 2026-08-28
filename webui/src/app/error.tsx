"use client";

import { useEffect } from "react";

import { Button } from "@/components/ui/button";

/** Recoverable route error surface without exposing backend details. */
export default function GlobalError({
  error,
  reset,
}: Readonly<{
  error: Error & { digest?: string };
  reset: () => void;
}>) {
  useEffect(() => {
    console.error("Webui route error", error);
  }, [error]);

  return (
    <section className="mx-auto flex max-w-lg flex-col items-start gap-4 py-16">
      <h1 className="text-2xl font-semibold tracking-tight">Något gick fel</h1>
      <p className="text-sm text-muted-foreground">
        Försök igen. Om felet kvarstår, kontrollera backendens status och loggar.
      </p>
      <Button onClick={reset}>Försök igen</Button>
    </section>
  );
}
