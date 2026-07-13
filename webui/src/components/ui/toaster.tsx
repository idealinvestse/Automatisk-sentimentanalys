"use client";

import { Toaster as SonnerToaster } from "sonner";
import { useTheme } from "next-themes";

/**
 * App-wide toast renderer. App-wide toast renderer (success/warning/error)
 * via `toast.success` / `toast.warning` / `toast.error` from `sonner`.
 */
export function Toaster() {
  const { resolvedTheme } = useTheme();
  return (
    <SonnerToaster
      theme={(resolvedTheme as "light" | "dark" | "system") ?? "system"}
      position="top-right"
      richColors
      closeButton
      toastOptions={{
        classNames: {
          toast: "border-border",
        },
      }}
    />
  );
}

export { toast } from "sonner";
