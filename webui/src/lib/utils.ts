import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind class lists, resolving conflicts (shadcn-style helper). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
