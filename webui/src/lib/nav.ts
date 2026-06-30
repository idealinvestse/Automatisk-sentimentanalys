export interface NavItem {
  href: string;
  label: string;
  /** Lucide icon name, kept as string to avoid importing every icon eagerly. */
  icon: string;
  /** Mirrors the legacy NiceGUI tab this route replaces (for migration tracking). */
  legacyTab?: string;
}

export const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Översikt", icon: "LayoutDashboard", legacyTab: "Översikt" },
  { href: "/analytics", label: "Analys & Trender", icon: "LineChart", legacyTab: "Analys & Trender" },
  { href: "/agents", label: "Agentprestanda", icon: "Users", legacyTab: "Agentprestanda" },
  { href: "/insights", label: "Fas 4 Insikter", icon: "Sparkles", legacyTab: "Fas 4 Insikter" },
  { href: "/transcription", label: "Transkribering", icon: "AudioLines", legacyTab: "Transkribering" },
  { href: "/testlab", label: "Testlabb", icon: "FlaskConical", legacyTab: "Testlabb" },
];
