export interface NavItem {
  href: string;
  label: string;
  /** Lucide icon name, kept as string to avoid importing every icon eagerly. */
  icon: string;
}

export const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Översikt", icon: "LayoutDashboard" },
  { href: "/analytics", label: "Analys & Trender", icon: "LineChart" },
  { href: "/analysis", label: "Analysdetaljer", icon: "Microscope" },
  { href: "/agents", label: "Agentprestanda", icon: "Users" },
  { href: "/insights", label: "Fas 4 Insikter", icon: "Sparkles" },
  { href: "/executive", label: "Executive Insights", icon: "Briefcase" },
  { href: "/transcription", label: "Transkribering", icon: "AudioLines" },
  { href: "/testlab", label: "Testlabb", icon: "FlaskConical" },
  { href: "/edge", label: "Edge AI", icon: "Cpu" },
];
