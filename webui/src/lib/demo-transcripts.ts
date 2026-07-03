/**
 * Canned Swedish call-center demo transcripts, ported 1:1 from the reference
 * dataset used by the legacy dashboard
 * (`app/services/data_services.py::DEMO_TRANSCRIPTS`).
 *
 * These are sent as-is to the real backend (`POST /analyze_pipeline` and the
 * Fas 4 aggregate endpoints), so the numbers shown in the UI (sentiment, QA
 * score, risk, hot topics, agent metrics) come from the real
 * `CallAnalysisPipeline`, not hardcoded mock values. Only the *conversations*
 * are synthetic/demo data — see docs/WEBUI_MODERNIZATION_PLAN.md §6
 * ("Datakälla"-frågan) for the background on this decision.
 */

export interface DemoSegment {
  start: number;
  end: number;
  text: string;
  speaker: "Agent" | "Kund";
}

export interface DemoTranscriptMeta {
  agent: string;
  duration_s: number;
  category: string;
}

export interface DemoTranscript {
  id: string;
  title: string;
  meta: DemoTranscriptMeta;
  segments: DemoSegment[];
}

export const DEMO_TRANSCRIPTS: DemoTranscript[] = [
  {
    id: "CALL-001",
    title: "Faktura fel – lyckad upplösning",
    meta: { agent: "Agent-Anna", duration_s: 420, category: "billing" },
    segments: [
      { start: 0, end: 8, speaker: "Agent", text: "Hej, jag heter Anna på kundtjänst, hur kan jag hjälpa dig idag?" },
      {
        start: 8,
        end: 18,
        speaker: "Kund",
        text: "Hej Anna, jag har fått en faktura på 890 kr som jag inte förstår. Det står att jag har ringt internationellt men det har jag inte.",
      },
      {
        start: 18,
        end: 32,
        speaker: "Agent",
        text: "Tack för att du ringer in. Jag förstår att det känns frustrerande. Kan jag få ditt kundnummer eller personnummer så kollar jag upp det direkt?",
      },
      { start: 32, end: 42, speaker: "Kund", text: "Ja, det är 19851203-1234. Och jag har aldrig ringt utomlands, jag lovar." },
      {
        start: 42,
        end: 65,
        speaker: "Agent",
        text: "Tack, jag ser nu i systemet att det finns en debitering från ett samtal till +49 den 12 maj. Men jag ser också att du har ett abonnemang som inkluderar EU-samtal. Det verkar som en felkodning i faktureringssystemet. Jag krediterar 890 kr nu direkt och skickar en rättad faktura.",
      },
      { start: 65, end: 75, speaker: "Kund", text: "Oj, tack! Det var snabbt. Hur lång tid tar det innan det syns på kontot?" },
      {
        start: 75,
        end: 88,
        speaker: "Agent",
        text: "Det syns på nästa faktura eller som kredit inom 3-5 vardagar. Jag lägger också en notering så att det inte händer igen. Är det något mer jag kan hjälpa dig med idag?",
      },
      { start: 88, end: 95, speaker: "Kund", text: "Nej, det var allt. Tack för hjälpen, Anna – du var jättebra!" },
      { start: 95, end: 102, speaker: "Agent", text: "Tack själv, ha en bra dag!" },
    ],
  },
  {
    id: "CALL-002",
    title: "Lång väntetid + arg kund – eskaleringsrisk",
    meta: { agent: "Agent-Bengt", duration_s: 310, category: "complaint" },
    segments: [
      { start: 0, end: 5, speaker: "Kund", text: "Ja hallå? Jag har väntat i 45 minuter i kön!" },
      { start: 5, end: 12, speaker: "Agent", text: "Hej, tack för att du väntar. Mitt namn är Bengt. Vad gäller ditt ärende?" },
      {
        start: 12,
        end: 25,
        speaker: "Kund",
        text: "Jag ringde för att säga upp mitt abonnemang för två veckor sedan och jag har fortfarande inte fått bekräftelse. Och nu kommer en ny faktura ändå! Detta är skandal!",
      },
      {
        start: 25,
        end: 35,
        speaker: "Agent",
        text: "Okej, jag förstår att du är upprörd. Men jag behöver ditt kundnummer för att kunna titta.",
      },
      {
        start: 35,
        end: 45,
        speaker: "Kund",
        text: "Jag har redan gett det i kön! Varför kan ni inte ha koll? Jag vill tala med en chef nu!",
      },
      {
        start: 45,
        end: 58,
        speaker: "Agent",
        text: "Jag kan inte koppla dig till chef direkt. Låt mig först kolla status på uppsägningen. Kan du upprepa kundnumret?",
      },
      {
        start: 58,
        end: 68,
        speaker: "Kund",
        text: "19851203-1234. Och jag vill ha skriftlig bekräftelse inom 24 timmar annars kontaktar jag Konsumentverket och ARN!",
      },
      {
        start: 68,
        end: 82,
        speaker: "Agent",
        text: "Okej, jag ser att uppsägningen registrerades den 14 maj men bekräftelsen gick inte iväg pga tekniskt fel. Jag skickar den nu manuellt och krediterar fakturan. Men jag kan tyvärr inte göra mer idag.",
      },
      { start: 82, end: 90, speaker: "Kund", text: "Det här duger inte. Jag är så less på er. Ni hör av er." },
      { start: 90, end: 95, speaker: "Agent", text: "Tack för samtalet." },
    ],
  },
  {
    id: "CALL-003",
    title: "Tekniskt fel + compliance near-miss (QA-flagg)",
    meta: { agent: "Agent-Cecilia", duration_s: 480, category: "tech_support" },
    segments: [
      { start: 0, end: 4, speaker: "Agent", text: "Tjenare, det är Cecilia på support." },
      {
        start: 4,
        end: 15,
        speaker: "Kund",
        text: "Hej, min router har varit nere hela helgen. Jag kan inte jobba. Jag har ringt tidigare och fick löfte om att någon skulle komma ut men ingenting har hänt.",
      },
      { start: 15, end: 22, speaker: "Agent", text: "Okej, tråkigt att höra. Har du provat att starta om routern?" },
      { start: 22, end: 30, speaker: "Kund", text: "Ja, tre gånger! Och jag har bytt sladd. Det är ert fel, inte mitt." },
      {
        start: 30,
        end: 45,
        speaker: "Agent",
        text: "Förstår. Jag kollar i systemet – din linje visar röd sedan fredag. Jag bokar en tekniker till imorgon mellan 8-12. Bekräftar du adressen Storgatan 12?",
      },
      {
        start: 45,
        end: 52,
        speaker: "Kund",
        text: "Ja, det stämmer. Men jag vill ha kompensation för stilleståndet. Jag har förlorat jobbintäkter.",
      },
      {
        start: 52,
        end: 68,
        speaker: "Agent",
        text: "Vi har tyvärr ingen policy för det just nu. Men jag kan ge dig 50 kr rabatt på nästa faktura. Är det okej?",
      },
      {
        start: 68,
        end: 78,
        speaker: "Kund",
        text: "50 kr? Det är ju ingenting. Ni har förstört min helg. Jag vill ha minst 300 kr eller så lämnar jag er.",
      },
      {
        start: 78,
        end: 92,
        speaker: "Agent",
        text: "Låt mig se vad jag kan göra... Okej, jag lägger in 200 kr goodwill-kredit manuellt. Och tekniker imorgon. Tack för tålamodet.",
      },
      { start: 92, end: 100, speaker: "Kund", text: "Okej, det får duga. Men se till att det blir rätt denna gången." },
      { start: 100, end: 108, speaker: "Agent", text: "Absolut. Ha en bra dag." },
    ],
  },
  {
    id: "CALL-004",
    title: "Betalningsproblem + root cause (LLM-berikad)",
    meta: { agent: "Agent-Daniel", duration_s: 390, category: "billing" },
    segments: [
      { start: 0, end: 6, speaker: "Agent", text: "Hej, Daniel här. Vad kan jag stå till tjänst med?" },
      {
        start: 6,
        end: 18,
        speaker: "Kund",
        text: "Jag har fått påminnelse om obetald faktura men jag betalade den förra månaden. Varför kommer det här?",
      },
      {
        start: 18,
        end: 30,
        speaker: "Agent",
        text: "Låt mig kolla. Jag ser att betalningen från 3 april inte har matchats mot rätt faktura i systemet. Det är ett känt problem just nu med vår bankkoppling.",
      },
      {
        start: 30,
        end: 40,
        speaker: "Kund",
        text: "Men jag har kvitto! Jag kan inte ha det här hängande över mig. Det påverkar min kreditvärdighet.",
      },
      {
        start: 40,
        end: 55,
        speaker: "Agent",
        text: "Jag beklagar verkligen. Jag markerar fakturan som betald manuellt nu och lägger en spärr så att inga fler påminnelser går ut. Jag skickar också bekräftelse till din e-post.",
      },
      { start: 55, end: 65, speaker: "Kund", text: "Okej... Men hur kunde det bli så här? Har ni inte koll på era system?" },
      {
        start: 65,
        end: 78,
        speaker: "Agent",
        text: "Det är ett internt IT-problem som vår leverantör håller på att fixa. Vi har haft flera fall den här veckan. Jag lägger en intern incidentrapport så att det inte drabbar fler.",
      },
      { start: 78, end: 88, speaker: "Kund", text: "Tack. Jag hoppas det löser sig fort. Annars byter jag operatör." },
      { start: 88, end: 95, speaker: "Agent", text: "Förstår. Är det något annat jag kan hjälpa till med medan vi har kontakt?" },
    ],
  },
  {
    id: "CALL-005",
    title: "De-eskalering + lätt upsell (positiv vändning)",
    meta: { agent: "Agent-Erika", duration_s: 275, category: "retention" },
    segments: [
      { start: 0, end: 7, speaker: "Agent", text: "Hej, det är Erika. Jag såg att du ringde angående ditt abonnemang." },
      {
        start: 7,
        end: 18,
        speaker: "Kund",
        text: "Ja, jag funderar på att säga upp. Priset har gått upp och jag använder det knappt längre.",
      },
      {
        start: 18,
        end: 28,
        speaker: "Agent",
        text: "Jag förstår. Många upplever samma sak just nu. Får jag fråga vad du använder mest – mobil eller bredband?",
      },
      { start: 28, end: 35, speaker: "Kund", text: "Främst mobilen. Bredbandet har jag via jobbet." },
      {
        start: 35,
        end: 48,
        speaker: "Agent",
        text: "Perfekt. Vi har just nu ett erbjudande där du kan behålla mobilt bredband + 100 GB för 199 kr/mån i 6 månader om du behåller abonnemanget. Det är 30 % lägre än nuvarande pris.",
      },
      { start: 48, end: 58, speaker: "Kund", text: "Hmm, 199 låter bättre. Men jag vill inte bindas i 24 månader igen." },
      {
        start: 58,
        end: 70,
        speaker: "Agent",
        text: "Ingen bindningstid på det här erbjudandet. Du kan säga upp när som helst efter 6 månader. Vill du att jag aktiverar det nu?",
      },
      { start: 70, end: 78, speaker: "Kund", text: "Okej, kör på. Men bara om det verkligen blir 199." },
      {
        start: 78,
        end: 88,
        speaker: "Agent",
        text: "Klart det blir. Jag aktiverar det nu och skickar bekräftelse. Tack för att du stannar hos oss – uppskattas!",
      },
      { start: 88, end: 93, speaker: "Kund", text: "Tack själv. Hej då." },
    ],
  },
];
