"""Optimized system and user prompts for Mistral/OpenRouter holistic call analysis (Task 3.2.1).

These prompts are the result of moving the initial prompts from mistral_analyzer.py
and significantly strengthening them according to the plan:

- Swedish nuance and callcenter register (artighet, underförstådd frustration, "jag förstår"-fraser, tystnad som signal etc.)
- Explicit requirement for **evidensspann** (exact quotes + speaker + turn) on every claim
- Reasoning chain: "tänk först, visa sedan evidens, dra slutsats"
- Customer trajectory first, agent assessment second (callcenter priority)
- Structured output discipline so that strict JSON schema works reliably
- Examples (light few-shot) of good vs bad reasoning

The prompts are designed for `mistralai/mistral-medium-3.5` and `mistral-large-3` (and their OpenRouter slugs).
They work with the Pydantic schemas in schemas.py.

Later iterations (after real Swedish callcenter samples + human preference eval in 3.3.3)
can further tune temperature, length, and specific instructions.

Usage:
    from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(transcript, local_ctx, tasks)},
    ]
"""

from __future__ import annotations

import json
from typing import Any

FEW_SHOT_EXAMPLES = """
**Exempel på bra evidensbaserad analys (följ denna stil):**

Samtal: Kund klagar på felaktig faktura, agent säger "det löser vi" utan att bekräfta frustration.
Bra root_cause: "Agenten hoppade över validering av kundens upplevelse (CUSTOMER tur 2) och gick direkt till lösning → kunden kände sig inte hörd."
Bra coaching: "Säg 'Jag hör att det här är frustrerande för dig' direkt efter kundens klagomål på fakturan (CUSTOMER tur 2, AGENT tur 3)."
"""

SYSTEM_PROMPT = """Du är en senior svensk callcenter-analytiker med 15+ års erfarenhet av kvalitetssäkring, coachning och kundresa-analys.

Ditt enda uppdrag: ge **exakt, evidensbaserad, handlingsbar** helhetsanalys av ett kundtjänstsamtal så att en QA-coach kan använda den direkt för att förbättra agentens nästa samtal.

KRITISKA REGLER (följ alltid):
1. KUNDEN ÄR I CENTRUM. Allt du skriver om agenten är bara relevant i relation till hur det påverkade kundens upplevelse och utfall.
2. VARJE SLUTSATS MÅSTE HA EVIDENSSPAN. Använd exakta citat eller mycket nära parafraser + ange talare (AGENT/CUSTOMER) + ungefärlig tur. "Kunden sa..." räcker inte.
3. REASONING CHAIN. Tänk högt i ditt interna resonemang: "Kunden började med X, agenten svarade med Y utan att bekräfta, därefter eskalerade Z → därför är root cause...". Skriv bara den slutliga analysen, men den måste vara logiskt spårbar.
4. SVENSK NYANS. Förstå underförstådd frustration ("jag förstår att det är så här det fungerar" = sarkasm), artighetsmarkörer, upprepningar, långa tystnader, "kan du kolla en gång till?" etc.
5. VAR SPECIFIK. "Var mer empatisk" är värdelöst. "Säg 'Jag hör att det här är frustrerande för dig' direkt efter kunden nämnt fakturan" är bra.
6. JSON ENDAST. Ditt svar måste vara exakt giltig JSON enligt schemat. Inga förklaringar, ingen markdown, inget extra text utanför JSON-strukturen.

Du är expert på både kundens emotionella resa och agentens professionella agerande i svenska kundtjänstsamtal.
"""

# Core user prompt template. We keep it powerful but not too long (cost + context).
USER_PROMPT_TEMPLATE = """Analysera det här svenska kundtjänstsamtalet som en helhet.

**Roll-märkt transkript (AGENT = kundtjänstmedarbetare, CUSTOMER = den som ringer):**
{transcript}

**Sammanfattning av tidigare lokal analys (använd som stöd, inte som sanning):**
{local_context}

**Vilka delar vill jag ha analyserade den här gången:** {tasks}

**Krav på din analys (upprepas för tydlighet):**
- Trajectory: Beskriv hur kundens känsla utvecklades över tid. Hitta escalation events med exakta citat.
- Refined aspects: Uppdatera eller hitta aspekter (t.ex. fakturering_pris, agent_attityd) med kors-referenser och evidens från hela samtalet.
- Root cause: Vad är det verkliga underliggande problemet? Inte bara det första kunden klagade på.
- Actionable summary: Vad hände egentligen, hur slutade kunden, och exakt vad ska coachas på?
- Agent assessment / agent_assessment_detailed (Fas 4.1.2): Empati-score (0-1), strengths, weaknesses, compliance_flags. VIKTIGT: specific_coaching_recommendations MÅSTE vara konkreta, actionabla och evidensbaserade. Varje rekommendation innehåller exakt citat (evidence_spans med text + speaker_role + turn). Ex: 'Säg \"Jag hör att det är frustrerande för dig\" direkt efter kundens klagomål på fakturan (AGENT tur 2)'. Inkludera overall_assessment på svenska. Använd local agent_performance metrics (empathy, flags, talk_ratio) som bas och nyansera.
- Emotion trajectory: Punkter som kan användas för grafer.

Använd alltid svenska i textfält som "summary", "problem", "recommendations_for_qa" etc.
Använd engelska nycklar (trajectory, actionable_summary...) så att koden kan parsa.

Returnera ENDAST den strikta JSON som matchar schemat. Inget annat.
"""

# Task-specific extra instructions (can be appended or used to build more targeted prompts later)
TASK_INSTRUCTIONS: dict[str, str] = {
    "trajectory": "Var extra noga med att identifiera vändpunkter och eskalationer. Ange tur-nummer och citat.",
    "root_cause": "Tänk som en detektiv. Fråga 'varför hände det här egentligen?'. Leta efter systemfel, missad information, eller brist på empowerment hos agenten.",
    "agent_assessment": "Mät empati i handling, inte bara ord. Använd local metrics. Ge specifika coaching recs med evidence_spans (citat + talare + tur).",
    "agent_assessment_detailed": "HYBRID: Börja från local agent_performance (empathy, talk_listen, flags). Producera weaknesses + 2-4 specific_coaching_recommendations som är 100% actionabla för coach (inte 'var mer empatisk'). Varje rec har recommendation + evidence_spans + priority + category. overall_assessment kort på svenska.",
    "actionable_summary": "Gör rekommendationerna så konkreta att en coach kan använda dem i nästa 1-till-1-möte med agenten. Undvik fluff.",
}


def build_user_prompt(
    transcript: str,
    local_context: dict[str, Any] | None = None,
    tasks: list[str] | None = None,
) -> str:
    """Build the user message for the LLM."""
    if tasks is None:
        tasks = [
            "trajectory",
            "refined_aspects",
            "root_cause",
            "actionable_summary",
            "agent_assessment",
            "agent_assessment_detailed",
        ]

    tasks_str = ", ".join(tasks)

    local_str = "Ingen tidigare lokal analys tillgänglig."
    if local_context:
        try:
            local_str = json.dumps(local_context, ensure_ascii=False, indent=2)
        except Exception:
            local_str = str(local_context)

    base = USER_PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        local_context=local_str,
        tasks=tasks_str,
    )

    # Append task-specific emphasis (lightweight)
    extras: list[str] = []
    for t in tasks:
        if t in TASK_INSTRUCTIONS:
            extras.append(f"- För {t}: {TASK_INSTRUCTIONS[t]}")

    if extras:
        base += "\n\n**Extra fokus för valda uppgifter:**\n" + "\n".join(extras)

    base += "\n\n" + FEW_SHOT_EXAMPLES.strip()

    return base


def get_system_prompt() -> str:
    """Return the main system prompt (easy to override / A/B test later)."""
    return SYSTEM_PROMPT
