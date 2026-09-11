"""Genererade svenska kundtjänsttexter för live-testning mot LM Studio (Qwen 3.5 9B 65k).

Varje scenario är en realistisk kundtjänstkonversation med olika sentiment-
 och eskalationsprofiler. Texterna är designade för att testa:
 - Svensk språkförståelse (nyanser, artighet, underförstådd frustration)
 - Evidensbaserad analys (citat + talare + tur)
 - Sentiment- och eskalationsdetektion
 - Root cause-analys och agentbedömning
 - Strukturerad JSON-output enligt CallLLMOutput-schema

Inga riktiga PII – all data är fiktiv.
"""

from __future__ import annotations

from typing import Any


def _seg(
    speaker: str,
    text: str,
    idx: int,
    start: float = 0.0,
    end: float = 0.0,
) -> dict[str, Any]:
    """Bygg ett segment-dict i formatet ConversationMistralAnalyzer förväntar."""
    return {
        "speaker": speaker,
        "text": text,
        "segment_id": idx,
        "start": start if start else idx * 5.0,
        "end": end if end else (idx + 1) * 5.0,
    }


# =============================================================================
# Scenario 1: Fakturatvist — arg kund, agent missar empati (eskalerande)
# =============================================================================

FAKTURA_TVIST: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten, med vad kan jag hjälpa dig idag?", 0),
    _seg("SPEAKER_01", "Ja, jag har fått en faktura som är helt fel. Ni har debiterat mig dubbelt.", 1),
    _seg("SPEAKER_00", "Okej, låt mig kolla upp det här. Kan du ge mig ditt fakturanummer?", 2),
    _seg("SPEAKER_01", "Det är 47291. Jag har redan ringt två gånger om det här och ingen har löst det.", 3),
    _seg("SPEAKER_00", "Jag förstår. Jag ska titta på det. Det verkar som att det är en teknisk fel.", 4),
    _seg("SPEAKER_01", "En teknisk fel? Jag har betalat för mycket i två månader nu. Det är inte okej.", 5),
    _seg("SPEAKER_00", "Ja, jag förstår att det är så här det fungerar ibland. Jag kan kreditnota dig.", 6),
    _seg("SPEAKER_01", "Så här det fungerar? Jag vill ha pengarna tillbaka, inte en kreditnota!", 7),
    _seg("SPEAKER_00", "Tyvärr kan jag bara göra en kreditnota enligt våra rutiner.", 8),
    _seg("SPEAKER_01", "Det här är helt oacceptabelt. Jag vill prata med en chef.", 9),
    _seg("SPEAKER_00", "Jag kan eskalera ärendet, men det kan ta upp till fem arbetsdagar.", 10),
    _seg("SPEAKER_01", "Fem arbetsdagar? För ett fel som ni har gjort? Ni måste skämta.", 11),
    _seg("SPEAKER_00", "Jag förstår att det är frustrerande, men det är vad jag kan göra just nu.", 12),
    _seg("SPEAKER_01", "Om det här inte är löst till veckan säger jag upp hela abonnemanget.", 13),
]

FAKTURA_TVIST_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 2: Enkel fråga — lugn, kort samtal (positivt)
# =============================================================================

ENKEL_FRAGA: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Hej och välkommen! Vad kan jag hjälpa dig med idag?", 0),
    _seg("SPEAKER_01", "Hej! Jag undrar bara vilka öppettider ni har på helgerna?", 1),
    _seg("SPEAKER_00", "Vi har öppet lördag och söndag mellan klockan 10 och 16.", 2),
    _seg("SPEAKER_01", "Bra, tack! Behöver jag boka tid eller kan jag bara komma in?", 3),
    _seg("SPEAKER_00", "Du kan bara komma in, men det kan vara lite kö på förmiddagen.", 4),
    _seg("SPEAKER_01", "Okej, då väntar jag till eftermiddag. Tack för hjälpen!", 5),
    _seg("SPEAKER_00", "Varsågod! Hoppas vi ses då. Ha en bra dag!", 6),
    _seg("SPEAKER_01", "Tack, detsamma!", 7),
]

ENKEL_FRAGA_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 3: Teknisk support — blandat sentiment, problemlösning
# =============================================================================

TEKNIK_SUPPORT: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Kundtjänsten, hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Ja, mitt internet har varit instabilt i tre dagar nu. Det fungerar och sen fungerar det inte.", 1),
    _seg("SPEAKER_00", "Jag förstår att det är frustrerande. Låt oss felsöka. Är din router inkopplad?", 2),
    _seg("SPEAKER_01", "Ja, allt är inkopplad som det ska. Jag har startat om den tre gånger.", 3),
    _seg("SPEAKER_00", "Okej, bra. Kan du berätta vilken ljusindikator du ser på routern just nu?", 4),
    _seg("SPEAKER_01", "Det lyser rött där det ska vara grönt. Internet-lampan alltså.", 5),
    _seg("SPEAKER_00", "Rött på internet-lampan tyder på att vi har ett problem på vår sida. Jag ska undersöka.", 6),
    _seg("SPEAKER_01", "Äntligen någon som lyssnar. Jag har försökt få hjälp i flera dagar.", 7),
    _seg("SPEAKER_00", "Jag hör dig. Det låter som att det kan finnas en störning i ditt område. Låt mig kolla.", 8),
    _seg("SPEAKER_01", "Okej, väntar.", 9),
    _seg("SPEAKER_00", "Ja, jag kan se att det finns en kabelbrott på din adress. Vi har en tekniker på väg.", 10),
    _seg("SPEAKER_01", "Bra! När tror ni att det är löst?", 11),
    _seg("SPEAKER_00", "Vi räknar med att det är åtgärdat inom 24 timmar. Jag lägger en notering på ditt konto.", 12),
    _seg("SPEAKER_01", "Tack, det känns bra att äntligen få ett svar.", 13),
    _seg("SPEAKER_00", "Varsågod. Ring oss igen om det inte fungerar imorgon. Ha en bra dag!", 14),
    _seg("SPEAKER_01", "Tack för hjälpen, hej då!", 15),
]

TEKNIK_SUPPORT_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 4: Uppsägningshot — hög eskalation, kund vill säga upp
# =============================================================================

UPPSAGNINGSHOT: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten. Hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Jag vill säga upp mitt abonnemang. Direkt.", 1),
    _seg("SPEAKER_00", "Jag är ledsen att höra det. Kan jag fråga varför du vill säga upp?", 2),
    _seg("SPEAKER_01", "För att er service är usel. Jag har haft problem i månader och ingen tar ansvar.", 3),
    _seg("SPEAKER_00", "Jag förstår att du är missnöjd. Låt mig se vad jag kan göra för dig.", 4),
    _seg("SPEAKER_01", "Vad du kan göra? Ni har logat mig att ringa tillbaka tre gånger och det har ni inte gjort.", 5),
    _seg("SPEAKER_00", "Det låter inte alls bra. Jag ber om ursäkt för det. Låt mig titta på din historik.", 6),
    _seg("SPEAKER_01", "Ursäkter hjälper inte. Jag vill ha en lösning eller så är jag borta.", 7),
    _seg("SPEAKER_00", "Jag hör dig. Jag kan erbjuda dig en rabatt på tre månader om du stannar.", 8),
    _seg("SPEAKER_01", "En rabatt? Tror du att det löser grundproblemet? Problemet är att ni inte levererar.", 9),
    _seg("SPEAKER_00", "Du har rätt. Låt mig eskalera det här till vår kundklubb som kan hantera det ordentligt.", 10),
    _seg("SPEAKER_01", "Äntligen. Men om inte den här gången fungerar så är det definitivt slut.", 11),
    _seg("SPEAKER_00", "Jag förstår. Jag lägger en prioriterad ärendehantering och de hör av sig inom 24 timmar.", 12),
    _seg("SPEAKER_01", "Okej. Det är er sista chans.", 13),
    _seg("SPEAKER_00", "Jag uppskattar att du ger oss den chansen. Jag noterar det i ärendet. Ha en bra dag.", 14),
    _seg("SPEAKER_01", "Vi får se. Hej.", 15),
]

UPPSAGNINGSHOT_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 5: Långt samtal — för kontextbudgettestning (65k)
# =============================================================================

def _build_long_conversation() -> list[dict[str, Any]]:
    """Bygg en lång konversation genom att upprepa och variera scenario 3."""
    base = TEKNIK_SUPPORT
    segments: list[dict[str, Any]] = []
    for cycle in range(8):
        for seg in base:
            new_seg = dict(seg)
            new_seg["segment_id"] = len(segments)
            new_seg["start"] = len(segments) * 5.0
            new_seg["end"] = (len(segments) + 1) * 5.0
            # Variera texten något per cykel för att undvika identisk upprepning
            suffix = f" (del {cycle + 1})" if cycle > 0 else ""
            new_seg["text"] = seg["text"] + suffix
            segments.append(new_seg)
    return segments


LANG_KONVERSATION: list[dict[str, Any]] = _build_long_conversation()
LANG_KONVERSATION_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 6: Sarkasm — underförstådd frustration, "jag förstår att det är så här"
# =============================================================================

SARKASM: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten, hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Ja, jag har väntat i 45 minuter i kö, men det är väl så det fungerar hos er.", 1),
    _seg("SPEAKER_00", "Jag ber om ursäkt för väntetiden. Vad gäller det?", 2),
    _seg("SPEAKER_01", "Jo, jag har ju betalat för en tjänst som inte fungerar. Men det är väl normalt att man betalar för saker som inte fungerar?", 3),
    _seg("SPEAKER_00", "Jag förstår att du är frustrerad. Kan du berätta mer om problemet?", 4),
    _seg("SPEAKER_01", "Åh, jag får gärna berätta mer. Det är ju inte som om jag har gjort det tre gånger redan.", 5),
    _seg("SPEAKER_00", "Jag ska verkligen titta på det här nu. Vad är det specifika problemet?", 6),
    _seg("SPEAKER_01", "Det specifika problemet? Tja, internet fungerar inte. Men det är väl för mycket begärt att det ska fungera när man betalar för det?", 7),
    _seg("SPEAKER_00", "Jag förstår. Låt mig felsöka. Är routern inkopplad?", 8),
    _seg("SPEAKER_01", "Självklart är den inkopplad. Tror ni att jag ringer för att jag inte vet hur man kopplar in en router?", 9),
    _seg("SPEAKER_00", "Jag ber om ursäkt, jag måste fråga för att utesluta grundläggande fel.", 10),
    _seg("SPEAKER_01", "Självklart, det är ju inte som om jag kan förvänta mig att ni har koll på mitt ärende redan.", 11),
    _seg("SPEAKER_00", "Jag kan se att det finns en störning i ditt område. Vi arbetar på det.", 12),
    _seg("SPEAKER_01", "Åh, ni arbetar på det? Vilken lyx. Då kan jag ju slappna av och vänta i evighet.", 13),
    _seg("SPEAKER_00", "Vi räknar med att det är löst inom 24 timmar.", 14),
    _seg("SPEAKER_01", "24 timmar? Självklart. Det är ju inte som om jag behöver internet för att jobba eller något.", 15),
]

SARKASM_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 7: Tystnad/avbrott — kund tystnar, agent fyller inte tystnaden
# =============================================================================

TYSTNAD: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Hej och välkommen till kundtjänsten. Vad kan jag hjälpa dig med?", 0),
    _seg("SPEAKER_01", "Jo... det är så att... jag vet inte riktigt hur jag ska börja.", 1),
    _seg("SPEAKER_00", "Ta god tid på dig.", 2),
    _seg("SPEAKER_01", "...", 3),
    _seg("SPEAKER_00", "Är du kvar?", 4),
    _seg("SPEAKER_01", "Ja, jag funderar. Det gäller mitt abonnemang.", 5),
    _seg("SPEAKER_00", "Okej.", 6),
    _seg("SPEAKER_01", "...", 7),
    _seg("SPEAKER_00", "Vad är det som gäller med abonnemanget?", 8),
    _seg("SPEAKER_01", "Jag vet inte om jag vill fortsätta. Det har varit så mycket problem.", 9),
    _seg("SPEAKER_00", "Jag förstår.", 10),
    _seg("SPEAKER_01", "...", 11),
    _seg("SPEAKER_00", "Finns det något specifikt som gör att du funderar på att säga upp?", 12),
    _seg("SPEAKER_01", "Det är inte en sak. Det är allt. Fakturorna, servicen, supporten.", 13),
    _seg("SPEAKER_00", "Jag hör att det har varit frustrerande. Låt oss titta på det tillsammans.", 14),
    _seg("SPEAKER_01", "Okej... jag kanske ger det en chans till om vi kan lösa det nu.", 15),
]

TYSTNAD_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 8: Multi-issue — kund med flera problem, agent tappar tråden
# =============================================================================

MULTI_ISSUE: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen! Hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Jag har flera problem. Först är min faktura fel, sen fungerar inte internet, och jag vill också byta paket.", 1),
    _seg("SPEAKER_00", "Okej, låt oss ta ett i taget. Börja med fakturan.", 2),
    _seg("SPEAKER_01", "Fakturan visar 800 kronor men jag betalar för 500-paketet.", 3),
    _seg("SPEAKER_00", "Jag kan se att du har lagt till en extra tjänst. Det är därför.", 4),
    _seg("SPEAKER_01", "Vilken extra tjänst? Jag har inte lagt till något.", 5),
    _seg("SPEAKER_00", "Det står här att du har premium-support.", 6),
    _seg("SPEAKER_01", "Jag har aldrig beställt premium-support. Och det leder mig till internet-problemet.", 7),
    _seg("SPEAKER_00", "Okej, vad är problemet med internet?", 8),
    _seg("SPEAKER_01", "Det har varit nere i två veckor. Jag har ringt tre gånger.", 9),
    _seg("SPEAKER_00", "Jag ser inte några tidigare ärenden om det.", 10),
    _seg("SPEAKER_01", "Då förstår du varför jag vill byta paket också. Jag betalar för mycket och får inget.", 11),
    _seg("SPEAKER_00", "Låt mig titta på alltihop. Det kan ta en stund.", 12),
    _seg("SPEAKER_01", "Så jag ska vänta igen? Medan jag betalar för saker jag inte har beställt?", 13),
    _seg("SPEAKER_00", "Jag förstår att det är frustrerande. Jag ska prioritera ditt ärende.", 14),
    _seg("SPEAKER_01", "Bra. Men jag vill ha en lösning på alla tre problem idag.", 15),
]

MULTI_ISSUE_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 9: Förvirrad kund — äldre person, agent måste förklara flera gånger
# =============================================================================

FORVIRRAD_KUND: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Hej och välkommen! Vad kan jag hjälpa dig med idag?", 0),
    _seg("SPEAKER_01", "Hej... jag vet inte riktigt. Jag fick ett brev från er, men jag förstår inte vad det betyder.", 1),
    _seg("SPEAKER_00", "Inga problem. Kan du berätta vad det står i brevet?", 2),
    _seg("SPEAKER_01", "Det står något om faktura, men jag betalar ju via autogiro. Varför får jag brev?", 3),
    _seg("SPEAKER_00", "Det kan vara en påminnelse eller en specifik faktura. Låt mig kolla ditt konto.", 4),
    _seg("SPEAKER_01", "Men jag betalar ju varje månad. Det har alltid fungerat.", 5),
    _seg("SPEAKER_00", "Jag förstår. Det verkar som att autogiro misslyckades förra månaden.", 6),
    _seg("SPEAKER_01", "Misslyckades? Hur kan det misslyckas när jag har pengar på kontot?", 7),
    _seg("SPEAKER_00", "Det kan bero på ett tekniskt fel. Vi kan åtgärda det. Vill du att jag hjälper dig med det?", 8),
    _seg("SPEAKER_01", "Ja, men jag förstår inte hur det här kan hända. Jag har haft autogiro i flera år.", 9),
    _seg("SPEAKER_00", "Jag förstår att det är förvirrande. Låt mig förklara vad som hände och hur vi fixar det.", 10),
    _seg("SPEAKER_01", "Vänligen, ta det lugnt. Jag är inte så bra på datorer och sånt här.", 11),
    _seg("SPEAKER_00", "Ingen fara. Jag går steg för steg. Först ska jag kontrollera att dina bankuppgifter stämmer.", 12),
    _seg("SPEAKER_01", "Okej. Men betalar jag extra för det här?", 13),
    _seg("SPEAKER_00", "Nej, du betalar inte extra. Det är ett fel från vår sida och vi åtgärdar det gratis.", 14),
    _seg("SPEAKER_01", "Åh, tack. Då känner jag mig lugnare. Kan du hjälpa mig att säkerställa att det inte händer igen?", 15),
]

FORVIRRAD_KUND_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 10: Empatiskt samtal — agent gör allt rätt (positivt)
# =============================================================================

EMPATISKT_SAMTAL: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Hej och varmt välkommen! Vad kan jag hjälpa dig med idag?", 0),
    _seg("SPEAKER_01", "Hej. Jag har haft ett problem med min faktura och jag är lite orolig.", 1),
    _seg("SPEAKER_00", "Jag förstår att det kan vara oroande. Låt oss titta på det tillsammans så löser vi det.", 2),
    _seg("SPEAKER_01", "Tack, det känns bra att höra. Fakturan visar 1200 kronor men det brukar vara 600.", 3),
    _seg("SPEAKER_00", "Okej, låt mig kolla. Jag kan se att det har lagts till en extra tjänst i mars. Vet du om du beställt något?", 4),
    _seg("SPEAKER_01", "Nej, jag har inte beställt något extra. Men min son kanske har gjort det?", 5),
    _seg("SPEAKER_00", "Det är möjligt. Jag kan se att beställningen kom från din enhet. Men oroa dig inte, vi kan kreditera beloppet och avsluta tjänsten.", 6),
    _seg("SPEAKER_01", "Åh, vad snällt. Det låter jättebra.", 7),
    _seg("SPEAKER_00", "Jag lägger en kredit på 600 kronor och avslutar extra-tjänsten. Det blir fixat inom 24 timmar.", 8),
    _seg("SPEAKER_01", "Tack så mycket! Det känns verkligen bra att få så bra hjälp.", 9),
    _seg("SPEAKER_00", "Varsågod. Jag vill också hjälpa dig att sätta ett lösenordsskydd så att det inte händer igen.", 10),
    _seg("SPEAKER_01", "Ja, det vore jättebra. Hur gör vi det?", 11),
    _seg("SPEAKER_00", "Jag skickar en länk till din e-post där du kan sätta en PIN-kod för framtida beställningar.", 12),
    _seg("SPEAKER_01", "Perfekt! Tack för all hjälp. Ni är verkligen professionella.", 13),
    _seg("SPEAKER_00", "Tack för de orden! Är det något annat jag kan hjälpa dig med idag?", 14),
    _seg("SPEAKER_01", "Nej, allt är löst. Ha en fortsatt bra dag!", 15),
]

EMPATISKT_SAMTAL_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 11: Kort argt samtal — kund avbryter och lägger på
# =============================================================================

KORT_ARGT: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten, vad gäller det?", 0),
    _seg("SPEAKER_01", "Det gäller att er service är skräp och jag vill ha pengarna tillbaka!", 1),
    _seg("SPEAKER_00", "Jag är ledsen att höra det. Kan du berätta vad som har hänt?", 2),
    _seg("SPEAKER_01", "Vad som hänt? Allt har hänt! Ingenting fungerar och ni gör inget åt det!", 3),
    _seg("SPEAKER_00", "Jag förstår att du är arg. Låt mig titta på ditt ärende.", 4),
    _seg("SPEAKER_01", "Ni har redan tittat fem gånger och inget har hänt. Jag är klar med er.", 5),
    _seg("SPEAKER_00", "Jag ber dig att ge mig en chans att lösa det här nu.", 6),
    _seg("SPEAKER_01", "Nej. Jag ringer inte mer. Jag säger upp och går till en konkurrent.", 7),
    _seg("SPEAKER_00", "Jag förstår ditt beslut. Men innan du går, kan jag erbjuda dig en lösning?", 8),
    _seg("SPEAKER_01", "Nej. Det är för sent. Hej.", 9),
]

KORT_ARGT_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 12: Återkommande kund — trött, har ringt många gånger
# =============================================================================

ATERKOMMANDE_KUND: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten. Hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Ja, det här är femte gången jag ringer om samma problem. Jag är trött på det här.", 1),
    _seg("SPEAKER_00", "Jag ber om ursäkt för att du har behövt ringa flera gånger. Vad är det som gäller?", 2),
    _seg("SPEAKER_01", "Mitt internet fungerar inte. Det har varit så i tre veckor nu. Varje gång jag ringer får jag löften om att det ska fixas.", 3),
    _seg("SPEAKER_00", "Jag förstår att det är frustrerande. Låt mig titta på din historik.", 4),
    _seg("SPEAKER_01", "Min historik? Den visar väl att inget har gjorts. Fem samtal och inget resultat.", 5),
    _seg("SPEAKER_00", "Jag kan se att det finns flera ärenden. Jag ska se till att det här prioriteras.", 6),
    _seg("SPEAKER_01", "Prioriteras? Det har ni sagt varje gång. Jag har hört det förut.", 7),
    _seg("SPEAKER_00", "Jag förstår din skepsis. Den är berättigad. Men jag ska personligen följa upp det här ärendet.", 8),
    _seg("SPEAKER_01", "Personligen? Vad betyder det i praktiken? Att du ringer mig tillbaka?", 9),
    _seg("SPEAKER_00", "Ja, jag lovar att jag ringer dig tillbaka inom 24 timmar med en konkret lösning.", 10),
    _seg("SPEAKER_01", "Okej. Men om inte det här fungerar så är det slut. Jag menar allvar.", 11),
    _seg("SPEAKER_00", "Jag tar det på stort allvar. Jag noterar ditt nummer och ärendet som prioriterat.", 12),
    _seg("SPEAKER_01", "Bra. Då väntar jag på ditt samtal. Men jag hoppas du förstår att mitt tålamod är slut.", 13),
    _seg("SPEAKER_00", "Jag förstår. Jag uppskattar att du ger oss en sista chans. Vi hörs imorgon.", 14),
    _seg("SPEAKER_01", "Okej. Hej då.", 15),
]

ATERKOMMANDE_KUND_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 13: Code-switching — svenska + engelska termer (IT-support)
# =============================================================================

CODE_SWITCHING: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till teknisk support. Hur kan jag hjälpa dig?", 0),
    _seg("SPEAKER_01", "Hej, jag har problem med min router. Den har ingen internet connection alls.", 1),
    _seg("SPEAKER_00", "Okej, låt oss felsöka. Kan du se om det är en röd eller grön LED på routern?", 2),
    _seg("SPEAKER_01", "Det är rött. Jag har försökt reset flera gånger men det hjälper inte.", 3),
    _seg("SPEAKER_00", "Har du kollat om det är ett problem med din ISP eller med själva routern?", 4),
    _seg("SPEAKER_01", "Jag vet inte. Men jag har försökt pinga 8.8.8.8 och det fungerar inte.", 5),
    _seg("SPEAKER_00", "Bra felsökning. Låt mig kolla om det är en störning på din adress.", 6),
    _seg("SPEAKER_01", "Okej. Men kan det också vara att min firmware är outdated?", 7),
    _seg("SPEAKER_00", "Det är möjligt. Men jag kan se att det finns en kabelbrott i ditt område.", 8),
    _seg("SPEAKER_01", "En kabelbrot? Då är det inte min router som är problemet alltså.", 9),
    _seg("SPEAKER_00", "Rätt. Vi har en tekniker på väg. Det bör vara löst inom 24 timmar.", 10),
    _seg("SPEAKER_01", "Okej, tack. Men ska jag uppdatera firmware ändå när det är fixat?", 11),
    _seg("SPEAKER_00", "Ja, det är alltid bra att hålla firmware uppdaterat. Jag skickar en guide.", 12),
    _seg("SPEAKER_01", "Perfekt, tack för hjälpen!", 13),
    _seg("SPEAKER_00", "Varsågod! Ring oss igen om det inte fungerar. Ha en bra dag!", 14),
    _seg("SPEAKER_01", "Tack, detsamma!", 15),
]

CODE_SWITCHING_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 14: Endast agent talar (edge case — monolog)
# =============================================================================

ENDAST_AGENT: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Välkommen till kundtjänsten. Hur kan jag hjälpa dig idag?", 0),
    _seg("SPEAKER_00", "Hallå? Är du kvar?", 1),
    _seg("SPEAKER_00", "Jag hör inget. Om du är kvar, säg till.", 2),
    _seg("SPEAKER_00", "Okej, det verkar som att vi tappade kontakten. Jag lägger på och du får ringa igen.", 3),
]

ENDAST_AGENT_ROLE_MAP: dict[str, str] = {
    "SPEAKER_00": "agent",
}


# =============================================================================
# Scenario 15: Endast kund talar (edge case — kund lämnar meddelande)
# =============================================================================

ENDAST_KUND: list[dict[str, Any]] = [
    _seg("SPEAKER_01", "Hej, jag ringer om min faktura. Den är fel.", 0),
    _seg("SPEAKER_01", "Hallå? Finns det någon där?", 1),
    _seg("SPEAKER_01", "Okej, ingen svarar. Jag ville bara säga att fakturan visar 1500 men det ska vara 800.", 2),
    _seg("SPEAKER_01", "Ring mig tillbaka på 070-1234567. Tack.", 3),
]

ENDAST_KUND_ROLE_MAP: dict[str, str] = {
    "SPEAKER_01": "customer",
}


# =============================================================================
# Scenario 16: Okända talare (edge case — ingen role_map)
# =============================================================================

OKANDA_TALARE: list[dict[str, Any]] = [
    _seg("SPEAKER_00", "Hej, vad gäller det?", 0),
    _seg("SPEAKER_01", "Jag har en fråga om mitt konto.", 1),
    _seg("SPEAKER_00", "Visa, låt mig kolla det.", 2),
    _seg("SPEAKER_01", "Tack, det uppskattar jag.", 3),
]

# Ingen role_map — testen ska verifiera att systemet hanterar okända roller


# =============================================================================
# Kort test-texter för chat_completion-tester
# =============================================================================

KORTA_TEXTER: list[dict[str, str]] = [
    {
        "prompt": "Skriv en kort svensk sammanfattning av: kund ringer och klagar på felaktig faktura, agenten lovar att kreditera.",
        "expect_contains": ["faktura", "kredit"],
    },
    {
        "prompt": "Vilken sentiment har följande mening? 'Jag är väldigt nöjd med er service.' Svara med ett ord.",
        "expect_contains": ["positiv"],
    },
    {
        "prompt": "Vilken sentiment har följande mening? 'Det här är helt oacceptabelt och jag är arg.' Svara med ett ord.",
        "expect_contains": ["negativ"],
    },
    {
        "prompt": "Vilken sentiment har följande mening? 'Jag vet inte, det kanske fungerar.' Svara med ett ord.",
        "expect_contains": ["neutral"],
    },
    {
        "prompt": "Översätt till formell svenska: 'Can you please check my invoice again?'",
        "expect_contains": ["faktura", "kontroller"],
    },
    {
        "prompt": "Skriv en empatisk svensk fras som en kundtjänstagent kan säga till en frustrerad kund.",
        "expect_contains": ["förstår"],
    },
    # --- Nya korta texter ---
    {
        "prompt": "Klassificera sentiment i följande svenska kundmeddelande: 'Tack för utmärkt hjälp, ni löste mitt problem snabbt!' Svara med: positiv, neutral eller negativ.",
        "expect_contains": ["positiv"],
    },
    {
        "prompt": "Klassificera sentiment i följande svenska kundmeddelande: 'Jag har väntat i 45 minuter och ingen svarar. Det här är en katastrof.' Svara med: positiv, neutral eller negativ.",
        "expect_contains": ["negativ"],
    },
    {
        "prompt": "Extrahera alla namn och belopp från följande text: 'Kund Anna Svensson ringde om faktura 4500 kr för juni. Hon betalade 5000 kr.' Returnera som en lista.",
        "expect_contains": ["anna", "svensson", "4500"],
    },
    {
        "prompt": "Skriv en svensk sammanfattning av följande kundtjänstsamtal: 'Agent: Välkommen! Kund: Jag vill säga upp mitt abonnemang. Agent: Jag förstår, kan jag fråga varför? Kund: Er service fungerar inte. Agent: Jag kan erbjuda en lösning. Kund: För sent.'",
        "expect_contains": ["uppsäg", "service"],
    },
    {
        "prompt": "Omformulera följande informella svenska till formell: 'Hej, jag har krångel med min grej, kan du kolla?'",
        "expect_contains": ["problem", "kontroller"],
    },
    {
        "prompt": "Skriv tre konkreta coaching-råd (på svenska) till en kundtjänstagent som missade att visa empati mot en arg kund.",
        "expect_contains": ["empathi", "kund"],
    },
    {
        "prompt": "Vilken risknivå (låg, medel, hög) indikerar följande kundmeddelande? 'Om ni inte löser det här idag ringer jag er chef och säger upp mig.' Svara med ett ord.",
        "expect_contains": ["hög"],
    },
    {
        "prompt": "Identifiera om följande mening innehåller sarkasm: 'Åh, jättebra, ännu en försening. Ni är verkligen bäst på att inte leverera.' Svara ja eller nej.",
        "expect_contains": ["ja"],
    },
    {
        "prompt": "Skriv en svensk deeskalationsfras som en agent kan använda när en kund skriker.",
        "expect_contains": ["lugn", "förstår"],
    },
]


# =============================================================================
# Registry för parametriserade tester
# =============================================================================

SCENARIER: list[dict[str, Any]] = [
    {
        "namn": "faktura_tvist",
        "segments": FAKTURA_TVIST,
        "role_map": FAKTURA_TVIST_ROLE_MAP,
        "expect_escalation": True,
        "expect_negative": True,
        # Root cause kan vara djupare än ytnivå — godkänn valfritt relevant nyckelord
        "expect_root_cause_any": ["faktura", "dubbel", "empathi", "empowerment", "kredit", "pengar", "agent", "befogenhet"],
        "expect_coaching": True,
    },
    {
        "namn": "enkel_fraga",
        "segments": ENKEL_FRAGA,
        "role_map": ENKEL_FRAGA_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": False,
        "expect_root_cause_any": ["öppettid", "fråga", "information", "helg", "enkelt"],
        "expect_coaching": False,
    },
    {
        "namn": "teknik_support",
        "segments": TEKNIK_SUPPORT,
        "role_map": TEKNIK_SUPPORT_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": True,
        "expect_root_cause_any": ["internet", "instabil", "teknisk", "kabel", "störning", "router", "uppkoppling"],
        "expect_coaching": True,
    },
    {
        "namn": "uppsagningshot",
        "segments": UPPSAGNINGSHOT,
        "role_map": UPPSAGNINGSHOT_ROLE_MAP,
        "expect_escalation": True,
        "expect_negative": True,
        "expect_root_cause_any": ["service", "usel", "missnöj", "uppsäg", "ansvar", "problem", "leverer", "eskalera"],
        "expect_coaching": True,
    },
    # --- Nya scenarier ---
    {
        "namn": "sarkasm",
        "segments": SARKASM,
        "role_map": SARKASM_ROLE_MAP,
        "expect_escalation": False,  # Sarkasm är inte direkt eskalation
        "expect_negative": True,
        "expect_root_cause_any": ["sarkasm", "frustr", "vänt", "kö", "internet", "service", "underförstådd", "missnöj"],
        "expect_coaching": True,
    },
    {
        "namn": "tystnad",
        "segments": TYSTNAD,
        "role_map": TYSTNAD_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": True,
        "expect_root_cause_any": ["tyst", "funder", "problem", "abonnemang", "uppsäg", "frustr", "tveksam"],
        "expect_coaching": True,
    },
    {
        "namn": "multi_issue",
        "segments": MULTI_ISSUE,
        "role_map": MULTI_ISSUE_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": True,
        "expect_root_cause_any": ["faktura", "internet", "paket", "problem", "tjänst", "premium", "beställ", "flera"],
        "expect_coaching": True,
    },
    {
        "namn": "forvirrad_kund",
        "segments": FORVIRRAD_KUND,
        "role_map": FORVIRRAD_KUND_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": False,  # Förvirrad, inte arg
        "expect_root_cause_any": ["autogiro", "betalning", "teknisk", "fel", "faktura", "förvirr", "missförstånd"],
        "expect_coaching": True,
    },
    {
        "namn": "empatiskt_samtal",
        "segments": EMPATISKT_SAMTAL,
        "role_map": EMPATISKT_SAMTAL_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": False,
        "expect_root_cause_any": ["faktura", "tjänst", "beställ", "kredit", "lösenord", "son", "extra"],
        "expect_coaching": False,  # Agent gjorde allt rätt
    },
    {
        "namn": "kort_argt",
        "segments": KORT_ARGT,
        "role_map": KORT_ARGT_ROLE_MAP,
        "expect_escalation": True,
        "expect_negative": True,
        "expect_root_cause_any": ["service", "skräp", "pengar", "fungerar", "uppsäg", "konkurrent", "arg"],
        "expect_coaching": True,
    },
    {
        "namn": "aterkommande_kund",
        "segments": ATERKOMMANDE_KUND,
        "role_map": ATERKOMMANDE_KUND_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": True,
        "expect_root_cause_any": ["internet", "fungerar", "veckor", "fem", "ringt", "prioriter", "löfte", "återring"],
        "expect_coaching": True,
    },
    {
        "namn": "code_switching",
        "segments": CODE_SWITCHING,
        "role_map": CODE_SWITCHING_ROLE_MAP,
        "expect_escalation": False,
        "expect_negative": True,
        "expect_root_cause_any": ["router", "internet", "connection", "reset", "led", "kabel", "firmware", "isp"],
        "expect_coaching": False,
    },
]


# =============================================================================
# Edge case-scenarier (för pipeline-robusthet)
# =============================================================================

EDGE_CASES: list[dict[str, Any]] = [
    {
        "namn": "endast_agent",
        "segments": ENDAST_AGENT,
        "role_map": ENDAST_AGENT_ROLE_MAP,
        "description": "Endast agent talar — kund tyst/avbruten",
    },
    {
        "namn": "endast_kund",
        "segments": ENDAST_KUND,
        "role_map": ENDAST_KUND_ROLE_MAP,
        "description": "Endast kund talar — agent ej ansluten",
    },
    {
        "namn": "okanda_talare",
        "segments": OKANDA_TALARE,
        "role_map": None,  # Ingen role_map
        "description": "Okända talare utan role_map",
    },
]


# =============================================================================
# Sentiment-klassifikationsfall för chat_completion (parametriserade)
# =============================================================================

SENTIMENT_KLASSIFIKATION: list[dict[str, str]] = [
    {"text": "Jag är väldigt nöjd med er utmärkta service!", "expect": "positiv"},
    {"text": "Tack för snabb och professionell hjälp.", "expect": "positiv"},
    {"text": "Det fungerade jättebra, tack!", "expect": "positiv"},
    {"text": "Det här är helt oacceptabelt och jag är arg.", "expect": "negativ"},
    {"text": "Jag har väntat i 45 minuter, det är en katastrof.", "expect": "negativ"},
    {"text": "Ingenting fungerar och ingen hjälper mig.", "expect": "negativ"},
    {"text": "Jag vet inte, det kanske fungerar.", "expect": "neutral"},
    {"text": "Okej, jag förstår.", "expect": "neutral"},
    {"text": "Det var inget större problem.", "expect": "neutral"},
]


# =============================================================================
# Texter med PII (för domäntester — testar om modellen identifierar PII)
# =============================================================================

PII_TEST_TEXTER: list[dict[str, str]] = [
    {
        "text": "Kundens personnummer är 19900101-1234 och hen bor på Storgatan 12, Stockholm.",
        "expect_pii_type": "personnummer",
        "expect_contains": ["personnummer", "19900101"],
    },
    {
        "text": "Kundens e-post är anna.svensson@example.com och telefon är 070-1234567.",
        "expect_pii_type": "kontaktuppgifter",
        "expect_contains": ["e-post", "email", "telefon"],
    },
    {
        "text": "Kundens kortnummer är 4111 1111 1111 1111 och CVV är 123.",
        "expect_pii_type": "kortnummer",
        "expect_contains": ["kort", "4111"],
    },
]
