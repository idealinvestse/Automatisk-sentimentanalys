import json
import structlog
from pathlib import Path
from src.observability.metrics import ...  # vi använder redan observability

logger = structlog.get_logger()

def gdpr_safe_import(source_path: str, target_corpus: str):
    """GDPR-safe import med PII-redaction + logging."""
    logger.info("Starting GDPR-safe data import", source=source_path)
    with tracer.start_as_current_span("data_import"):
        # Exempel: läs anonymiserad JSONL, redigera PII, spara
        data = []
        for line in Path(source_path).read_text().splitlines():
            record = json.loads(line)
            # PII-redaction (använd befintlig funktion från pipeline)
            record = redact_pii(record)
            data.append(record)
        
        Path(target_corpus).write_text(json.dumps(data, ensure_ascii=False))
        logger.info("Data import completed", records=len(data), target=target_corpus)
        return len(data)