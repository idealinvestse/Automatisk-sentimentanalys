# Coding Guidelines for Windsurf LLM Agents

This document provides comprehensive guidelines for LLM coding agents to deliver high-quality, maintainable, and efficient code while following best practices for collaboration with human developers.

## Core Principles

### 1. Code Quality & Maintainability
- **Write self-documenting code** with clear variable/function names
- **Follow language-specific conventions** (PEP 8 for Python, etc.)
- **Keep functions focused** - single responsibility principle
- **Use type hints** where supported (Python, TypeScript, etc.)
- **Add docstrings/comments** for complex logic, not obvious code
- **Prefer composition over inheritance** when designing classes

### 2. Error Handling & Robustness
- **Always handle expected errors** gracefully with try/catch blocks
- **Use specific exception types** rather than broad catches
- **Log errors with context** (file paths, input values, stack traces)
- **Validate inputs** at function boundaries
- **Provide meaningful error messages** for users
- **Fail fast** - detect problems early rather than propagating bad state

### 3. Performance & Efficiency
- **Avoid premature optimization** - write clear code first
- **Use appropriate data structures** (sets for membership, dicts for lookups)
- **Minimize I/O operations** - batch reads/writes when possible
- **Cache expensive computations** when results are reused
- **Use generators/iterators** for large datasets
- **Profile before optimizing** - measure actual bottlenecks

## File & Project Structure

### 4. Organization
- **Group related functionality** in modules/packages
- **Use clear directory structure** (`src/`, `tests/`, `docs/`, `config/`)
- **Separate concerns** - business logic, data access, presentation
- **Keep configuration external** (environment variables, config files)
- **Use consistent naming** across the project

### 5. Dependencies & Requirements
- **Pin dependency versions** in requirements files
- **Separate dev/prod dependencies** (`requirements-dev.txt`, `requirements.txt`)
- **Document system dependencies** (ffmpeg, databases, etc.)
- **Use virtual environments** for Python projects
- **Keep dependencies minimal** - avoid unnecessary packages

## Code Implementation

### 6. Functions & Methods
```python
# Good: Clear name, type hints, docstring, error handling
def transcribe_audio(
    audio_path: str, 
    model: str = "whisper-large", 
    language: str = "sv"
) -> Dict[str, Any]:
    """Transcribe audio file using specified model.
    
    Args:
        audio_path: Path to audio file
        model: Model name to use for transcription
        language: Language code for transcription
        
    Returns:
        Dictionary with transcript and metadata
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If model is not supported
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Implementation...
```

### 7. Classes & Data Models
```python
# Good: Use dataclasses/Pydantic for data models
from dataclasses import dataclass
from typing import Optional

@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio."""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    
    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end - self.start
```

### 8. Configuration & Settings
```python
# Good: Use environment variables with defaults
import os
from typing import Optional

class Config:
    """Application configuration."""
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Model settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "whisper-large")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./cache")
    
    # Security
    API_KEY: Optional[str] = os.getenv("API_KEY")
```

## API & Interface Design

### 9. REST API Guidelines
- **Use proper HTTP methods** (GET, POST, PUT, DELETE)
- **Return consistent response formats** with status codes
- **Include request/response validation** (Pydantic models)
- **Provide clear error messages** in API responses
- **Use pagination** for large result sets
- **Version your APIs** (`/v1/`, `/v2/`)

```python
# Good: Clear request/response models
class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file")
    model: str = Field("whisper-large", description="Model to use")
    language: str = Field("sv", description="Language code")
    
class TranscribeResponse(BaseModel):
    transcript: Dict[str, Any]
    processing_time: float
    timestamp: str
```

### 10. CLI Design
- **Use argument parsing libraries** (argparse, typer, click)
- **Provide help text** for all commands and options
- **Support batch operations** when applicable
- **Show progress** for long-running operations
- **Allow output format selection** (JSON, CSV, plain text)

## Testing & Quality Assurance

### 11. Testing Strategy
- **Write unit tests** for core business logic
- **Test error conditions** and edge cases
- **Use fixtures** for test data and setup
- **Mock external dependencies** (APIs, file system)
- **Test CLI commands** with different argument combinations
- **Include integration tests** for critical workflows

### 12. Logging & Monitoring
```python
import logging

# Good: Structured logging with context
logger = logging.getLogger(__name__)

def process_audio(audio_path: str) -> Dict[str, Any]:
    logger.info("Processing audio file", extra={
        "audio_path": audio_path,
        "file_size": os.path.getsize(audio_path)
    })
    
    try:
        result = transcribe(audio_path)
        logger.info("Audio processing completed", extra={
            "audio_path": audio_path,
            "segments": len(result.get("segments", [])),
            "duration": result.get("duration", 0)
        })
        return result
    except Exception as e:
        logger.error("Audio processing failed", extra={
            "audio_path": audio_path,
            "error": str(e)
        }, exc_info=True)
        raise
```

## Security & Best Practices

### 13. Security Considerations
- **Validate all inputs** - never trust user data
- **Use parameterized queries** for database operations
- **Store secrets securely** (environment variables, key vaults)
- **Implement rate limiting** for APIs
- **Log security events** (failed authentication, suspicious requests)
- **Keep dependencies updated** to patch vulnerabilities

### 14. Resource Management
```python
# Good: Use context managers for resource cleanup
from contextlib import contextmanager

@contextmanager
def audio_processor(model_name: str):
    """Context manager for audio processing resources."""
    processor = load_model(model_name)
    try:
        yield processor
    finally:
        cleanup_model(processor)

# Usage
with audio_processor("whisper-large") as processor:
    result = processor.transcribe(audio_path)
```

## Documentation & Communication

### 15. Documentation Standards
- **Write clear README files** with installation and usage instructions
- **Document API endpoints** with request/response examples
- **Include code examples** for common use cases
- **Keep documentation up-to-date** with code changes
- **Use consistent formatting** (Markdown, proper headings)

### 16. Version Control & Collaboration
- **Write descriptive commit messages** explaining what and why
- **Use feature branches** for new development
- **Keep commits focused** - one logical change per commit
- **Include tests** in the same commit as the feature
- **Update documentation** when changing interfaces

## Language-Specific Guidelines

### 17. Python Best Practices
```python
# Good: Modern Python patterns
from __future__ import annotations  # Enable forward references
from pathlib import Path
from typing import Protocol

# Use pathlib for file operations
def read_config(config_path: Path) -> dict[str, Any]:
    """Read configuration from file."""
    return json.loads(config_path.read_text(encoding="utf-8"))

# Use protocols for duck typing
class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> dict[str, Any]: ...

# Use dataclasses for simple data containers
@dataclass(frozen=True)
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int
```

### 18. JavaScript/TypeScript Guidelines
```typescript
// Good: TypeScript with proper types
interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  confidence?: number;
}

class AudioProcessor {
  private readonly model: string;
  
  constructor(model: string = "whisper-large") {
    this.model = model;
  }
  
  async transcribe(audioPath: string): Promise<TranscriptSegment[]> {
    // Implementation with proper error handling
    try {
      const result = await this.processAudio(audioPath);
      return result.segments;
    } catch (error) {
      console.error(`Transcription failed for ${audioPath}:`, error);
      throw new Error(`Failed to transcribe audio: ${error.message}`);
    }
  }
}
```

## Performance & Scalability

### 19. Concurrency & Parallelism
```python
# Good: Use appropriate concurrency patterns
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

async def process_audio_batch(audio_files: list[str]) -> list[dict]:
    """Process multiple audio files concurrently."""
    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, transcribe_audio, file_path)
            for file_path in audio_files
        ]
        return await asyncio.gather(*tasks)

# Use ProcessPoolExecutor for CPU-bound tasks
def process_cpu_intensive_batch(data: list[Any]) -> list[Any]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(cpu_intensive_function, data))
```

### 20. Memory Management
- **Use generators** for processing large datasets
- **Implement pagination** for database queries
- **Clean up resources** explicitly when needed
- **Monitor memory usage** in long-running processes
- **Use streaming** for large file operations

## Deployment & Operations

### 21. Container & Deployment
```dockerfile
# Good: Multi-stage Docker build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 22. Monitoring & Observability
- **Include health check endpoints** (`/health`, `/ready`)
- **Expose metrics** (processing time, error rates, queue sizes)
- **Use structured logging** with correlation IDs
- **Implement graceful shutdown** for long-running processes
- **Monitor resource usage** (CPU, memory, disk)

## Common Anti-Patterns to Avoid

### 23. Code Smells
❌ **Avoid:**
```python
# Bad: Unclear names, no error handling, mixed concerns
def process(data):
    result = []
    for item in data:
        # Complex processing without error handling
        processed = some_complex_operation(item)
        result.append(processed)
        # Side effect: writing to file
        with open("output.txt", "a") as f:
            f.write(str(processed))
    return result
```

✅ **Prefer:**
```python
# Good: Clear separation of concerns
def process_items(items: list[InputItem]) -> list[ProcessedItem]:
    """Process a list of items, handling errors gracefully."""
    results = []
    for item in items:
        try:
            processed = transform_item(item)
            results.append(processed)
        except ProcessingError as e:
            logger.warning(f"Failed to process item {item.id}: {e}")
            continue
    return results

def save_results(results: list[ProcessedItem], output_path: Path) -> None:
    """Save processed results to file."""
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result.to_json()}\n")
```

### 24. Performance Anti-Patterns
❌ **Avoid:**
- Loading entire large files into memory
- N+1 query problems in database operations
- Synchronous operations in async contexts
- Recreating expensive objects in loops
- Not using appropriate data structures

## Summary Checklist

When implementing new features, ensure:

- [ ] **Code is readable** and follows naming conventions
- [ ] **Error handling** is comprehensive and informative
- [ ] **Type hints** are used where applicable
- [ ] **Tests** cover main functionality and edge cases
- [ ] **Documentation** is updated for new features
- [ ] **Dependencies** are properly managed
- [ ] **Security** considerations are addressed
- [ ] **Performance** implications are considered
- [ ] **Logging** provides useful debugging information
- [ ] **Configuration** is externalized and documented

---

*This document should be regularly updated as new patterns emerge and best practices evolve.*
