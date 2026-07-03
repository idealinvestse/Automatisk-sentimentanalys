"use client";

import { Fragment, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

export interface TranscriptTurn {
  speaker: string;
  text: string;
  start: number;
}

/**
 * Virtualized transcript list using @tanstack/react-virtual.
 *
 * Renders only the visible rows (+overscan) regardless of transcript length,
 * so calls with hundreds/thousands of turns stay smooth. For short demo
 * transcripts the virtualizer is a no-op (all rows fit in the viewport).
 */
export function TranscriptView({
  turns,
  className,
}: {
  turns: TranscriptTurn[];
  className?: string;
}) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: turns.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 64, // approx row height; measured rows override this
    overscan: 8,
  });

  if (turns.length === 0) {
    return <p className="text-sm text-muted-foreground">Inget transkript.</p>;
  }

  const items = virtualizer.getVirtualItems();

  return (
    <div
      ref={parentRef}
      className={`max-h-96 overflow-y-auto ${className ?? ""}`}
      aria-label="Transkript"
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        {items.map((virtualItem) => {
          const turn = turns[virtualItem.index];
          if (!turn) return null;
          return (
            <div
              key={virtualItem.key}
              data-index={virtualItem.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualItem.start}px)`,
              }}
              className="flex gap-3 py-1.5"
            >
              <span className="w-10 shrink-0 text-xs text-muted-foreground tabular-nums">
                {turn.start}s
              </span>
              <div className="flex min-w-0 flex-col">
                <span
                  className={
                    turn.speaker === "Agent"
                      ? "text-xs font-medium text-primary"
                      : "text-xs font-medium text-muted-foreground"
                  }
                >
                  {turn.speaker}
                </span>
                <span className="text-sm">{turn.text}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Non-virtualized fallback for very short transcripts (kept for parity). */
export function TranscriptList({
  turns,
  className,
}: {
  turns: TranscriptTurn[];
  className?: string;
}) {
  if (turns.length === 0) {
    return <p className="text-sm text-muted-foreground">Inget transkript.</p>;
  }
  return (
    <div className={`flex max-h-96 flex-col gap-3 overflow-y-auto ${className ?? ""}`}>
      {turns.map((turn, i) => (
        <Fragment key={i}>
          <div className="flex gap-3">
            <span className="w-10 shrink-0 text-xs text-muted-foreground tabular-nums">
              {turn.start}s
            </span>
            <div className="flex min-w-0 flex-col">
              <span
                className={
                  turn.speaker === "Agent"
                    ? "text-xs font-medium text-primary"
                    : "text-xs font-medium text-muted-foreground"
                }
              >
                {turn.speaker}
              </span>
              <span className="text-sm">{turn.text}</span>
            </div>
          </div>
        </Fragment>
      ))}
    </div>
  );
}
