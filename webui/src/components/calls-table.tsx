"use client";

import * as React from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUpDown, AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { RiskBadge, QaBadge, SentimentBadge } from "@/components/status-badges";
import type { CallRow } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

function sortableHeader(label: string) {
  return function Header({ column }: { column: { toggleSorting: (desc?: boolean) => void; getIsSorted: () => false | "asc" | "desc" } }) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className="-ml-3 h-8"
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
      >
        {label}
        <ArrowUpDown className="size-3.5" />
      </Button>
    );
  };
}

const columns: ColumnDef<CallRow>[] = [
  {
    accessorKey: "title",
    header: sortableHeader("Samtal"),
    cell: ({ row }) => (
      <div className="flex min-w-0 flex-col">
        <span className="truncate font-medium">{row.original.title}</span>
        <span className="text-xs text-muted-foreground">{row.original.callId}</span>
      </div>
    ),
  },
  {
    accessorKey: "agent",
    header: sortableHeader("Agent"),
  },
  {
    accessorKey: "category",
    header: sortableHeader("Kategori"),
    cell: ({ row }) => <span className="capitalize">{row.original.category}</span>,
  },
  {
    accessorKey: "sentiment",
    header: sortableHeader("Sentiment"),
    cell: ({ row }) => <SentimentBadge value={row.original.sentiment} />,
  },
  {
    accessorKey: "riskLevel",
    header: sortableHeader("Risk"),
    cell: ({ row }) => (
      <div className="flex items-center gap-1.5">
        <RiskBadge value={row.original.riskLevel} />
        {row.original.alertCount > 0 ? (
          <span className="inline-flex items-center gap-0.5 text-xs text-warning-text">
            <AlertTriangle className="size-3.5" />
            {row.original.alertCount}
          </span>
        ) : null}
      </div>
    ),
  },
  {
    accessorKey: "qaScore",
    header: sortableHeader("QA"),
    cell: ({ row }) => <QaBadge passed={row.original.qaPassed} score={row.original.qaScore} />,
  },
];

export function CallsTable({
  data,
  onSelectCall,
}: {
  data: CallRow[];
  onSelectCall?: (callId: string) => void;
}) {
  const [sorting, setSorting] = React.useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50 text-left text-muted-foreground">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id} className="whitespace-nowrap px-4 py-2 font-medium">
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4 py-8 text-center text-muted-foreground">
                Inga samtal matchar filtret.
              </td>
            </tr>
          ) : (
            table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => onSelectCall?.(row.original.callId)}
                className={cn(
                  "border-t border-border transition-colors",
                  onSelectCall && "cursor-pointer hover:bg-accent/50",
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-2.5 align-middle">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
