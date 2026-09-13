import type { Direction } from "../types";

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
  signed = false
): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(digits)}%`;
}

export function formatPctPoint(
  value: number | null | undefined,
  digits = 2,
  signed = true
): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(digits)} p.p.`;
}

// Research CSV improvement fields are already expressed in percent units.
export function formatPercentValue(
  value: number | null | undefined,
  digits = 2,
  signed = true
): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}%`;
}

export function formatPrice(ticker: string, value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const currency = ticker.toUpperCase().endsWith(".SA") ? "BRL" : "USD";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);
}

export function formatInteger(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export function directionLabel(direction: Direction): string {
  if (direction === "up") return "Up";
  if (direction === "down") return "Down";
  return "Flat";
}

export function safeNumber(record: Record<string, unknown> | undefined, key: string): number | null {
  const value = record?.[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function safeString(record: Record<string, unknown> | undefined, key: string): string {
  const value = record?.[key];
  return value == null ? "" : String(value);
}
