// Pure presentation helpers shared by the QuickPick (recommendCommand) and the
// Webview (evidencePanel) renderers. No VS Code API imports — keep this module
// trivially unit-testable and reusable from the webview's host side.

import { Candidate, Evidence } from "./types";

export const METER_SEGMENTS = 5;

/** The strongest supporting context (highest cosine similarity), if any. */
export function bestEvidence(candidate: Candidate): Evidence | undefined {
  if (candidate.evidence.length === 0) {
    return undefined;
  }
  return candidate.evidence.reduce((best, current) =>
    current.similarity > best.similarity ? current : best,
  );
}

export interface MatchMeter {
  dots: string;
  pct: number;
}

/**
 * A five-segment dot meter from the best evidence cosine similarity — an honest
 * semantic-match signal in [0, 1]. The backend `score` mixes in a corroboration
 * bonus and is unbounded, so it is used only for ordering, never shown as a
 * percentage.
 */
export function matchMeter(candidate: Candidate): MatchMeter | undefined {
  const evidence = bestEvidence(candidate);
  if (!evidence) {
    return undefined;
  }
  const pct = Math.max(0, Math.min(100, Math.round(evidence.similarity * 100)));
  const filled = Math.round((pct / 100) * METER_SEGMENTS);
  const dots = "●".repeat(filled) + "○".repeat(METER_SEGMENTS - filled);
  return { dots, pct };
}

export function authorLabel(candidate: Candidate): string {
  const surname = firstAuthorSurname(candidate.authors[0]);
  if (!surname) {
    return "Unknown";
  }
  return candidate.authors.length > 1 ? `${surname} et al.` : surname;
}

export function yearLabel(candidate: Candidate): string {
  return candidate.year !== null ? String(candidate.year) : "n.d.";
}

/** Surname from "Last, First" or "First Last"; `null` if unusable. */
export function firstAuthorSurname(raw: string | undefined): string | null {
  if (!raw) {
    return null;
  }
  const name = raw.trim();
  if (!name) {
    return null;
  }
  if (name.includes(",")) {
    return name.split(",", 1)[0].trim();
  }
  const tokens = name.split(/\s+/);
  return tokens[tokens.length - 1];
}

export function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value;
  }
  return `${value.slice(0, max - 1).trimEnd()}…`;
}
