// Shared API DTOs and feedback contracts used by the command, the webview
// panel, and the feedback client. Mirrors the backend Pydantic schemas in
// src/api/schemas.py.

export interface Evidence {
  sentence: string;
  citing_year: number | null;
  similarity: number;
}

export interface Candidate {
  paper_id: number;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  citation_key: string;
  score: number;
  evidence: Evidence[];
  bibtex: string;
  // Phase 9: ID of the persisted recommendation_results row. Null when backend
  // logging failed; feedback then cannot be attributed to this candidate.
  result_id: string | null;
}

export interface RecommendResponse {
  candidates: Candidate[];
  // Phase 9: ID of the persisted recommendation_events row. Null when logging
  // failed.
  event_id: string | null;
}

// Mirror of api.schemas.FeedbackType.
export type FeedbackType =
  | "accepted"
  | "rejected"
  | "thumbs_up"
  | "thumbs_down"
  | "copied_bibtex"
  | "opened_url";

export interface FeedbackPayload {
  event_id: string;
  result_id?: string | null;
  feedback_type: FeedbackType;
  feedback_value?: number | null;
  reason?: string | null;
}
