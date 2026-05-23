import { FeedbackPayload } from "./types";

const FEEDBACK_PATH = "/feedback";

/**
 * Sends feedback interactions to the backend `POST /feedback` endpoint.
 *
 * Feedback is telemetry, not a user-facing action: `send` is fire-and-forget
 * and never throws or shows an error popup — a failed POST is logged to the
 * console and dropped, so logging outages never interrupt citing.
 */
export class FeedbackClient {
  constructor(
    private readonly backendUrl: string,
    private readonly timeoutMs: number,
  ) {}

  /** Queue a feedback POST without awaiting it. Errors are swallowed. */
  send(payload: FeedbackPayload): void {
    if (!payload.event_id) {
      // Backend logging failed for this run — there is no event to attribute
      // feedback to. Silently skip.
      return;
    }
    void this.post(payload).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      console.warn(`Missing Citations: feedback POST failed — ${message}`);
    });
  }

  private async post(payload: FeedbackPayload): Promise<void> {
    const url = joinUrl(this.backendUrl, FEEDBACK_PATH);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(`backend returned ${res.status} ${res.statusText}`);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
}

function joinUrl(base: string, path: string): string {
  const trimmedBase = base.replace(/\/+$/, "");
  const trimmedPath = path.startsWith("/") ? path : `/${path}`;
  return `${trimmedBase}${trimmedPath}`;
}
