import * as vscode from "vscode";

import { authorLabel, matchMeter, truncate, yearLabel } from "../display";
import { Candidate } from "../types";

/** Rejection reasons offered in each candidate's dropdown. */
const REJECTION_REASONS: readonly string[] = [
  "too generic",
  "not relevant",
  "wrong time period",
  "wrong subfield",
  "already cited",
  "low quality",
];

const EVIDENCE_MAX_CHARS = 300;
const TITLE_MAX_CHARS = 50;

export type EvidenceActionType =
  | "insert"
  | "thumbsUp"
  | "thumbsDown"
  | "copyBibtex"
  | "openUrl"
  | "reject";

export interface EvidenceAction {
  type: EvidenceActionType;
  candidate: Candidate;
  /** Present only for `reject` actions. */
  reason?: string;
}

export type EvidenceActionHandler = (
  action: EvidenceAction,
) => void | Promise<void>;

/** Messages posted from the webview script back to the extension host. */
interface WebviewMessage {
  kind: EvidenceActionType;
  index: number;
  reason?: string;
}

/**
 * A reusable Webview panel that renders citation candidates with their evidence
 * sentences, match meter, and per-candidate feedback controls (insert,
 * thumbs-up/down, copy BibTeX, open online, reject + reason).
 *
 * The panel is pure UI: it resolves a clicked candidate by index and forwards
 * an {@link EvidenceAction} to the host-supplied handler, which owns insertion
 * and feedback logging. A single instance is reused across runs.
 */
export class EvidencePanel {
  public static current: EvidencePanel | undefined;
  private static readonly viewType = "missingCitations.evidence";

  private readonly panel: vscode.WebviewPanel;
  private readonly disposables: vscode.Disposable[] = [];
  private candidates: Candidate[] = [];
  private handler: EvidenceActionHandler = () => undefined;

  static show(
    query: string,
    candidates: Candidate[],
    handler: EvidenceActionHandler,
  ): void {
    const column = vscode.ViewColumn.Beside;

    if (EvidencePanel.current) {
      EvidencePanel.current.update(query, candidates, handler);
      EvidencePanel.current.panel.reveal(column, true);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      EvidencePanel.viewType,
      "Citation Evidence",
      { viewColumn: column, preserveFocus: true },
      { enableScripts: true, retainContextWhenHidden: true },
    );
    EvidencePanel.current = new EvidencePanel(panel);
    EvidencePanel.current.update(query, candidates, handler);
  }

  private constructor(panel: vscode.WebviewPanel) {
    this.panel = panel;
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (message: WebviewMessage) => this.onMessage(message),
      null,
      this.disposables,
    );
  }

  private update(
    query: string,
    candidates: Candidate[],
    handler: EvidenceActionHandler,
  ): void {
    this.candidates = candidates;
    this.handler = handler;
    this.panel.title = `Citations: ${truncate(query, TITLE_MAX_CHARS)}`;
    this.panel.webview.html = this.render(query);
  }

  private onMessage(message: WebviewMessage): void {
    const candidate = this.candidates[message.index];
    if (!candidate) {
      return;
    }
    void this.handler({
      type: message.kind,
      candidate,
      reason: message.reason,
    });
  }

  private dispose(): void {
    EvidencePanel.current = undefined;
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
    this.panel.dispose();
  }

  // ── HTML rendering ───────────────────────────────────────────────

  private render(query: string): string {
    const nonce = makeNonce();
    const cspSource = this.panel.webview.cspSource;
    const cards = this.candidates
      .map((candidate, index) => renderCard(candidate, index))
      .join("\n");

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Citation Evidence</title>
  <style>${STYLES}</style>
</head>
<body>
  <header class="header">
    <h1>Citation candidates</h1>
    <p class="query">for “${escapeHtml(truncate(query, 200))}”</p>
  </header>
  <main class="cards">
    ${cards || '<p class="empty">No candidates returned.</p>'}
  </main>
  <script nonce="${nonce}">${SCRIPT}</script>
</body>
</html>`;
  }
}

function renderCard(candidate: Candidate, index: number): string {
  const meter = matchMeter(candidate);
  const meterHtml = meter
    ? `<span class="meter" title="best evidence cosine similarity">${meter.dots} ${meter.pct}%</span>`
    : "";
  const venueHtml = candidate.venue
    ? `<span class="venue">${escapeHtml(candidate.venue)}</span>`
    : "";

  const evidenceHtml = candidate.evidence
    .map((evidence) => {
      const year = evidence.citing_year
        ? `<span class="cite-year">cited ${evidence.citing_year}</span>`
        : "";
      return `<li><blockquote>${escapeHtml(
        truncate(evidence.sentence, EVIDENCE_MAX_CHARS),
      )}</blockquote>${year}</li>`;
    })
    .join("\n");

  const reasonOptions = REJECTION_REASONS.map(
    (reason) => `<option value="${escapeHtml(reason)}">${escapeHtml(reason)}</option>`,
  ).join("");

  // NOTE (Phase 9): a role-distribution row (background / method_use / …) will
  // slot in here once the API surfaces citation_role on evidence.

  return `<article class="card" data-index="${index}">
  <div class="card-head">
    <h2 class="title">${escapeHtml(candidate.title)}</h2>
    <div class="byline">
      <span class="authors">${escapeHtml(authorLabel(candidate))}</span>
      <span class="year">(${escapeHtml(yearLabel(candidate))})</span>
      ${venueHtml}
      ${meterHtml}
    </div>
    <code class="key">${escapeHtml(candidate.citation_key)}</code>
  </div>
  <ul class="evidence">${evidenceHtml}</ul>
  <div class="actions">
    <button class="primary" data-action="insert" data-index="${index}">Insert citation</button>
    <button data-action="thumbsUp" data-index="${index}" title="Useful">👍</button>
    <button data-action="thumbsDown" data-index="${index}" title="Not useful">👎</button>
    <button data-action="copyBibtex" data-index="${index}">Copy BibTeX</button>
    <button data-action="openUrl" data-index="${index}">Search online</button>
    <span class="reject-group">
      <select class="reason" id="reason-${index}" aria-label="Rejection reason">${reasonOptions}</select>
      <button class="danger" data-action="reject" data-index="${index}">Reject</button>
    </span>
  </div>
</article>`;
}

const STYLES = `
  :root { color-scheme: light dark; }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 0 16px 24px;
  }
  .header { position: sticky; top: 0; background: var(--vscode-editor-background); padding: 12px 0 8px; }
  .header h1 { font-size: 1.1em; margin: 0; }
  .query { color: var(--vscode-descriptionForeground); margin: 4px 0 0; }
  .empty { color: var(--vscode-descriptionForeground); }
  .card {
    border: 1px solid var(--vscode-panel-border);
    border-radius: 6px;
    padding: 12px 14px;
    margin: 12px 0;
    background: var(--vscode-editorWidget-background);
  }
  .card.dimmed { opacity: 0.5; }
  .title { font-size: 1.02em; margin: 0 0 4px; }
  .byline { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; color: var(--vscode-descriptionForeground); font-size: 0.9em; }
  .meter { font-family: var(--vscode-editor-font-family); letter-spacing: 1px; }
  .key { display: inline-block; margin-top: 6px; font-size: 0.85em; color: var(--vscode-textPreformat-foreground); }
  .evidence { list-style: none; padding: 0; margin: 10px 0; }
  .evidence li { margin: 6px 0; }
  .evidence blockquote {
    margin: 0; padding: 6px 10px;
    border-left: 3px solid var(--vscode-textBlockQuote-border);
    background: var(--vscode-textBlockQuote-background);
    font-style: italic;
  }
  .cite-year { font-size: 0.8em; color: var(--vscode-descriptionForeground); }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 8px; }
  .reject-group { display: inline-flex; gap: 6px; margin-left: auto; align-items: center; }
  button {
    color: var(--vscode-button-secondaryForeground);
    background: var(--vscode-button-secondaryBackground);
    border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;
  }
  button:hover { background: var(--vscode-button-secondaryHoverBackground); }
  button.primary { color: var(--vscode-button-foreground); background: var(--vscode-button-background); }
  button.primary:hover { background: var(--vscode-button-hoverBackground); }
  button.danger { color: var(--vscode-errorForeground); }
  button.active { outline: 2px solid var(--vscode-focusBorder); }
  select.reason {
    color: var(--vscode-dropdown-foreground);
    background: var(--vscode-dropdown-background);
    border: 1px solid var(--vscode-dropdown-border);
    border-radius: 4px; padding: 4px;
  }
`;

// Webview-side script. Forwards clicks to the host and gives lightweight
// optimistic feedback (highlight thumbs, dim a rejected card).
const SCRIPT = `
  const vscode = acquireVsCodeApi();
  document.querySelectorAll('button[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const index = Number(btn.getAttribute('data-index'));
      const kind = btn.getAttribute('data-action');
      const card = btn.closest('.card');
      if (kind === 'reject') {
        const select = document.getElementById('reason-' + index);
        const reason = select ? select.value : '';
        if (card) { card.classList.add('dimmed'); }
        vscode.postMessage({ kind, index, reason });
        return;
      }
      if (kind === 'thumbsUp' || kind === 'thumbsDown') {
        if (card) {
          card.querySelectorAll('button[data-action="thumbsUp"], button[data-action="thumbsDown"]')
            .forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
        }
      }
      vscode.postMessage({ kind, index });
    });
  });
`;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function makeNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i += 1) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}
