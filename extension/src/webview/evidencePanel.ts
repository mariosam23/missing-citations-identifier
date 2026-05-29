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
    <p class="query">${escapeHtml(truncate(query, 200))}</p>
    <span class="count">${this.candidates.length} result${this.candidates.length !== 1 ? "s" : ""}</span>
  </header>
  <main class="cards">
    ${cards || '<p class="empty">No candidates returned.</p>'}
  </main>
  <div id="toast" class="toast"></div>
  <script nonce="${nonce}">${SCRIPT}</script>
</body>
</html>`;
  }
}

function renderCard(candidate: Candidate, index: number): string {
  const meter = matchMeter(candidate);
  const rank = index + 1;

  // Slim horizontal bar for match strength
  let meterHtml = "";
  if (meter) {
    meterHtml = `<div class="meter" title="Semantic match ${meter.pct}%">
      <div class="meter-track"><div class="meter-fill" style="width:${meter.pct}%"></div></div>
      <span class="meter-pct">${meter.pct}%</span>
    </div>`;
  }

  const venueHtml = candidate.venue
    ? ` &middot; ${escapeHtml(candidate.venue)}`
    : "";

  const evidenceHtml = candidate.evidence
    .map((ev) => {
      const year = ev.citing_year ? `<span class="ev-year">${ev.citing_year}</span>` : "";
      return `<li>
        <blockquote>${escapeHtml(truncate(ev.sentence, EVIDENCE_MAX_CHARS))}</blockquote>
        <span class="ev-meta">${year}</span>
      </li>`;
    })
    .join("\n");

  const reasonOptions = REJECTION_REASONS.map(
    (r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`,
  ).join("");

  // NOTE (Phase 9): a role-distribution row (background / method_use / …) will
  // slot in here once the API surfaces citation_role on evidence.

  return `<article class="card" data-index="${index}">
  <div class="card-head">
    <span class="rank">${rank}</span>
    <div class="card-meta">
      <h2 class="title">${escapeHtml(candidate.title)}</h2>
      <p class="byline">${escapeHtml(authorLabel(candidate))}, ${escapeHtml(yearLabel(candidate))}${venueHtml}</p>
      <code class="key">${escapeHtml(candidate.citation_key)}</code>
    </div>
  </div>
  ${meterHtml}
  <ul class="evidence">${evidenceHtml}</ul>
  <div class="actions">
    <button class="btn primary with-icon" data-action="insert" data-index="${index}">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M14 7v1H8v6H7V8H1V7h6V1h1v6h6z"/></svg>
      Insert
    </button>
    <button class="btn with-icon" data-action="copyBibtex" data-index="${index}" title="Copy BibTeX">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4 4v1h7v7h1V4H4zm-2 2v9h9V6H2zm1 1h7v7H3V7z"/></svg>
      Copy
    </button>
    <button class="btn with-icon" data-action="openUrl" data-index="${index}" title="Search online">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M10 2v1h2.3L5.5 9.8l.7.7L13 3.7V6h1V2h-4zM2 4v10h10V9h-1v4H3V5h4V4H2z"/></svg>
      Open
    </button>
    <span class="spacer"></span>
    <div class="feedback-group">
      <button class="btn icon-btn" data-action="thumbsUp" data-index="${index}" title="Useful">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2 6v8h2V6H2zm10 0h-3.5l1.2-4.1c.1-.4-.2-.9-.7-.9H8L4.5 5.5V14h7c.5 0 .9-.4.9-.9V6c0-.5-.4-.9-.9-.9z"/></svg>
      </button>
      <button class="btn icon-btn" data-action="thumbsDown" data-index="${index}" title="Not useful">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2 10V2h2v8H2zm10 0h-3.5l1.2 4.1c.1.4-.2.9-.7.9H8l-3.5-4.5V2h7c.5 0 .9.4.9.9v7.1c0 .5-.4.9-.9.9z"/></svg>
      </button>
      <span class="reject-group" style="display: none;">
        <select class="reason" id="reason-${index}" aria-label="Rejection reason">${reasonOptions}</select>
        <button class="btn danger" data-action="reject" data-index="${index}">Submit</button>
      </span>
    </div>
  </div>
</article>`;
}

const STYLES = `
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }

  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    padding: 0 16px 28px;
    margin: 0;
    line-height: 1.55;
  }

  /* ── Header ─────────────────────────────────── */
  .header {
    position: sticky; top: 0; z-index: 10;
    background: var(--vscode-editor-background);
    padding: 14px 0 10px;
    border-bottom: 1px solid var(--vscode-panel-border);
  }
  .header h1 {
    font-size: 1.08em;
    font-weight: 600;
    margin: 0 0 2px;
  }
  .query {
    color: var(--vscode-descriptionForeground);
    margin: 0;
    font-size: 0.9em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .count {
    display: inline-block;
    margin-top: 6px;
    font-size: 0.78em;
    color: var(--vscode-descriptionForeground);
    opacity: 0.8;
  }
  .empty { color: var(--vscode-descriptionForeground); padding: 32px 0; }

  /* ── Cards ──────────────────────────────────── */
  .cards {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding-top: 8px;
  }
  .card {
    padding: 14px 16px;
    border-radius: 6px;
    background: var(--vscode-editorWidget-background);
    border: 1px solid transparent;
    transition: border-color 0.15s, opacity 0.3s;
  }
  .card:hover { border-color: var(--vscode-panel-border); }
  .card.dimmed { opacity: 0.35; pointer-events: none; }
  .card.inserted { border-color: var(--vscode-testing-iconPassed, #388e3c); }

  /* ── Card head ──────────────────────────────── */
  .card-head {
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }
  .rank {
    flex-shrink: 0;
    width: 22px; height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 600;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    margin-top: 2px;
  }
  .card-meta { min-width: 0; }
  .title {
    font-size: 0.98em;
    font-weight: 600;
    margin: 0 0 2px;
    line-height: 1.35;
  }
  .byline {
    color: var(--vscode-descriptionForeground);
    font-size: 0.85em;
    margin: 0;
  }
  .key {
    display: inline-block;
    margin-top: 4px;
    font-size: 0.8em;
    color: var(--vscode-textPreformat-foreground);
    opacity: 0.85;
  }

  /* ── Meter ──────────────────────────────────── */
  .meter {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 10px 0 2px 34px;
  }
  .meter-track {
    flex: 1;
    height: 3px;
    border-radius: 2px;
    background: var(--vscode-panel-border);
    overflow: hidden;
  }
  .meter-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--vscode-textLink-foreground, #007acc);
    transition: width 0.4s ease;
  }
  .meter-pct {
    font-size: 0.75em;
    color: var(--vscode-descriptionForeground);
    min-width: 28px;
    text-align: right;
  }

  /* ── Evidence ───────────────────────────────── */
  .evidence {
    list-style: none;
    padding: 0;
    margin: 10px 0 0;
  }
  .evidence li { margin: 6px 0; }
  .evidence blockquote {
    margin: 0;
    padding: 6px 10px;
    border-left: 2px solid var(--vscode-textBlockQuote-border);
    background: var(--vscode-textBlockQuote-background);
    font-style: italic;
    font-size: 0.9em;
    border-radius: 0 4px 4px 0;
  }
  .ev-meta {
    display: inline-flex;
    gap: 6px;
    margin-left: 14px;
    font-size: 0.78em;
    color: var(--vscode-descriptionForeground);
  }

  /* ── Actions ────────────────────────────────── */
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--vscode-panel-border);
  }
  .spacer { flex: 1; }

  .btn {
    font-family: inherit;
    font-size: 0.82em;
    padding: 4px 10px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    color: var(--vscode-button-secondaryForeground);
    background: var(--vscode-button-secondaryBackground);
    transition: background 0.12s;
  }
  .btn.with-icon {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn:hover { background: var(--vscode-button-secondaryHoverBackground); }
  .btn:active { opacity: 0.8; }
  .btn.primary {
    color: var(--vscode-button-foreground);
    background: var(--vscode-button-background);
  }
  .btn.primary:hover { background: var(--vscode-button-hoverBackground); }
  .btn.danger {
    color: var(--vscode-errorForeground);
    background: var(--vscode-button-secondaryBackground);
  }
  .icon-btn {
    width: 26px;
    height: 26px;
    padding: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    color: var(--vscode-descriptionForeground);
  }
  .icon-btn:hover { background: rgba(128,128,128,0.12); }
  .icon-btn.active-up {
    color: var(--vscode-testing-iconPassed, #388e3c);
    background: rgba(56,142,60,0.1);
  }
  .icon-btn.active-down {
    color: var(--vscode-errorForeground, #d32f2f);
    background: rgba(211,47,47,0.1);
  }

  .feedback-group {
    display: inline-flex;
    gap: 4px;
    align-items: center;
  }
  .reject-group {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    margin-left: 6px;
    padding-left: 10px;
    border-left: 1px solid var(--vscode-panel-border);
  }
  .reason {
    font-family: inherit;
    font-size: 0.8em;
    color: var(--vscode-dropdown-foreground);
    background: var(--vscode-dropdown-background);
    border: 1px solid var(--vscode-dropdown-border);
    border-radius: 4px;
    padding: 3px 6px;
  }

  /* ── Toast ──────────────────────────────────── */
  .toast {
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%) translateY(40px);
    padding: 6px 16px;
    border-radius: 4px;
    font-size: 0.85em;
    background: var(--vscode-editorWidget-background);
    color: var(--vscode-foreground);
    border: 1px solid var(--vscode-panel-border);
    opacity: 0;
    pointer-events: none;
    transition: transform 0.2s ease, opacity 0.2s ease;
    z-index: 100;
  }
  .toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
`;

const SCRIPT = `
  const vscode = acquireVsCodeApi();

  function toast(msg) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 1800);
  }

  document.querySelectorAll('button[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const index = Number(btn.getAttribute('data-index'));
      const kind = btn.getAttribute('data-action');
      const card = btn.closest('.card');

      if (kind === 'reject') {
        const sel = document.getElementById('reason-' + index);
        const reason = sel ? sel.value : '';
        if (card) card.classList.add('dimmed');
        toast('Rejected');
        vscode.postMessage({ kind, index, reason });
        return;
      }

      if (kind === 'thumbsUp' || kind === 'thumbsDown') {
        if (card) {
          card.querySelectorAll('.icon-btn').forEach(b => {
            b.classList.remove('active-up', 'active-down');
          });
          btn.classList.add(kind === 'thumbsUp' ? 'active-up' : 'active-down');
          
          const rejectGroup = card.querySelector('.reject-group');
          if (rejectGroup) {
            rejectGroup.style.display = kind === 'thumbsDown' ? 'inline-flex' : 'none';
          }
        }
      }

      if (kind === 'insert' && card) {
        card.classList.add('inserted');
        toast('Inserted');
      }

      if (kind === 'copyBibtex') toast('Copied');

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
