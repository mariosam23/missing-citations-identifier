import * as vscode from "vscode";

import { formatCitation, PaperLike } from "./citationFormatter";
import { BibTeXManager, AppendResult } from "./bibtexManager";
import { FeedbackClient } from "./feedbackClient";
import {
  authorLabel,
  bestEvidence,
  matchMeter,
  truncate,
  yearLabel,
} from "./display";
import {
  Candidate,
  FeedbackPayload,
  FeedbackType,
  RecommendResponse,
  ScanItem,
  ScanResponse,
} from "./types";
import {
  EvidenceAction,
  EvidencePanel,
} from "./webview/evidencePanel";

const CONFIG_SECTION = "missingCitations";
const RECOMMEND_PATH = "/recommend";
const SCAN_PATH = "/scan";

const LATEX_LANGUAGE_IDS = new Set(["latex", "tex"]);

interface CandidatePickItem extends vscode.QuickPickItem {
  candidate: Candidate;
}

interface ScanPickItem extends vscode.QuickPickItem {
  item: ScanItem;
}

/** Shared BibTeX manager instance — serialises .bib writes. */
const bibManager = new BibTeXManager();

export async function recommendCitationsForSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage(
      "Missing Citations: open a document and select a sentence first.",
    );
    return;
  }

  const selectedText = editor.document.getText(editor.selection).trim();
  if (!selectedText) {
    vscode.window.showInformationMessage(
      "Missing Citations: select a sentence to get recommendations.",
    );
    return;
  }

  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  const backendUrl = (config.get<string>("backendUrl") ?? "").trim();
  if (!backendUrl) {
    vscode.window.showErrorMessage(
      "Missing Citations: `missingCitations.backendUrl` is not configured.",
    );
    return;
  }
  const topK = config.get<number>("topK") ?? 10;
  const timeoutMs = config.get<number>("requestTimeoutMs") ?? 15000;
  const uiMode = config.get<string>("uiMode") ?? "webview";
  const languageId = editor.document.languageId;
  const documentPath = vscode.workspace.asRelativePath(editor.document.uri);

  // Capture the selection now — in webview mode the user may move the cursor
  // before clicking "Insert", and we still want to cite the original sentence.
  const selection = editor.selection;

  let response: RecommendResponse;
  try {
    response = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Finding citations…",
        cancellable: true,
      },
      (_progress, token) =>
        fetchRecommendations({
          backendUrl,
          text: selectedText,
          languageId,
          documentPath,
          topK,
          timeoutMs,
          token,
        }),
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    vscode.window.showErrorMessage(`Missing Citations: ${message}`);
    return;
  }

  if (response.candidates.length === 0) {
    vscode.window.showInformationMessage(
      "Missing Citations: no candidate papers returned for this selection.",
    );
    return;
  }

  const feedbackClient = new FeedbackClient(backendUrl, timeoutMs);
  const eventId = response.event_id;

  if (uiMode === "webview") {
    EvidencePanel.show(selectedText, response.candidates, (action) =>
      handleEvidenceAction(action, {
        editor,
        selection,
        languageId,
        feedbackClient,
        eventId,
      }),
    );
    return;
  }

  await runQuickPick({
    editor,
    selection,
    languageId,
    candidates: response.candidates,
    query: selectedText,
    feedbackClient,
    eventId,
  });
}

export async function scanDocumentForMissingCitations(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage(
      "Missing Citations: open a document to scan first.",
    );
    return;
  }

  const documentText = editor.document.getText();
  if (!documentText.trim()) {
    vscode.window.showInformationMessage(
      "Missing Citations: the active document is empty.",
    );
    return;
  }

  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  const backendUrl = (config.get<string>("backendUrl") ?? "").trim();
  if (!backendUrl) {
    vscode.window.showErrorMessage(
      "Missing Citations: `missingCitations.backendUrl` is not configured.",
    );
    return;
  }

  const topK = config.get<number>("topK") ?? 10;
  const timeoutMs = config.get<number>("scanRequestTimeoutMs") ?? 120000;
  const maxSentences = config.get<number>("scanMaxSentences") ?? 250;
  const minConfidence = config.get<number>("scanMinConfidence") ?? 0.55;
  const languageId = editor.document.languageId;
  const documentPath = vscode.workspace.asRelativePath(editor.document.uri);

  let response: ScanResponse;
  try {
    response = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Scanning for missing citations…",
        cancellable: true,
      },
      (_progress, token) =>
        fetchScanResults({
          backendUrl,
          text: documentText,
          languageId,
          documentPath,
          topK,
          timeoutMs,
          maxSentences,
          minConfidence,
          token,
        }),
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    vscode.window.showErrorMessage(`Missing Citations: ${message}`);
    return;
  }

  if (response.items.length === 0) {
    vscode.window.showInformationMessage(
      "Missing Citations: no likely missing citations found.",
    );
    return;
  }

  const chosen = await pickScanItem(response.items);
  if (!chosen) {
    return;
  }
  if (chosen.candidates.length === 0) {
    vscode.window.showInformationMessage(
      "Missing Citations: the selected sentence has no candidate references.",
    );
    return;
  }

  const selection = new vscode.Selection(
    editor.document.positionAt(chosen.start_offset),
    editor.document.positionAt(chosen.end_offset),
  );
  editor.selection = selection;
  editor.revealRange(selection, vscode.TextEditorRevealType.InCenterIfOutsideViewport);

  const feedbackClient = new FeedbackClient(backendUrl, timeoutMs);
  EvidencePanel.show(chosen.text, chosen.candidates, (action) =>
    handleEvidenceAction(action, {
      editor,
      selection,
      languageId,
      feedbackClient,
      eventId: chosen.recommendation_event_id,
    }),
  );
}

// ── Webview action handling ────────────────────────────────────────

interface ActionContext {
  editor: vscode.TextEditor;
  selection: vscode.Selection;
  languageId: string;
  feedbackClient: FeedbackClient;
  eventId: string | null;
}

async function handleEvidenceAction(
  action: EvidenceAction,
  ctx: ActionContext,
): Promise<void> {
  const { candidate } = action;
  switch (action.type) {
    case "insert": {
      const insertedKey = await insertCitation(
        ctx.editor,
        ctx.selection,
        candidate,
        ctx.languageId,
      );
      if (insertedKey === null) {
        return; // insertion failed — warning already shown
      }
      await maybeAppendBibtex(ctx.editor, candidate, insertedKey);
      sendFeedback(ctx, candidate, "accepted");
      break;
    }
    case "thumbsUp":
      sendFeedback(ctx, candidate, "thumbs_up", 1);
      break;
    case "thumbsDown":
      sendFeedback(ctx, candidate, "thumbs_down", -1);
      break;
    case "copyBibtex":
      await vscode.env.clipboard.writeText(candidate.bibtex);
      vscode.window.setStatusBarMessage(
        `$(check) Copied BibTeX for ${candidate.citation_key}`,
        3000,
      );
      sendFeedback(ctx, candidate, "copied_bibtex");
      break;
    case "openUrl":
      await openOnline(candidate);
      sendFeedback(ctx, candidate, "opened_url");
      break;
    case "reject":
      sendFeedback(ctx, candidate, "rejected", undefined, action.reason);
      break;
  }
}

function sendFeedback(
  ctx: ActionContext,
  candidate: Candidate,
  feedbackType: FeedbackType,
  feedbackValue?: number,
  reason?: string,
): void {
  if (!ctx.eventId) {
    return; // backend logging failed — nothing to attribute feedback to
  }
  const payload: FeedbackPayload = {
    event_id: ctx.eventId,
    result_id: candidate.result_id,
    feedback_type: feedbackType,
    feedback_value: feedbackValue ?? null,
    reason: reason ?? null,
  };
  ctx.feedbackClient.send(payload);
}

// ── BibTeX auto-append logic ───────────────────────────────────────

async function maybeAppendBibtex(
  editor: vscode.TextEditor,
  candidate: Candidate,
  insertedKey: string,
): Promise<void> {
  const languageId = editor.document.languageId;
  if (!LATEX_LANGUAGE_IDS.has(languageId)) {
    return;
  }

  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  const autoAppend = config.get<boolean>("autoAppendBibTeX") ?? true;
  if (!autoAppend) {
    return;
  }

  try {
    const bibUri = await bibManager.findOrCreate(editor.document);
    if (!bibUri) {
      vscode.window.showWarningMessage(
        "Missing Citations: no .bib file found and auto-creation is disabled.",
      );
      return;
    }

    const result: AppendResult = await bibManager.appendEntry(
      bibUri,
      candidate.bibtex,
      candidate.citation_key,
    );

    const relativeBib = vscode.workspace.asRelativePath(bibUri);

    if (result.skipped) {
      // Entry already present — nothing to do.
      return;
    }

    // If the key was renamed due to collision, update the \cite{...} we
    // just inserted in the document.
    if (result.wroteKey !== insertedKey) {
      await rewriteInsertedKey(editor, insertedKey, result.wroteKey);
      vscode.window.showInformationMessage(
        `Missing Citations: key renamed to ${result.wroteKey} (collision) — appended to ${relativeBib}`,
      );
    } else {
      const authorName = candidate.authors[0] ?? "Unknown";
      const yearStr = candidate.year !== null ? String(candidate.year) : "n.d.";
      vscode.window.showInformationMessage(
        `Missing Citations: appended ${authorName} (${yearStr}) to ${relativeBib}`,
      );
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    vscode.window.showWarningMessage(
      `Missing Citations: could not append BibTeX — ${msg}. ` +
      `The \\cite{} was inserted; copy the BibTeX entry manually.`,
    );
  }
}

/**
 * Find the `\cite{oldKey}` (or \citep / \citet) that was just inserted
 * and rewrite it to `\cite{newKey}`.
 */
async function rewriteInsertedKey(
  editor: vscode.TextEditor,
  oldKey: string,
  newKey: string,
): Promise<void> {
  const doc = editor.document;
  const fullText = doc.getText();
  // Search from the end (most recently inserted) for the old key.
  const pattern = new RegExp(
    `(\\\\cite[pt]?)\\{${escapeRegex(oldKey)}\\}`,
    "g",
  );
  let lastMatch: RegExpExecArray | null = null;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(fullText)) !== null) {
    lastMatch = m;
  }
  if (!lastMatch) {
    return;
  }
  const start = doc.positionAt(lastMatch.index);
  const end = doc.positionAt(lastMatch.index + lastMatch[0].length);
  await editor.edit((builder) => {
    builder.replace(
      new vscode.Range(start, end),
      `${lastMatch![1]}{${newKey}}`,
    );
  });
}

// ── Fetch helper ───────────────────────────────────────────────────

interface FetchArgs {
  backendUrl: string;
  text: string;
  languageId: string;
  documentPath: string;
  topK: number;
  timeoutMs: number;
  token: vscode.CancellationToken;
}

async function fetchRecommendations(args: FetchArgs): Promise<RecommendResponse> {
  const url = joinUrl(args.backendUrl, RECOMMEND_PATH);
  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), args.timeoutMs);
  const cancelSub = args.token.onCancellationRequested(() => controller.abort());

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: args.text,
        top_k: args.topK,
        language: args.languageId,
        document_path: args.documentPath,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const detail = await safeReadText(res);
      throw new Error(
        `backend returned ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`,
      );
    }

    const payload = (await res.json()) as RecommendResponse;
    if (!payload || !Array.isArray(payload.candidates)) {
      throw new Error("backend returned malformed response (missing 'candidates').");
    }
    return payload;
  } catch (err) {
    if (isAbortError(err)) {
      if (args.token.isCancellationRequested) {
        throw new Error("request cancelled.");
      }
      throw new Error(`request timed out after ${args.timeoutMs} ms.`);
    }
    if (err instanceof Error) {
      throw new Error(`could not reach ${url} — ${err.message}`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutHandle);
    cancelSub.dispose();
  }
}

interface ScanFetchArgs extends FetchArgs {
  maxSentences: number;
  minConfidence: number;
}

async function fetchScanResults(args: ScanFetchArgs): Promise<ScanResponse> {
  const url = joinUrl(args.backendUrl, SCAN_PATH);
  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), args.timeoutMs);
  const cancelSub = args.token.onCancellationRequested(() => controller.abort());

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: args.text,
        top_k: args.topK,
        language: args.languageId,
        document_path: args.documentPath,
        max_sentences: args.maxSentences,
        min_confidence: args.minConfidence,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const detail = await safeReadText(res);
      throw new Error(
        `backend returned ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`,
      );
    }

    const payload = (await res.json()) as ScanResponse;
    if (!payload || !Array.isArray(payload.items)) {
      throw new Error("backend returned malformed response (missing 'items').");
    }
    return payload;
  } catch (err) {
    if (isAbortError(err)) {
      if (args.token.isCancellationRequested) {
        throw new Error("request cancelled.");
      }
      throw new Error(`request timed out after ${args.timeoutMs} ms.`);
    }
    if (err instanceof Error) {
      throw new Error(`could not reach ${url} — ${err.message}`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutHandle);
    cancelSub.dispose();
  }
}

// ── QuickPick flow ─────────────────────────────────────────────────

async function pickScanItem(items: ScanItem[]): Promise<ScanItem | undefined> {
  const picked = await vscode.window.showQuickPick(
    items.map(toScanQuickPickItem),
    {
      title: "Missing citation candidates",
      placeHolder: "Select a flagged sentence to inspect candidate references",
      matchOnDescription: true,
      matchOnDetail: true,
    },
  );
  return picked?.item;
}

function toScanQuickPickItem(item: ScanItem): ScanPickItem {
  const confidencePct = Math.round(item.confidence * 100);
  const bestCandidate = item.candidates[0];
  const candidateLabel = bestCandidate
    ? `${bestCandidate.title} (${yearLabel(bestCandidate)})`
    : "No candidate reference";

  // Hard-wrapped sentences carry their source line breaks/indentation; collapse
  // them so the single-line QuickPick label reads cleanly.
  const preview = item.text.replace(/\s+/g, " ").trim();

  return {
    label: `$(warning) ${confidencePct}%  ${truncate(preview, 92)}`,
    description: candidateLabel,
    detail: item.reasons.join(" · "),
    item,
  };
}

const COPY_BIBTEX_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon("copy"),
  tooltip: "Copy BibTeX entry to clipboard",
};

const SEARCH_ONLINE_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon("search"),
  tooltip: "Search for this paper online",
};

const EVIDENCE_DETAIL_MAX = 220;

interface QuickPickArgs {
  editor: vscode.TextEditor;
  selection: vscode.Selection;
  languageId: string;
  candidates: Candidate[];
  query: string;
  feedbackClient: FeedbackClient;
  eventId: string | null;
}

async function runQuickPick(args: QuickPickArgs): Promise<void> {
  const chosen = await pickCandidate(args);
  if (!chosen) {
    return;
  }

  const insertedKey = await insertCitation(
    args.editor,
    args.selection,
    chosen,
    args.languageId,
  );
  if (insertedKey === null) {
    return; // insertion failed — warning already shown
  }

  await maybeAppendBibtex(args.editor, chosen, insertedKey);
  logFeedback(args.feedbackClient, args.eventId, chosen, "accepted");
}

/**
 * Show a styled QuickPick of candidates and resolve to the chosen one
 * (or `undefined` if the picker is dismissed).
 *
 * Uses `createQuickPick` rather than `showQuickPick` so each item can carry
 * action buttons (Copy BibTeX, Search online) that fire — and are logged as
 * feedback — without closing the picker.
 */
function pickCandidate(args: QuickPickArgs): Promise<Candidate | undefined> {
  return new Promise((resolve) => {
    const picker = vscode.window.createQuickPick<CandidatePickItem>();
    picker.title = `Citations for "${truncate(args.query, 60)}"`;
    picker.placeholder = "Select a paper to cite — type to filter";
    picker.matchOnDescription = true;
    picker.matchOnDetail = true;
    picker.ignoreFocusOut = true;
    picker.items = args.candidates.map(toQuickPickItem);

    let accepted = false;

    picker.onDidTriggerItemButton(async (event) => {
      const candidate = event.item.candidate;
      if (event.button === COPY_BIBTEX_BUTTON) {
        await vscode.env.clipboard.writeText(candidate.bibtex);
        vscode.window.setStatusBarMessage(
          `$(check) Copied BibTeX for ${candidate.citation_key}`,
          3000,
        );
        logFeedback(args.feedbackClient, args.eventId, candidate, "copied_bibtex");
      } else if (event.button === SEARCH_ONLINE_BUTTON) {
        await openOnline(candidate);
        logFeedback(args.feedbackClient, args.eventId, candidate, "opened_url");
      }
    });

    picker.onDidAccept(() => {
      accepted = true;
      const selected = picker.selectedItems[0];
      picker.hide();
      resolve(selected?.candidate);
    });

    picker.onDidHide(() => {
      picker.dispose();
      if (!accepted) {
        resolve(undefined);
      }
    });

    picker.show();
  });
}

function toQuickPickItem(candidate: Candidate): CandidatePickItem {
  return {
    label: `$(book) ${authorLabel(candidate)} (${yearLabel(candidate)}) — ${candidate.title}`,
    description: buildDescription(candidate),
    detail: buildDetail(candidate),
    buttons: [COPY_BIBTEX_BUTTON, SEARCH_ONLINE_BUTTON],
    candidate,
  };
}

/** Right-hand metadata line: match meter · venue · author count. */
function buildDescription(candidate: Candidate): string {
  const parts: string[] = [];
  const meter = matchMeter(candidate);
  if (meter) {
    parts.push(`${meter.dots} ${meter.pct}% match`);
  }
  if (candidate.venue) {
    parts.push(candidate.venue);
  }
  const count = candidate.authors.length;
  if (count > 0) {
    parts.push(count === 1 ? "1 author" : `${count} authors`);
  }
  return parts.join("  ·  ");
}

/** Secondary line: the strongest supporting sentence, with its citing year. */
function buildDetail(candidate: Candidate): string | undefined {
  const evidence = bestEvidence(candidate);
  if (!evidence?.sentence) {
    return undefined;
  }
  const quote = `❝ ${truncate(evidence.sentence, EVIDENCE_DETAIL_MAX)} ❞`;
  return evidence.citing_year ? `${quote}  — cited ${evidence.citing_year}` : quote;
}

function logFeedback(
  client: FeedbackClient,
  eventId: string | null,
  candidate: Candidate,
  feedbackType: FeedbackType,
): void {
  if (!eventId) {
    return;
  }
  client.send({
    event_id: eventId,
    result_id: candidate.result_id,
    feedback_type: feedbackType,
  });
}

// ── Insertion ──────────────────────────────────────────────────────

const SCHOLAR_SEARCH_URL = "https://scholar.google.com/scholar?q=";

async function openOnline(candidate: Candidate): Promise<void> {
  const url = SCHOLAR_SEARCH_URL + encodeURIComponent(candidate.title);
  await vscode.env.openExternal(vscode.Uri.parse(url));
}

/**
 * Insert the citation marker for `candidate` at the end of `selection`.
 * Returns the citation key that was inserted, or `null` if insertion failed.
 */
async function insertCitation(
  editor: vscode.TextEditor,
  selection: vscode.Selection,
  candidate: Candidate,
  languageId: string,
): Promise<string | null> {
  const formatted = formatCitation(candidate satisfies PaperLike, languageId);
  const { position, insertion } = computeInsertion(editor, selection, formatted);

  const success = await editor.edit((builder) => {
    builder.insert(position, insertion);
  });
  if (!success) {
    vscode.window.showWarningMessage(
      "Missing Citations: could not modify the document (read-only?).",
    );
    return null;
  }

  // Move the cursor to just after the inserted marker.
  const insertedEndOffset =
    editor.document.offsetAt(position) + insertion.length;
  const newCursor = editor.document.positionAt(insertedEndOffset);
  editor.selection = new vscode.Selection(newCursor, newCursor);

  return candidate.citation_key;
}

const TERMINAL_PUNCTUATION = /[.!?:;,]$/;

interface InsertionPlan {
  position: vscode.Position;
  insertion: string;
}

/**
 * Insert the cite marker at the end of the selection, but *before* terminal
 * punctuation if the selection ends with any (LaTeX/Pandoc convention:
 * "…similarity \cite{key}." rather than "…similarity. \cite{key}").
 */
function computeInsertion(
  editor: vscode.TextEditor,
  selection: vscode.Selection,
  marker: string,
): InsertionPlan {
  const doc = editor.document;
  const selectionText = doc.getText(selection);

  const trailingWhitespaceMatch = /\s+$/.exec(selectionText);
  const trailingWhitespaceLength = trailingWhitespaceMatch
    ? trailingWhitespaceMatch[0].length
    : 0;
  const trimmedRight = trailingWhitespaceLength
    ? selectionText.slice(0, -trailingWhitespaceLength)
    : selectionText;

  const endOfSelectionOffset = doc.offsetAt(selection.end);
  // Anchor at the last non-whitespace character of the selection.
  let insertionOffset = endOfSelectionOffset - trailingWhitespaceLength;

  if (TERMINAL_PUNCTUATION.test(trimmedRight)) {
    insertionOffset -= 1;
  }

  const charBefore =
    insertionOffset > 0 ? doc.getText().charAt(insertionOffset - 1) : "";
  const needsLeadingSpace = charBefore !== "" && !/\s/.test(charBefore);
  const insertion = needsLeadingSpace ? ` ${marker}` : marker;

  return {
    position: doc.positionAt(insertionOffset),
    insertion,
  };
}

// ── Small HTTP helpers ─────────────────────────────────────────────

function joinUrl(base: string, path: string): string {
  const trimmedBase = base.replace(/\/+$/, "");
  const trimmedPath = path.startsWith("/") ? path : `/${path}`;
  return `${trimmedBase}${trimmedPath}`;
}

async function safeReadText(res: Response): Promise<string> {
  try {
    const text = await res.text();
    return text.slice(0, 500);
  } catch {
    return "";
  }
}

function isAbortError(err: unknown): boolean {
  return (
    err instanceof Error &&
    (err.name === "AbortError" || err.message.toLowerCase().includes("aborted"))
  );
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
