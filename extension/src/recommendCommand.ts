import * as vscode from "vscode";

import { formatCitation, PaperLike } from "./citationFormatter";
import { BibTeXManager, AppendResult } from "./bibtexManager";

const CONFIG_SECTION = "missingCitations";
const RECOMMEND_PATH = "/recommend";

const LATEX_LANGUAGE_IDS = new Set(["latex", "tex"]);

interface Evidence {
  sentence: string;
  citing_year: number | null;
  similarity: number;
}

interface Candidate {
  paper_id: number;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  citation_key: string;
  score: number;
  evidence: Evidence[];
  bibtex: string;
}

interface RecommendResponse {
  candidates: Candidate[];
}

interface CandidatePickItem extends vscode.QuickPickItem {
  candidate: Candidate;
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
  const languageId = editor.document.languageId;

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

  const chosen = await pickCandidate(response.candidates, selectedText);
  if (!chosen) {
    return;
  }

  const insertedKey = await insertCitation(editor, chosen, languageId);
  if (insertedKey === null) {
    return; // insertion failed — warning already shown
  }

  // ── BibTeX auto-append (LaTeX only) ──────────────────────────────
  await maybeAppendBibtex(editor, chosen, insertedKey);
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
      const authorLabel = candidate.authors[0] ?? "Unknown";
      const yearStr = candidate.year !== null ? String(candidate.year) : "n.d.";
      vscode.window.showInformationMessage(
        `Missing Citations: appended ${authorLabel} (${yearStr}) to ${relativeBib}`,
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

// ── Fetch / insert helpers (unchanged from Phase 4) ────────────────

interface FetchArgs {
  backendUrl: string;
  text: string;
  languageId: string;
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

// ── Rich candidate picker ──────────────────────────────────────────

const COPY_BIBTEX_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon("copy"),
  tooltip: "Copy BibTeX entry to clipboard",
};

const SEARCH_ONLINE_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon("search"),
  tooltip: "Search for this paper online",
};

const SCHOLAR_SEARCH_URL = "https://scholar.google.com/scholar?q=";
const METER_SEGMENTS = 5;
const EVIDENCE_DETAIL_MAX = 220;

/**
 * Show a styled QuickPick of candidates and resolve to the chosen one
 * (or `undefined` if the picker is dismissed).
 *
 * Uses `createQuickPick` rather than `showQuickPick` so each item can carry
 * action buttons (Copy BibTeX, Search online) that fire without closing the
 * picker.
 */
function pickCandidate(
  candidates: Candidate[],
  query: string,
): Promise<Candidate | undefined> {
  return new Promise((resolve) => {
    const picker = vscode.window.createQuickPick<CandidatePickItem>();
    picker.title = `Citations for "${truncate(query, 60)}"`;
    picker.placeholder = "Select a paper to cite — type to filter";
    picker.matchOnDescription = true;
    picker.matchOnDetail = true;
    picker.ignoreFocusOut = true;
    picker.items = candidates.map(toQuickPickItem);

    let accepted = false;

    picker.onDidTriggerItemButton(async (event) => {
      const candidate = event.item.candidate;
      if (event.button === COPY_BIBTEX_BUTTON) {
        await vscode.env.clipboard.writeText(candidate.bibtex);
        vscode.window.setStatusBarMessage(
          `$(check) Copied BibTeX for ${candidate.citation_key}`,
          3000,
        );
      } else if (event.button === SEARCH_ONLINE_BUTTON) {
        const url = SCHOLAR_SEARCH_URL + encodeURIComponent(candidate.title);
        await vscode.env.openExternal(vscode.Uri.parse(url));
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
    parts.push(meter);
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

/**
 * A five-segment dot meter built from the best evidence cosine similarity.
 * This is an honest semantic-match signal in [0, 1]; the backend `score`
 * (which mixes in a corroboration bonus and is not bounded) is used only
 * for ordering, never shown as a percentage.
 */
function matchMeter(candidate: Candidate): string | undefined {
  const evidence = bestEvidence(candidate);
  if (!evidence) {
    return undefined;
  }
  const pct = Math.max(0, Math.min(100, Math.round(evidence.similarity * 100)));
  const filled = Math.round((pct / 100) * METER_SEGMENTS);
  const dots = "●".repeat(filled) + "○".repeat(METER_SEGMENTS - filled);
  return `${dots} ${pct}% match`;
}

function bestEvidence(candidate: Candidate): Evidence | undefined {
  if (candidate.evidence.length === 0) {
    return undefined;
  }
  return candidate.evidence.reduce((best, current) =>
    current.similarity > best.similarity ? current : best,
  );
}

function authorLabel(candidate: Candidate): string {
  const surname = firstAuthorSurname(candidate.authors[0]);
  if (!surname) {
    return "Unknown";
  }
  return candidate.authors.length > 1 ? `${surname} et al.` : surname;
}

function yearLabel(candidate: Candidate): string {
  return candidate.year !== null ? String(candidate.year) : "n.d.";
}

/** Surname from "Last, First" or "First Last"; `null` if unusable. */
function firstAuthorSurname(raw: string | undefined): string | null {
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

/**
 * Insert the citation marker into the editor.
 * Returns the citation key that was inserted, or `null` if insertion failed.
 */
async function insertCitation(
  editor: vscode.TextEditor,
  candidate: Candidate,
  languageId: string,
): Promise<string | null> {
  const formatted = formatCitation(candidate satisfies PaperLike, languageId);
  const { position, insertion } = computeInsertion(editor, formatted);

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
  marker: string,
): InsertionPlan {
  const doc = editor.document;
  const selection = editor.selection;
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

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value;
  }
  return `${value.slice(0, max - 1).trimEnd()}…`;
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
