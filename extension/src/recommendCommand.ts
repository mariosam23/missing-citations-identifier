import * as vscode from "vscode";

import { formatCitation, PaperLike } from "./citationFormatter";

const CONFIG_SECTION = "missingCitations";
const RECOMMEND_PATH = "/recommend";

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
}

interface RecommendResponse {
  candidates: Candidate[];
}

interface CandidatePickItem extends vscode.QuickPickItem {
  candidate: Candidate;
}

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

  const pick = await vscode.window.showQuickPick(
    response.candidates.map(toQuickPickItem),
    {
      matchOnDescription: true,
      matchOnDetail: true,
      placeHolder: "Select a paper to cite",
      ignoreFocusOut: true,
    },
  );
  if (!pick) {
    return;
  }

  await insertCitation(editor, pick.candidate, languageId);
}

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

function toQuickPickItem(candidate: Candidate): CandidatePickItem {
  const author = candidate.authors[0] ?? "Unknown";
  const yearStr = candidate.year !== null ? String(candidate.year) : "n.d.";
  const firstEvidence = candidate.evidence[0]?.sentence ?? "";
  return {
    label: `${author} (${yearStr}) — ${candidate.title}`,
    description: candidate.venue ?? undefined,
    detail: firstEvidence ? truncate(firstEvidence, 240) : undefined,
    candidate,
  };
}

async function insertCitation(
  editor: vscode.TextEditor,
  candidate: Candidate,
  languageId: string,
): Promise<void> {
  const formatted = formatCitation(candidate satisfies PaperLike, languageId);
  const { position, insertion } = computeInsertion(editor, formatted);

  const success = await editor.edit((builder) => {
    builder.insert(position, insertion);
  });
  if (!success) {
    vscode.window.showWarningMessage(
      "Missing Citations: could not modify the document (read-only?).",
    );
    return;
  }

  // Move the cursor to just after the inserted marker.
  const insertedEndOffset =
    editor.document.offsetAt(position) + insertion.length;
  const newCursor = editor.document.positionAt(insertedEndOffset);
  editor.selection = new vscode.Selection(newCursor, newCursor);
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
