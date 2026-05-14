/**
 * Manages `.bib` file discovery, key-collision resolution, and atomic
 * appends.  All file operations go through `vscode.workspace.fs` so that
 * VS Code's file watcher, source control, and autosave hooks are respected.
 *
 * Concurrency: a single in-memory mutex serialises writes so two rapid
 * "insert citation" commands cannot interleave reads and appends.
 */

import * as vscode from "vscode";
import { extractKeys } from "./bibUtils";

const CONFIG_SECTION = "missingCitations";

/**
 * Result of an `appendEntry` operation.
 */
export interface AppendResult {
  /** The key that was actually written (may differ from input after collision). */
  wroteKey: string;
  /** `true` if a brand-new file was created. */
  created: boolean;
  /** `true` if the entry already existed and was skipped. */
  skipped: boolean;
}

export class BibTeXManager {
  /** Serialises `.bib` writes. */
  private writeLock: Promise<void> = Promise.resolve();

  // ── Public API ──────────────────────────────────────────────────────

  /**
   * Discover (or create) the `.bib` file for the given document.
   *
   * Resolution order:
   *  1. Explicit `missingCitations.bibFile` setting.
   *  2. Walk from the document's directory up to the workspace root,
   *     picking the closest `*.bib`.
   *  3. If `createBibFileIfMissing` is true, create `references.bib`
   *     next to the document.
   */
  async findOrCreate(
    document: vscode.TextDocument,
  ): Promise<vscode.Uri | null> {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

    // 1) Explicit setting.
    const explicitPath = (config.get<string>("bibFile") ?? "").trim();
    if (explicitPath) {
      return this.resolveExplicit(explicitPath, document);
    }

    // 2) Walk upward.
    const found = await this.walkForBib(document);
    if (found) {
      return found;
    }

    // 3) Create if allowed.
    const create = config.get<boolean>("createBibFileIfMissing") ?? true;
    if (!create) {
      return null;
    }

    const docDir = vscode.Uri.joinPath(document.uri, "..");
    const newBib = vscode.Uri.joinPath(docDir, "references.bib");
    await vscode.workspace.fs.writeFile(newBib, new Uint8Array());
    vscode.window.showInformationMessage(
      `Missing Citations: created ${vscode.workspace.asRelativePath(newBib)}`,
    );
    return newBib;
  }

  /**
   * Read the `.bib` file and return all existing citation keys.
   */
  async listExistingKeys(uri: vscode.Uri): Promise<Set<string>> {
    const raw = await vscode.workspace.fs.readFile(uri);
    const text = new TextDecoder("utf-8").decode(raw);
    return extractKeys(text);
  }

  /**
   * Append a BibTeX entry to the `.bib` file, handling collisions.
   *
   * - If `key` is not yet present → append.
   * - If `key` is present with substantively identical metadata → skip.
   * - If `key` is present but different → rename to `keyA`, `keyB`, etc.
   */
  async appendEntry(
    uri: vscode.Uri,
    entry: string,
    key: string,
  ): Promise<AppendResult> {
    // Serialise all writes through the lock.
    const resultPromise = new Promise<AppendResult>((resolve, reject) => {
      this.writeLock = this.writeLock
        .then(async () => {
          const result = await this.doAppend(uri, entry, key);
          resolve(result);
        })
        .catch(reject);
    });
    return resultPromise;
  }

  // ── Private helpers ─────────────────────────────────────────────────

  private async resolveExplicit(
    relPath: string,
    document: vscode.TextDocument,
  ): Promise<vscode.Uri | null> {
    const ws = vscode.workspace.getWorkspaceFolder(document.uri);
    if (!ws) {
      return null;
    }
    const target = vscode.Uri.joinPath(ws.uri, relPath);
    try {
      await vscode.workspace.fs.stat(target);
      return target;
    } catch {
      const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
      const create = config.get<boolean>("createBibFileIfMissing") ?? true;
      if (create) {
        await vscode.workspace.fs.writeFile(target, new Uint8Array());
        vscode.window.showInformationMessage(
          `Missing Citations: created ${vscode.workspace.asRelativePath(target)}`,
        );
        return target;
      }
      return null;
    }
  }

  private async walkForBib(
    document: vscode.TextDocument,
  ): Promise<vscode.Uri | null> {
    const ws = vscode.workspace.getWorkspaceFolder(document.uri);
    if (!ws) {
      return null;
    }

    let dir = vscode.Uri.joinPath(document.uri, "..");
    const wsRoot = ws.uri.fsPath;

    while (true) {
      try {
        const entries = await vscode.workspace.fs.readDirectory(dir);
        const bibEntry = entries.find(
          ([name, type]) =>
            type === vscode.FileType.File &&
            name.toLowerCase().endsWith(".bib"),
        );
        if (bibEntry) {
          return vscode.Uri.joinPath(dir, bibEntry[0]);
        }
      } catch {
        // Permission or read error — stop walking.
        break;
      }

      if (dir.fsPath === wsRoot || dir.fsPath === vscode.Uri.joinPath(dir, "..").fsPath) {
        break;
      }
      dir = vscode.Uri.joinPath(dir, "..");
    }

    return null;
  }

  private async doAppend(
    uri: vscode.Uri,
    entry: string,
    key: string,
  ): Promise<AppendResult> {
    let existingBytes: Uint8Array;
    let created = false;

    try {
      existingBytes = await vscode.workspace.fs.readFile(uri);
    } catch {
      // File disappeared between findOrCreate and now; recreate.
      existingBytes = new Uint8Array();
      created = true;
    }

    const existingText = new TextDecoder("utf-8").decode(existingBytes);
    const existingKeys = extractKeys(existingText);

    let wroteKey = key;

    if (existingKeys.has(key)) {
      // Check if the existing entry is substantively identical.
      if (this.isSameEntry(existingText, entry)) {
        return { wroteKey: key, created: false, skipped: true };
      }
      // Collision — find a free suffix.
      wroteKey = this.resolveCollision(key, existingKeys);
      // Rewrite the key inside the entry string.
      entry = entry.replace(
        new RegExp(`^(@\\w+\\{)${escapeRegex(key)},`, "m"),
        `$1${wroteKey},`,
      );
    }

    // Build the new file content.
    let newText = existingText;
    if (newText.length > 0 && !newText.endsWith("\n\n")) {
      newText = newText.trimEnd() + "\n\n";
    }
    newText += entry;
    if (!newText.endsWith("\n")) {
      newText += "\n";
    }

    await vscode.workspace.fs.writeFile(
      uri,
      new TextEncoder().encode(newText),
    );

    return { wroteKey, created, skipped: false };
  }

  /**
   * Cheap heuristic: does the existing file already contain an entry
   * with both the same title and year? If so, treat it as substantively
   * identical.
   */
  private isSameEntry(existingText: string, newEntry: string): boolean {
    const titleMatch = /title\s*=\s*\{(.+?)\}/i.exec(newEntry);
    const yearMatch = /year\s*=\s*\{(\d{4})\}/i.exec(newEntry);
    if (!titleMatch) {
      return false;
    }
    const titleInExisting = existingText.includes(titleMatch[1]);
    const yearInExisting = yearMatch
      ? existingText.includes(yearMatch[1])
      : true;
    return titleInExisting && yearInExisting;
  }

  private resolveCollision(
    key: string,
    existing: Set<string>,
  ): string {
    for (let i = 0; i < 26; i++) {
      const candidate = `${key}${String.fromCharCode(97 + i)}`;
      if (!existing.has(candidate)) {
        return candidate;
      }
    }
    // Fallback: numeric suffix.
    for (let i = 26; ; i++) {
      const candidate = `${key}_${i}`;
      if (!existing.has(candidate)) {
        return candidate;
      }
    }
  }
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
