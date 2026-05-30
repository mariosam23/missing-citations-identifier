import * as vscode from "vscode";

import {
  recommendCitationsForSelection,
  scanDocumentForMissingCitations,
} from "./recommendCommand";

const RECOMMEND_COMMAND_ID = "missingCitations.recommend";
const SCAN_COMMAND_ID = "missingCitations.scanDocument";

export function activate(context: vscode.ExtensionContext): void {
  const recommendCommand = vscode.commands.registerCommand(
    RECOMMEND_COMMAND_ID,
    recommendCitationsForSelection,
  );
  const scanCommand = vscode.commands.registerCommand(
    SCAN_COMMAND_ID,
    scanDocumentForMissingCitations,
  );
  context.subscriptions.push(recommendCommand, scanCommand);
}

export function deactivate(): void {
  // No-op: all disposables are owned by the extension context.
}
