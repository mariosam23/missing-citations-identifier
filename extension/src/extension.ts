import * as vscode from "vscode";

import { recommendCitationsForSelection } from "./recommendCommand";

const COMMAND_ID = "missingCitations.recommend";

export function activate(context: vscode.ExtensionContext): void {
  const disposable = vscode.commands.registerCommand(
    COMMAND_ID,
    recommendCitationsForSelection,
  );
  context.subscriptions.push(disposable);
}

export function deactivate(): void {
  // No-op: all disposables are owned by the extension context.
}
