'use strict';
/**
 * Launches a REAL VS Code (the locally installed one) in an isolated profile with the HARNESS
 * extension loaded, then runs test/integration/suite.js inside the extension host.
 * Requires: engine on :8089 and `harness-cli serve` on :8090.
 */
const path = require('path');
const fs = require('fs');
const os = require('os');
const { runTests } = require('@vscode/test-electron');

async function main() {
  const localCode = path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Microsoft VS Code', 'Code.exe');
  const vscodeExecutablePath = process.env.VSCODE_EXE || (fs.existsSync(localCode) ? localCode : undefined);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'harness-vscode-'));
  const workspace = path.join(profile, 'workspace');
  fs.mkdirSync(workspace, { recursive: true });

  await runTests({
    vscodeExecutablePath,
    extensionDevelopmentPath: path.resolve(__dirname, '..'),
    extensionTestsPath: path.resolve(__dirname, 'integration', 'suite.js'),
    extensionTestsEnv: { HARNESS_TEST_WORKSPACE: workspace },
    launchArgs: [
      workspace,
      '--user-data-dir', path.join(profile, 'user-data'),
      '--extensions-dir', path.join(profile, 'extensions'),
      '--disable-workspace-trust',
      '--skip-welcome',
      '--skip-release-notes',
    ],
  });
}

main().catch((err) => {
  console.error('Integration tests failed:', err && err.message ? err.message : err);
  process.exit(1);
});
