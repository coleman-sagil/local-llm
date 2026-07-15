# Google Drive + Calendar MCP Setup

This sets up the `google-drive` entry in `cli/mcp_servers.json` (currently
`"enabled": false` because it has no credentials yet). It uses
[`@piotr-agier/google-drive-mcp`](https://github.com/piotr-agier/google-drive-mcp),
an npx-launched, stdio-transport MCP server covering Google Drive, Docs,
Sheets, Slides, and Calendar - actively maintained (v2.5.0, 188 stars, MIT
license) and a good fit here because it's Node-based like the other MCP
servers in this project (filesystem, playwright), and its setup is exactly
"download a `credentials.json`, run an `auth` command" - no extra runtime to
install.

**Steps 1-8 below need your own interactive Google login and clicking around
the Google Cloud Console. Nobody else, and no script, can do that part for
you - it's your Google account giving consent, on purpose.** Everything
after that (steps 9-10) is copy-pasteable from a terminal.

## Why a *new* Google Cloud project

Use a brand-new project for this, **separate from any existing K&S business
project** (e.g. whatever backs `~/ks-pm/integrations/`). This is a personal
local-LLM tool talking to your personal Google account with broad Drive/
Calendar scopes and a locally-cached OAuth token - it should not share a
trust boundary, a project, or credentials with client/business tooling.

## 1. Create the Google Cloud project

1. Go to <https://console.cloud.google.com/projectcreate>.
2. Project name: something recognizable, e.g. `local-llm-mcp`.
3. Leave "Organization" / "Location" at their defaults unless you know you
   want otherwise.
4. Click **Create**, then make sure the new project is selected in the
   project switcher (top left) before continuing - every step below applies
   to whichever project is currently selected.

## 2. Enable the required APIs

The server exposes tools for Drive, Docs, Sheets, Slides, and Calendar, and
each needs its own API enabled or that tool group will fail at call time. Go
to **APIs & Services > Library** and enable each of these (search by name,
click it, click **Enable**):

- Google Drive API
- Google Docs API
- Google Sheets API
- Google Slides API
- Google Calendar API

If you only care about Drive + Calendar tools right now, you can skip Docs/
Sheets/Slides and enable them later if you ever need those tools - the
others will just fail cleanly until then.

## 3. Configure the OAuth consent screen

1. Go to **APIs & Services > OAuth consent screen**.
2. User type: **External** (this is the only option for a regular
   `@gmail.com`/personal account; **Internal** only exists for Google
   Workspace-managed accounts).
3. Fill in the required fields: app name (e.g. `local-llm-mcp`), your email
   as user support email, and your email again as developer contact.
4. On the **Scopes** step, add these scopes (click **Add or Remove Scopes**
   and paste them, or search by name):
   - `https://www.googleapis.com/auth/drive`
   - `https://www.googleapis.com/auth/drive.file`
   - `https://www.googleapis.com/auth/drive.readonly`
   - `https://www.googleapis.com/auth/documents`
   - `https://www.googleapis.com/auth/spreadsheets`
   - `https://www.googleapis.com/auth/presentations`
   - `https://www.googleapis.com/auth/calendar`
   - `https://www.googleapis.com/auth/calendar.events`
5. On the **Test users** step, add your own Google account's email address.
   While the app is in "Testing" status, only accounts on this list can
   authenticate.
6. Save through to the summary page.

**Note on refresh tokens:** while the app stays in "Testing" publishing
status, Google expires refresh tokens after 7 days, which means re-running
the browser auth flow every week. Once you've confirmed the first auth
works (step 10), go back to **OAuth consent screen** and click **Publish
App**. You don't need Google's verification review for your own personal
use (you'll just see an "unverified app" warning during consent, which is
expected and fine to click through) - publishing just removes the 7-day
expiry.

## 4. Create the OAuth client credentials

1. Go to **APIs & Services > Credentials**.
2. Click **Create Credentials > OAuth client ID**.
3. Application type: **Desktop app** (not "Web application" - this matters,
   Desktop app credentials handle the loopback redirect that the CLI auth
   flow needs automatically).
4. Name it something recognizable, e.g. `local-llm-mcp-desktop`.
5. Click **Create**, then **Download JSON** on the resulting credential.

## 5. Save the credentials file to the exact expected path

Move (don't copy-and-leave-a-duplicate) the downloaded file to:

```
/home/mateo/local-llm/cli/google_credentials.json
```

This exact path is already wired into `cli/mcp_servers.json` as the
`google-drive` entry's `GOOGLE_DRIVE_OAUTH_CREDENTIALS` env var, and is
already listed in `.gitignore` so it can never get committed by accident.
Double-check it's really gitignored before moving on:

```bash
cd /home/mateo/local-llm
git check-ignore -v cli/google_credentials.json
```

That should print a line pointing at the `.gitignore` rule. If it prints
nothing, stop and fix that before putting real credentials in the file.

## 6. Node version gotcha (read before running anything)

The system Node on this machine (`/usr/bin/node`, v12, from apt, EOL) cannot
run this MCP server - it uses modern JS syntax and dies with
`SyntaxError: Unexpected reserved word`. A modern Node 22 LTS is already
installed standalone (no sudo) at `/home/mateo/local-llm/.tools/node22/`.
Put it first on `PATH` for the commands below:

```bash
export PATH=/home/mateo/local-llm/.tools/node22/bin:$PATH
node --version   # should print v22.x, not v12.x - if it doesn't, stop here
```

(This is the same fix `cli/mcp_smoke_test.py` hardcodes for the filesystem/
playwright MCP servers - same underlying problem, same fix.)

## 7. Sanity-check npx can reach the package

```bash
npx -y @piotr-agier/google-drive-mcp --version
```

First run downloads the package (needs internet); later runs are cached and
fast. If this hangs or errors, re-check step 6 (PATH) before anything else.

## 8. Run the first-run auth command

```bash
GOOGLE_DRIVE_OAUTH_CREDENTIALS=/home/mateo/local-llm/cli/google_credentials.json \
  npx -y @piotr-agier/google-drive-mcp auth
```

**This is the step that requires you personally:** it opens a browser
window, you log into the Google account you added as a test user in step 3,
and you click through the consent screen (including the "unverified app"
warning if the app isn't published yet - see the note in step 3). On
success it writes a token cache to `~/.config/google-drive-mcp/tokens.json`
(outside this repo, nothing to gitignore there).

If no browser opens automatically (e.g. running over SSH), the command
should print a URL to open manually - paste it into a browser on any machine
you're logged into Google on, and it'll redirect back to `localhost` for the
CLI to pick up.

## 9. Flip the server on

Once step 8 succeeds, open `cli/mcp_servers.json` and change the
`google-drive` entry's `"enabled"` field from `false` to `true`.

## 10. Verify

However the MCP client ends up invoking configured servers (see
`cli/mcp_smoke_test.py` for the standalone proof-of-plumbing pattern this
project already validated for the filesystem/playwright servers), confirm
`google-drive` shows up with its tool list and that a low-risk read-only
call (e.g. listing recent Drive files, or listing calendars) succeeds before
trusting it with anything that writes.

## Troubleshooting

- **`SyntaxError: Unexpected reserved word`** - system Node (v12) got used
  instead of the node22 toolchain. Re-check step 6; note that `env node`
  inside `npx`/`npm`'s own shebang re-resolves "node" via `PATH` at
  execution time, so having node22 first on `PATH` for the *outer* command
  isn't automatically enough for everything it spawns - see the long
  comment at the top of `cli/mcp_smoke_test.py` for the full explanation and
  the direct-binary-invocation workaround if you hit this from a non-shell
  context (e.g. a client that doesn't inherit your shell's `PATH`).
- **`invalid_grant` / token errors after a week of inactivity** - almost
  certainly the 7-day "Testing" refresh-token expiry from step 3. Publish
  the app (still no verification needed for personal use) and re-run step 8
  once.
- **Consent screen shows "Google hasn't verified this app"** - expected for
  an unverified personal app; click **Advanced > Go to (app name) (unsafe)**
  to proceed. This is safe because it's your own app and your own Google
  Cloud project.
