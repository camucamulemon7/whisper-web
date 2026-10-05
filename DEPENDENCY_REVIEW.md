# Frontend dependency review

The original audit reported seven affected **package entries**, not seven
independent application vulnerabilities. The least disruptive update is Vite
6.4.3 with a lockfile; React 18 and Tailwind 3 remain unchanged. The Dockerfile
uses `npm ci` so the tested dependency tree is reproducible.

## Impact and remediation

| Original package | Scope and reachability in this repository | Remediation/status |
| --- | --- | --- |
| vite 5.4.21 | Dev-server file access vulnerabilities. `vite.config.ts` and the frontend Dockerfile expose the dev server on `0.0.0.0`; therefore these tools are also relevant to the existing container deployment, even though Vite is a devDependency. The source-map issue can affect Mac/Linux. The UNC/alternate-path issues require Windows. Static built assets do not contain Vite. | Update to 6.4.3, the lowest supported patched major line. Compatible with the existing Node 20 container, React plugin 4.7.0, and project configuration. |
| esbuild 0.21.5 | The advisory affects esbuild's own serving API. This application uses Vite's server and does not call `esbuild.serve()`. Not identified as reachable from this app's audio or transcription inputs. | Vite 6.4.3 resolves esbuild 0.25.12; the reported serving advisory is patched from 0.25.0. |
| braces 3.0.3 | Deeply nested attacker-controlled brace/glob patterns can exhaust the stack. It is used by build/watch tooling, not the browser application. Tailwind's two content globs are fixed in `tailwind.config.js`; no API or transcription input is passed to glob matching. Arbitrary third-party build configuration would change this assessment. | The official advisory lists no patched version; the registry's latest braces is still 3.0.3. A patch/minor update or compatible version override cannot remove it. |
| chokidar 3.6.0 | Inherits the braces finding through Tailwind's file watcher. Same build/watch input boundary. | Remains with Tailwind 3. |
| micromatch 4.0.8 | Inherits the braces finding through Tailwind glob matching. Same build/watch input boundary. | Remains with Tailwind 3. |
| fast-glob 3.3.3 | Inherits the micromatch/braces finding. Same build/watch input boundary. | Remains with Tailwind 3. |
| tailwindcss 3.4.19 | Inherits the preceding dependencies. CSS compilation uses trusted repository sources; no user speech/text is passed to its glob configuration. Compiled CSS has no Node glob dependency. | Remains at 3.4.19; the latest 3.x release does not remove braces. |

After the Vite update, the full audit has **five high package entries** tracing
to one remaining braces advisory; `npm audit --omit=dev` has **zero findings**.
This does not mean that the Docker deployment is unaffected: it deliberately
runs the development server and includes development dependencies.

Primary references:

- [Vite source-map traversal](https://github.com/advisories/GHSA-4w7w-66w2-5vf9)
- [Windows UNC editor disclosure](https://github.com/advisories/GHSA-v6wh-96g9-6wx3)
- [Windows alternate-path bypass](https://github.com/advisories/GHSA-fx2h-pf6j-xcff)
- [esbuild serving advisory](https://github.com/advisories/GHSA-67mh-4wv8-2f99)
- [Vite maintainer's reachability explanation](https://github.com/vitejs/vite/issues/19412)
- [braces advisory, with no patched release](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
- [Supported Vite release lines](https://vite.dev/releases)

## Compatibility evidence

- Clean `npm ci --ignore-scripts`, TypeScript check and Vite production build pass.
- The Vite 5 and Vite 6 emitted CSS have identical parsed rules/declarations;
  only whitespace in media query formatting differs.
- Four Node/jsdom tests render the real component, exercise theme persistence,
  parameter visibility/defaults, and synthetic audio permission rejection while
  checking that all original screen-capture hints are preserved.
- Six GPU-free backend lifecycle tests pass, including disposal of a new worker
  when the old worker's cleanup raises.
- The isolated localhost dev server returns its HTML and transformed TSX source.
  API calls in this validation target a separate unused localhost port; DOM
  tests mock all fetch/media calls.
- Eight Playwright tests run in the installed Google Chrome with a fresh
  headless profile. They verify both themes, parameters, synthetic PCM buffers,
  explicit stop, connection failure, remote close, retry, stale socket callbacks,
  and audio initialization failure. Screenshots of both themes and a synthetic
  caption were inspected. No real microphone/screen capture or permissions are
  used; native capture/audio and transcription services are mocked.
- A browser regression exposed lingering capture/socket/timer resources after
  failures. The component now releases them on error, close, and setup failure;
  stale events cannot stop a newer retry. Captions remain after stopping.
- Real screen/audio sharing and NVIDIA inference remain environment checks.
  CUA has no connected browser, but installed Chrome works through Playwright.

## Tailwind 4 experiment

The separate `experiment/tailwind4-compatibility` branch updates Tailwind to
4.3.3 and uses `@tailwindcss/postcss`. Replacing the CSS directives with an
import and explicit config makes its production build pass and its audit reach
zero findings. It is **not adopted** in this fix branch.

The generated CSS loses `.bg-opacity-75` and `.bg-opacity-50`, which this app
uses for transcription/correction/summary panels and the overlay. Tailwind 4
also changes defaults and requires Safari 16.4+, Chrome 111+, and Firefox 128+.
These are documented in the [official migration guide](https://tailwindcss.com/docs/upgrade-guide).

The smallest migration candidate is to replace the four affected JSX class
combinations with `bg-black/75` or `bg-black/50`, review changed default styles,
and compare light/dark and correction/summary layouts in supported browsers.
Confirm the supported-browser requirement before adopting that migration.
