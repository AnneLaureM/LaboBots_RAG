# Vendored: pdf.js (legacy build)

- **Source**: [pdfjs-dist](https://www.npmjs.com/package/pdfjs-dist) on npm, version `4.10.38`.
- **Files**: `pdf.min.mjs` + `pdf.worker.min.mjs`, copied verbatim from that package's
  `legacy/build/` directory (the browser-compatible build with the widest compatibility --
  matches this extension's `strict_min_version: 115.0`).
- **License**: Apache License 2.0 -- see `LICENSE` in this directory (copied from the same
  package). Mozilla's own project: <https://github.com/mozilla/pdf.js>.
- **Why vendored, not CDN-loaded**: this extension only ever talks to `localhost`/`127.0.0.1`
  (manifest host permissions) and works fully offline once installed -- loading a script from a
  remote CDN would break both of those properties, and add a live dependency on a third party at
  runtime for something as core as reading an attachment.
- **Used by**: `background.js`'s `extractPdfText()` -- dynamically imports `pdf.min.mjs` and
  points `GlobalWorkerOptions.workerSrc` at `pdf.worker.min.mjs`, both via `browser.runtime.getURL()`.

To update: `npm pack pdfjs-dist@<version>`, extract, copy `legacy/build/pdf.min.mjs`,
`legacy/build/pdf.worker.min.mjs`, and `LICENSE` here, then bump the version above.
