/*
 * LaboBots Mail Agent -- streaming support, pure logic only.
 *
 * Deliberately free of any browser, DOM, or extension API: line-buffering, the two wire-format
 * parsers, and one error-policy decision. That is what lets tests/stream.test.mjs load this very
 * file under Node and unit-test it; background.js loads it as a classic script (manifest.json)
 * right before itself, so the same code is what runs inside Thunderbird.
 *
 * Both wire formats are "line based" at this layer, which is all the streaming layer needs to
 * know:
 * - Ollama /api/chat with stream:true emits one JSON object per line;
 * - OpenAI-compatible SSE (LiteLLM proxy) emits "data: <json>" lines, keep-alive comment lines,
 *   and a final "data: [DONE]".
 */

/**
 * Collects a stream of text fragments and hands back only COMPLETE lines, holding the trailing
 * partial line until the bytes that finish it arrive (a chunk boundary can fall anywhere --
 * mid-line, mid-JSON, even mid-UTF-8-character -- and callers must never parse half a line).
 * Handles \n, \r\n, and a lone \r, because servers and the SSH-tunnel proxy in front of LiteLLM
 * do not always agree on the convention.
 */
class StreamLineBuffer {
  constructor() {
    this._pending = "";
  }

  /** Push new text in; returns the list of newly-complete lines (each without its terminator). */
  feed(text) {
    // A BOM may arrive as the very first bytes of the body (some servers emit one): it would
    // otherwise ride on the first line and make its JSON.parse fail, so strip it here, where
    // all text from both backends enters the pipeline.
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1); // UTF-8 BOM (see note above)
    this._pending += text;
    const lines = [];
    for (;;) {
      const nl = this._pending.indexOf("\n");
      const cr = this._pending.indexOf("\r");
      let pos = -1;
      let extra = 0;
      if (cr !== -1 && (nl === -1 || cr < nl)) {
        if (this._pending[cr + 1] === "\n") {
          // CRLF: one terminator, skip both characters.
          pos = cr;
          extra = 1;
        } else if (cr === this._pending.length - 1) {
          // A \r sitting at the very END of what we have seen so far is ambiguous: the \n of a
          // CRLF may arrive with the next chunk. Defer the decision -- emitting "the line now"
          // and "an empty line" later (if the \n does arrive) would be a visible protocol bug.
          break;
        } else {
          // A \r followed by some other character is a genuine lone-\r line end (old-Mac style).
          pos = cr;
        }
      } else if (nl !== -1) {
        pos = nl;
      }
      if (pos === -1) break;
      lines.push(this._pending.slice(0, pos));
      this._pending = this._pending.slice(pos + 1 + extra);
    }
    return lines;
  }

  /**
   * Called exactly once when the stream ends: returns whatever partial line is still waiting
   * (and clears it), because some backends close the connection after their last line without a
   * final newline instead of sending their end-of-stream marker -- that line is still a line.
   */
  flush() {
    const rest = this._pending;
    this._pending = "";
    return rest;
  }
}

/**
 * Parses ONE already-complete line of an Ollama /api/chat stream:true response into
 * { done, token }. Two things are errors: a line that is not JSON at all (the stream is not
 * what we asked for -- silently dropping it would hide a real problem such as a wrong endpoint
 * or an HTML error page), and a line carrying a non-empty `error` key (Ollama's way of reporting
 * an in-stream failure like "model not found" -- tolerating it would present a partial draft as
 * a complete one). Any OTHER valid-JSON shape is tolerated (empty token) because skipping it is
 * less bad than breaking on a server we don't fully understand.
 */
function parseOllamaLine(line) {
  if (line.trim() === "") return { done: false, token: "" };
  let obj;
  try {
    obj = JSON.parse(line);
  } catch (err) {
    throw new Error(`Ollama: invalid JSON line in the stream (near: ${line.slice(0, 120)}) -- is this really the /api/chat endpoint of an Ollama server?`);
  }
  if (obj && typeof obj.error === "string" && obj.error !== "") {
    // A mid-stream error line means the rest of the "draft" is not trustworthy: surface it
    // instead of keeping a partial draft that would look complete to the user.
    throw new Error(`Ollama stream error: ${obj.error}`);
  }
  if (obj && obj.done === true) {
    // The final line may carry one last piece of content; keep it if there is any.
    const token = obj.message && typeof obj.message.content === "string" ? obj.message.content : "";
    return { done: true, token };
  }
  const content = obj && obj.message ? obj.message.content : undefined;
  // The very first line of an Ollama stream has content:"" (the role announcement) -- that is
  // not a token, and emitting it would just send an empty token to the UI.
  return { done: false, token: typeof content === "string" ? content : "" };
}

/**
 * Parses ONE already-complete line of an OpenAI-compatible SSE stream (what the LiteLLM proxy
 * sends with stream:true) into { done, token, dataLine }. Empty lines, ":" keep-alive comments,
 * and other non-data SSE fields are protocol noise and yield no token; a "data:" payload that is
 * neither [DONE] nor valid JSON is an error.
 *
 * `dataLine` (false for every non-data line) is what lets the caller distinguish "the proxy
 * answered 200 with some non-SSE body" (zero data lines over the whole stream) from "a legal
 * SSE stream that simply produced no content" -- see assertSseStreamSawData below.
 *
 * This is the STATELESS entry point: each line is treated as a complete single-"data:"-line
 * event, which is what current proxies actually send (and how the test-suite drives it). A live
 * stream that must also handle the spec-legal multi-"data:"-line event (one JSON payload split
 * over several data: lines) must instead feed every line of the stream into ONE SSEParser
 * instance and call its flush() at the end.
 */
function parseSseLine(line) {
  if (line.trim() === "") return { done: false, token: "", dataLine: false };
  // SSE comments (": keep-alive", ":::") and any other non-data field (event:, id:...) --
  // both are legitimate protocol noise, never content.
  if (line.startsWith(":") || !line.startsWith("data:")) return { done: false, token: "", dataLine: false };
  const data = line.slice("data:".length).trim();
  if (data === "[DONE]") return { done: true, token: "", dataLine: true };
  let obj;
  try {
    obj = JSON.parse(data);
  } catch (err) {
    throw new Error(`LiteLLM: invalid JSON in an SSE data payload (near: ${data.slice(0, 120)}) -- is the proxy really OpenAI-compatible?`);
  }
  return { done: false, token: sseTokenOf(obj), dataLine: true };
}

/**
 * Stateful parser for a full SSE stream: accumulates consecutive "data:" lines of ONE event,
 * joins them with "\n", and parses the reassembled payload when the event closes (an empty
 * line, or the terminal [DONE]). [DONE] is special-cased before the accumulation so a bare
 * "data: [DONE]" event always terminates, even if a previous event's data lines are still open.
 * Hold ONE instance per stream; the accumulation state must not be shared between streams.
 */
class SSEParser {
  constructor() {
    this._pending = null; // accumulated data payload of the event in progress, or null
  }

  parseLine(line) {
    if (line.trim() === "") {
      // An empty line closes the current event: parse what was accumulated (if anything).
      return this._closeEvent();
    }
    if (line.startsWith(":") || !line.startsWith("data:")) return { done: false, token: "", dataLine: false };
    const data = line.slice("data:".length).trim();
    if (data === "[DONE]") {
      // Terminal marker: whatever was pending is abandoned -- a well-formed proxy never mixes
      // [DONE] into a partial event, and being strict here only protects against truncation.
      this._pending = null;
      return { done: true, token: "", dataLine: true };
    }
    this._pending = this._pending === null ? data : this._pending + "\n" + data;
    return { done: false, token: "", dataLine: true };
  }

  _closeEvent() {
    const data = this._pending;
    this._pending = null;
    if (data === null) return { done: false, token: "", dataLine: false };
    let obj;
    try {
      obj = JSON.parse(data);
    } catch (err) {
      throw new Error(`LiteLLM: invalid JSON in an SSE data payload (near: ${data.slice(0, 120)}) -- is the proxy really OpenAI-compatible?`);
    }
    return { done: false, token: sseTokenOf(obj), dataLine: true };
  }

  /** End-of-stream: a trailing event that never received its closing empty line is still an event. */
  flush() {
    return this._closeEvent();
  }
}

function sseTokenOf(obj) {
  // choices can be an empty array (or absent) on some events -- not an error, just no token.
  const choice = obj && Array.isArray(obj.choices) ? obj.choices[0] : null;
  const content = choice && choice.delta ? choice.delta.content : undefined;
  // Several models send delta.content:null between real tokens -- that is not a token either,
  // and treating null as content would have shown "[object Object]" to the user.
  return typeof content === "string" ? content : "";
}

/**
 * A proxy that ignored stream:true (or an HTML error page from a broken SSH tunnel) still
 * answers HTTP 200, and every line of that body is classified as "SSE noise" -- zero tokens,
 * zero error, and the caller would present an empty string as a finished draft. If NO data line
 * was seen over the whole stream, the body was not an SSE stream at all: say so, loudly, with
 * a slice of the body the user can actually act on.
 */
function assertSseStreamSawData(dataLinesSeen, fullBody) {
  if (dataLinesSeen === 0) {
    throw new Error(
      `The LiteLLM proxy returned a 200 response that is not an SSE stream (near: ${fullBody.slice(0, 120).trimEnd() || "(empty body)"}) -- is the SSH tunnel / proxy healthy?`
    );
  }
}

/**
 * Decides what to do when a streamed request fails:
 * - "abort"    -- the user (or the port disconnecting) cancelled it: total silence, and NO
 *                 fallback, because re-running an LLM call the user just asked to stop would be
 *                 exactly the surprise they were trying to avoid;
 * - "fallback" -- nothing was shown yet, so retrying the identical non-streaming call loses
 *                 nothing and may still deliver a complete draft;
 * - "error"    -- tokens are already on screen: a fallback would duplicate the partial text, so
 *                 keep what the user saw and simply report the failure.
 */
function nextStepOnError(err, tokensEmitted) {
  if (err && err.name === "AbortError") return "abort";
  if (tokensEmitted === 0) return "fallback";
  return "error";
}

// Classic-script export so tests/stream.test.mjs can load this file under Node; inside the
// extension `module` is undefined, so the names stay plain globals of the background script.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { StreamLineBuffer, parseOllamaLine, parseSseLine, SSEParser, assertSseStreamSawData, nextStepOnError };
}
