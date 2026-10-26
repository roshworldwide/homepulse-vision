"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LABEL = { ALLOWED: "ALLOWED", BLOCKED: "BLOCKED", ASKED_FAMILY: "ASKED THE FAMILY" };
const MAX_BYTES = 4 * 1024 * 1024;
let approvalToken = null;
let lastRun = null;

async function init() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("stack").innerHTML = [`OpenCV ${h.opencv}`, h.planner.replace("bedrock:", "Bedrock · "), "camera: simulated", "lock: simulated"]
      .map((x) => `<span>${esc(x)}</span>`).join("");
  } catch { $("stack").textContent = ""; }
  const list = await (await fetch("/api/scenarios")).json();
  $("scenarios").innerHTML = list.map((s) => `
    <button class="sc" role="listitem" data-id="${esc(s.id)}">
      <img src="/api/scenarios/${encodeURIComponent(s.id)}/thumb.jpg" alt="" loading="lazy" width="96" height="72">
      <span><b>${esc(s.title)}</b><span class="blurb">${esc(s.blurb)}</span><br>
      <span class="exp ${esc(s.expect)}">expect ${esc(LABEL[s.expect] || s.expect)}</span></span>
    </button>`).join("");
  $("scenarios").addEventListener("click", (e) => {
    const b = e.target.closest(".sc");
    if (!b) return;
    document.querySelectorAll(".sc").forEach((x) => x.classList.toggle("active", x === b));
    history.replaceState(null, "", "#scenario=" + b.dataset.id);
    run({ scenario: b.dataset.id });
  });
  $("file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $("rawToggle").addEventListener("change", (e) => { $("raw").hidden = !e.target.checked; $("trace").hidden = e.target.checked; });
  $("approve").addEventListener("click", () => familyReply(true));
  $("deny").addEventListener("click", () => familyReply(false));
  document.addEventListener("click", (e) => {
    if (e.target.matches(".view img")) {
      const z = document.createElement("div");
      z.className = "zoom";
      z.innerHTML = `<img src="${e.target.src}" alt="${esc(e.target.alt)}">`;
      z.addEventListener("click", () => z.remove());
      document.body.appendChild(z);
    }
  });
  const m = location.hash.match(/scenario=([a-z_]+)/);
  if (m) {
    const b = document.querySelector(`.sc[data-id="${m[1]}"]`);
    if (b) b.click();
  }
}

function upload(file) {
  if (file.size > MAX_BYTES) { showError("Image is larger than 4 MB."); return; }
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { showError("Use a JPEG, PNG or WebP image."); return; }
  document.querySelectorAll(".sc").forEach((x) => x.classList.remove("active"));
  const r = new FileReader();
  r.onload = () => run({ image_b64: r.result });
  r.readAsDataURL(file);
}

function showError(msg) {
  $("busy").hidden = true; $("empty").hidden = false;
  $("empty").innerHTML = `<p class="err">${esc(msg)}</p>`;
}

async function run(body) {
  $("empty").hidden = true; $("result").hidden = true; $("busy").hidden = false;
  const steps = ["capture_snapshot", "analyze_frame (OpenCV 5)", "Nova 2 Lite plans the next step", "guard check"];
  let i = 0;
  const tick = setInterval(() => { $("busytext").textContent = steps[i++ % steps.length]; }, 900);
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) {
      showError(res.status === 429 ? `Rate limited (${data.scope}). Try again in ${data.retry_after_s}s.` : (data.error || `Error ${res.status}`));
      return;
    }
    lastRun = data;
    render(data);
  } catch (e) {
    showError("Request failed: " + e.message);
  } finally {
    clearInterval(tick);
    $("busy").hidden = true;
  }
}

function render(d) {
  $("result").hidden = false;
  const steps = d.trace.filter((e) => e.type === "tool_call").length;
  const exp = d.expected ? `<span>expected: ${esc(LABEL[d.expected])} ${d.expected === d.decision ? "✓" : "✗"}</span>` : "";
  $("decision").className = "decision " + d.decision;
  $("decision").innerHTML = `
    <div class="big">${esc(LABEL[d.decision] || d.decision)}</div>
    <div class="why">${esc(d.reason)}</div>
    ${d.summary ? `<div class="summary">Agent: “${esc(d.summary)}”</div>` : ""}
    <div class="meta"><span>${steps} tool calls</span><span>${(d.wall_ms / 1000).toFixed(1)} s</span>
    <span>${d.usage.inputTokens + d.usage.outputTokens} tokens</span><span>guard refusals: ${d.trace.filter((e) => e.type === "guard" && !e.allowed).length}</span>${exp}</div>`;
  $("views").innerHTML = d.views.map((v) => {
    const q = v.quality;
    const chips = [
      `<span class="chip">blur ${q.blur_effect.toFixed(3)}</span>`,
      `<span class="chip">Laplacian var ${Math.round(q.laplacian_var_edges)}</span>`,
      `<span class="chip">luma ${Math.round(q.mean_luma)}</span>`,
      `<span class="chip">backlight ${q.backlight_score.toFixed(2)}</span>`,
      `<span class="chip">watermark ${(q.watermark_fraction * 100).toFixed(1)}%</span>`,
      q.issues.length ? q.issues.map((x) => `<span class="chip bad">${esc(x)}</span>`).join("") : `<span class="chip good">trustworthy</span>`,
    ].join("");
    return `<figure class="view"><img src="data:image/jpeg;base64,${v.jpeg_b64}" alt="${esc(v.frame_id)} with OpenCV overlays">
      <figcaption class="cap"><b>${esc(v.frame_id)}</b> · ${esc(v.origin)}<div class="chips">${chips}</div></figcaption></figure>`;
  }).join("");
  $("trace").innerHTML = traceRows(d.trace).join("");
  $("raw").textContent = d.trace.map((e) => JSON.stringify(e)).join("\n");
  approvalToken = d.family_approval ? d.family_approval.token : null;
  $("familyPanel").hidden = !approvalToken;
  $("familyOut").innerHTML = "";
  $("approve").disabled = $("deny").disabled = false;
  if (approvalToken) {
    const r = d.family_approval.request;
    $("familyMsg").innerHTML = `<b>HomePulse:</b> ${esc(r.message)} <span class="muted small">(${esc(r.reason)})</span>`;
  }
}

function because(ev) {
  const e = ev.evidence;
  if (!e) return "";
  const parts = [];
  if (e.quality_issues && e.quality_issues.length) parts.push(`quality: ${e.quality_issues.join(", ")}`);
  if (e.qr_located_not_decoded) parts.push(`QR located but not decoded ×${e.qr_located_not_decoded}`);
  if (e.qr_decoded_on) parts.push(`QR decoded on ${e.qr_decoded_on} (${e.frame_origin})`);
  if (e.instruction_like_text) parts.push(`instruction-like text: ${e.matched_phrases.join(", ")}`);
  if (e.invalid_pass_reasons && e.invalid_pass_reasons.length) parts.push(`pass rejected: ${e.invalid_pass_reasons.join(", ")}`);
  if (ev.name === "ask_family" && e.faces) parts.push(`${e.faces} face(s) present`);
  if (e.verifications) parts.push(`verifications: ${JSON.stringify(e.verifications)}`);
  return parts.length ? `<div class="because">because ${esc(parts.join(" · "))}</div>` : "";
}

function resultText(name, o) {
  if (!o) return "";
  if (o.error) return `<span class="err">error: ${esc(o.error)}</span>`;
  switch (name) {
    case "capture_snapshot": return `${esc(o.frame_id)} from ${esc(o.source)}`;
    case "request_new_snapshot": return o.available ? `new frame ${esc(o.frame_id)}` : esc(o.message);
    case "improve_view": return `new frame ${esc(o.frame_id)} · ${esc(o.op)} ${esc(JSON.stringify(o.params))}`;
    case "analyze_frame": {
      const q = o.quality, t = o.visible_text;
      const qr = o.qr_codes.length ? o.qr_codes.map((c) => c.looks_like_homepulse_pass ? `${c.qr_ref}: HomePulse pass` : `${c.qr_ref}: <span class="untrusted">${esc(c.untrusted_content)}</span>`).join(", ")
        : (o.qr_located_but_not_decoded ? `${o.qr_located_but_not_decoded} located, not decoded` : "none");
      return `quality <b>${q.trustworthy ? "trustworthy" : esc(q.issues.join(", "))}</b> (blur ${q.blur_effect}, luma ${q.mean_luma}) · faces ${o.people.faces_detected} · QR: ${qr}` +
        (t.text ? ` · text <span class="untrusted">${esc(t.text)}</span>${t.instruction_like ? " <b>instruction-like</b>" : ""}` : "");
    }
    case "verify_pass": return o.valid ? `<b>valid</b> (pass ${esc(o.pid_short)}…, display age ${o.display_age_s}s)` : `<b>invalid: ${esc(o.reason)}</b>`;
    case "ask_family": return `${esc(o.status)} · ${esc(o.request_id)}`;
    case "unlock_door": return o.unlocked ? `unlocked (${esc(o.basis)})` : `refused: ${esc(o.refused)}`;
    default: return esc(JSON.stringify(o));
  }
}

function traceRows(events) {
  const rows = [];
  let n = 0;
  const results = {};
  events.forEach((e) => { if (e.type === "tool_result") results[e.seq - 1] = e; });
  for (const e of events) {
    let k = "", t = "", d = "", ms = "", cls = "";
    if (e.type === "run_start") { k = "start"; t = `planner ${e.planner}`; d = `camera: ${e.camera} · lock: ${e.lock}`; }
    else if (e.type === "model_call") {
      k = "model"; t = e.tool_calls.length ? `Nova 2 Lite → ${e.tool_calls.join(", ")}` : "Nova 2 Lite → final message";
      d = e.text ? esc(e.text) : ""; ms = `${Math.round(e.latency_ms)} ms · ${e.tokens.in ?? "?"}/${e.tokens.out ?? "?"} tok`;
      rows.push(row(++n, k, t, d, "", ms, "")); continue;
    } else if (e.type === "tool_call") {
      const r = results[e.seq] || events.find((x) => x.type === "tool_result" && x.seq > e.seq && x.name === e.name);
      k = e.name === "analyze_frame" ? "perception" : "tool";
      const args = Object.keys(e.input || {}).length ? `(${esc(Object.entries(e.input).map(([a, b]) => `${a}=${typeof b === "string" && b.length > 60 ? b.slice(0, 60) + "…" : b}`).join(", "))})` : "()";
      t = `${esc(e.name)}${args}`;
      d = r ? resultText(e.name, r.output) : "";
      ms = r ? `${Math.round(r.latency_ms)} ms` : "";
      rows.push(row(++n, k, t, d, because(e), ms, "")); continue;
    } else if (e.type === "tool_result") { continue; }
    else if (e.type === "guard") {
      k = "guard"; cls = e.allowed ? "guard-ok" : "guard-no";
      t = e.allowed ? `unlock allowed: ${e.basis}` : "unlock REFUSED";
      d = esc(e.allowed ? JSON.stringify(e.checks) : e.refused_reason) + (e.model_reason ? ` · model said: “${esc(e.model_reason)}” (not consulted)` : "");
    } else if (e.type === "decision") { k = "decision"; t = `${LABEL[e.decision] || e.decision}`; d = esc(e.reason); }
    else if (e.type === "family_reply") { k = "family"; t = e.approve ? "family approved" : "family declined"; d = esc(e.channel); }
    else if (["safety_net", "model_error", "step_limit"].includes(e.type)) { k = "warn"; t = e.type.replace("_", " "); d = esc(e.why || e.error || ""); }
    else { k = e.type; }
    rows.push(row(++n, k, esc(t), d, "", ms, cls));
  }
  return rows;
}

function row(n, k, t, d, b, ms, cls) {
  return `<li class="ev ${cls}"><span class="n">${n}</span><div><span class="k ${k}">${esc(k)}</span><span class="t">${t}</span>
    ${d ? `<div class="d">${d}</div>` : ""}${b}</div><span class="ms">${esc(ms)}</span></li>`;
}

async function familyReply(approve) {
  if (!approvalToken) return;
  $("approve").disabled = $("deny").disabled = true;
  const res = await fetch("/api/family", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approval_token: approvalToken, approve }) });
  const data = await res.json();
  if (!res.ok) { $("familyOut").innerHTML = `<span class="err">${esc(data.error || res.status)}</span>`; return; }
  approvalToken = null;
  $("familyOut").innerHTML = `Result: <b>${esc(LABEL[data.decision])}</b>`;
  lastRun.trace = lastRun.trace.concat(data.trace.map((e) => ({ ...e, seq: 10000 + e.seq })));
  $("trace").innerHTML = traceRows(lastRun.trace).join("");
  $("raw").textContent = lastRun.trace.map((e) => JSON.stringify(e)).join("\n");
  $("decision").className = "decision " + data.decision;
  $("decision").querySelector(".big").textContent = `${LABEL[data.decision]} (family ${approve ? "approved" : "declined"})`;
}

init();
