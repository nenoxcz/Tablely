"""The single page served by ``tablely ui`` (no external files or CDNs)."""

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="tablely-token" content="__TOKEN__">
<title>Tablely</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #667085; --line: #e4e7ec;
  --track: #eef0f3; --accent: #3d63dd; --ok: #1f9d55; --fail: #d64545; --run: #3d63dd;
  --wait: #98a2b3; --warn: #c27a00; --on-accent: #ffffff; --shadow: 0 1px 2px rgba(16, 24, 40, .06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111418; --card: #1a1f26; --text: #e6e9ee; --muted: #98a2b3; --line: #2a313b;
    --track: #262d36; --accent: #7b9cff; --ok: #3ccf7e; --fail: #ff6b6b; --run: #7b9cff;
    --wait: #6b7484; --warn: #f0b429; --on-accent: #0b1020; --shadow: none;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; padding: 16px 20px;
  border-bottom: 1px solid var(--line); background: var(--card); position: sticky; top: 0; z-index: 1; }
.brand { font-weight: 700; font-size: 18px; letter-spacing: -.01em; }
.machine, .updated { color: var(--muted); font-size: 13px; }
.updated { margin-left: auto; }
main { max-width: 1200px; margin: 0 auto; padding: 8px 20px 40px; }
h2 { font-size: 15px; margin: 22px 0 10px; display: flex; gap: 8px; align-items: baseline; }
h2 .sub { color: var(--muted); font-weight: 400; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px;
  box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.card.ended { opacity: .82; }
.head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.title { font-weight: 600; overflow-wrap: anywhere; }
.meta { color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
.pct { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { height: 8px; background: var(--track); border-radius: 99px; overflow: hidden; }
.bar.thin { height: 4px; }
.bar > span { display: block; height: 100%; background: var(--accent); border-radius: 99px; transition: width .4s ease; }
.bar > span.ok { background: var(--ok); } .bar > span.failed { background: var(--fail); }
.bar > span.cancelled { background: var(--wait); } .bar > span.warn { background: var(--warn); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 12px; padding: 1px 8px; border-radius: 99px; background: var(--track); color: var(--muted); }
.chip.ok { color: var(--ok); } .chip.failed { color: var(--fail); } .chip.running { color: var(--run); }
ul { list-style: none; margin: 0; padding: 0; }
.jobs { display: flex; flex-direction: column; gap: 8px; }
.row { display: flex; align-items: center; gap: 8px; min-width: 0; }
.name { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
.small-pct { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.state { font-size: 11px; padding: 0 6px; border-radius: 4px; border: 1px solid currentColor; white-space: nowrap; }
.s-ok { color: var(--ok); } .s-failed { color: var(--fail); } .s-running { color: var(--run); }
.s-waiting, .s-cancelled { color: var(--wait); } .s-working { color: var(--run); } .s-idle { color: var(--ok); }
.s-stale, .s-ended { color: var(--wait); }
.detail { color: var(--muted); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
.todos li { display: flex; gap: 8px; font-size: 13px; padding: 1px 0; }
.todos .mark { width: 16px; text-align: center; flex: none; }
.todos .completed { color: var(--muted); text-decoration: line-through; }
.quote { font-size: 13px; color: var(--muted); border-left: 3px solid var(--line); padding-left: 8px;
  overflow-wrap: anywhere; max-height: 7.5em; overflow: hidden; }
.actions { display: flex; justify-content: flex-end; margin-top: auto; }
button { font: inherit; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--text);
  padding: 6px 14px; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); font-weight: 600; }
button:hover { filter: brightness(1.05); } button:disabled { opacity: .6; cursor: progress; }
.agents { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.notes li { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
.notes .who { font-weight: 600; margin-right: 6px; } .notes .when { color: var(--muted); font-size: 12px; margin-right: 6px; }
.empty { color: var(--muted); padding: 18px; text-align: center; border: 1px dashed var(--line); border-radius: 10px; }
dialog { border: 1px solid var(--line); border-radius: 12px; background: var(--card); color: var(--text);
  width: min(820px, 94vw); padding: 18px; }
dialog::backdrop { background: rgba(0, 0, 0, .45); }
dialog h3 { margin: 0 0 6px; font-size: 16px; }
#resume-status { color: var(--muted); font-size: 13px; margin: 0 0 10px; overflow-wrap: anywhere; }
#resume-status.error { color: var(--fail); }
#resume-text { width: 100%; height: 50vh; font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 10px; resize: vertical; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px; }
@media (max-width: 420px) { .grid, .agents { grid-template-columns: 1fr; } main, header { padding-left: 12px; padding-right: 12px; } }
</style>
</head>
<body>
<header>
  <div class="brand">Tablely</div>
  <div class="machine" id="machine">불러오는 중…</div>
  <div class="updated" id="updated"></div>
</header>
<main>
  <h2>에이전트 <span class="sub" id="agents-sub"></span></h2>
  <div class="agents" id="agents"></div>

  <h2>학습 작업 <span class="sub" id="runs-sub"></span></h2>
  <div class="grid" id="runs"></div>

  <section id="sessions-section" hidden>
    <h2>코딩 세션 <span class="sub" id="sessions-sub"></span></h2>
    <div class="grid" id="sessions"></div>
  </section>

  <h2>메모</h2>
  <ul class="notes" id="notes"></ul>
</main>

<dialog id="resume-dialog">
  <h3 id="resume-title">재개 요약</h3>
  <p id="resume-status"></p>
  <textarea id="resume-text" readonly></textarea>
  <div class="dialog-actions">
    <button id="copy-btn">복사</button>
    <button class="primary" id="close-btn">닫기</button>
  </div>
</dialog>

<script>
"use strict";
const TOKEN = document.querySelector('meta[name="tablely-token"]').content;
const STATE_KO = { ok: "완료", failed: "실패", cancelled: "취소", running: "실행 중", waiting: "대기",
                   working: "작업 중", idle: "응답 끝", ended: "종료", stale: "오래됨" };
const $ = (id) => document.getElementById(id);

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
const shortPaths = (text) => String(text).replace(/(?:\/[^\s\/]+)+\/([^\s\/]+)/g, "$1");  // full path on hover
const pct = (f) => (f === null || f === undefined) ? "–" : Math.round(f * 100) + "%";
const clock = (t) => t ? new Date(t * 1000).toLocaleString("ko-KR", { month: "numeric", day: "numeric",
  hour: "2-digit", minute: "2-digit" }) : "?";
function bar(fraction, kind, thin) {
  const width = Math.max(0, Math.min(1, fraction || 0)) * 100;
  return el("div", { class: "bar" + (thin ? " thin" : "") }, el("span", { class: kind || "", style: `width:${width}%` }));
}
function empty(text) { return el("div", { class: "empty" }, text); }
function resumeButton(kind, id, label) {
  return el("div", { class: "actions" },
    el("button", { class: "primary", onclick: (e) => resume(kind, id, label, e.currentTarget) }, "재개하기"));
}

function jobItem(j) {
  const fraction = j.state === "ok" ? 1 : j.fraction;
  const detailBits = [];
  if (j.progress) detailBits.push(j.progress);
  if (j.state === "running" && j.placement) detailBits.push(j.placement);
  if (["waiting", "failed", "cancelled"].includes(j.state) && j.detail) detailBits.push(j.detail);
  if (j.switches) detailBits.push(`CPU/GPU 이동 ${j.switches}회`);
  return el("li", {},
    el("div", { class: "row" },
      el("span", { class: "state s-" + j.state }, STATE_KO[j.state] || j.state),
      el("span", { class: "name", title: j.task || "" }, j.name),
      el("span", { class: "small-pct" }, j.state === "waiting" ? "" : pct(fraction))),
    bar(fraction, j.state === "running" ? "" : j.state, true),
    detailBits.length ? el("div", { class: "detail", title: detailBits.join(" · ") },
      shortPaths(detailBits.join(" · "))) : null);
}

function runCard(r) {
  const c = r.counts;
  const chips = [["ok", "완료"], ["failed", "실패"], ["running", "실행 중"], ["waiting", "대기"], ["cancelled", "취소"]]
    .filter(([k]) => c[k]).map(([k, label]) => el("span", { class: "chip " + k }, `${label} ${c[k]}`));
  const when = r.live ? `${clock(r.started)} 시작 · 진행 중` : `${clock(r.ended)} 종료`;
  return el("article", { class: "card" + (r.live ? "" : " ended") },
    el("div", { class: "head" },
      el("div", {}, el("div", { class: "title" }, r.task || `run ${r.run}`),
        el("div", { class: "meta" }, `${r.agent} · ${when} · run ${r.run}`)),
      el("div", { class: "pct" }, pct(r.achievement))),
    bar(r.achievement, r.live ? "" : (c.failed ? "warn" : "ok")),
    el("div", { class: "chips" }, chips),
    el("ul", { class: "jobs" }, r.jobs.map(jobItem)),
    resumeButton("run", r.run, r.task || r.run));
}

function agentCard(a) {
  return el("article", { class: "card" },
    el("div", { class: "head" },
      el("div", {}, el("div", { class: "title" }, a.name),
        el("div", { class: "meta" }, `실행 ${a.runs}개` + (a.live_runs ? ` · 진행 중 ${a.live_runs}` : ""))),
      el("div", { class: "pct" }, pct(a.achievement))),
    bar(a.achievement),
    a.note ? el("div", { class: "quote" }, a.note.detail) : null,
    resumeButton("agent", a.name, a.name));
}

function sessionCard(s) {
  const marks = { completed: "✓", in_progress: "◐" };
  return el("article", { class: "card" + (s.status === "ended" ? " ended" : "") },
    el("div", { class: "head" },
      el("div", {}, el("div", { class: "title" }, s.task || `세션 ${s.short}`),
        el("div", { class: "meta" }, `${s.short} · ${s.branch || "?"} · ${s.updated_ago}`)),
      el("div", { class: "pct" }, s.total ? pct(s.achievement) : "–")),
    s.total ? bar(s.achievement) : null,
    el("div", { class: "chips" },
      el("span", { class: "chip s-" + s.status }, STATE_KO[s.status] || s.status),
      s.total ? el("span", { class: "chip" }, `할 일 ${s.done}/${s.total}`) : null),
    s.todos.length ? el("ul", { class: "todos" }, s.todos.map((t) =>
      el("li", {}, el("span", { class: "mark" }, marks[t.status] || "○"),
        el("span", { class: t.status === "completed" ? "completed" : "" }, t.subject)))) : null,
    s.handoff ? el("div", { class: "quote" }, "핸드오프: " + s.handoff)
      : (s.last_reply ? el("div", { class: "quote" }, s.last_reply) : null),
    s.files.length ? el("div", { class: "detail" }, "파일: " + s.files.join(", ")) : null,
    resumeButton("session", s.session, s.task || s.short));
}

function render(state) {
  const pool = state.pool;
  $("machine").textContent = pool
    ? `CPU 코어 ${pool.cpus.length}개 · GPU ${pool.gpus.length ? pool.gpus.join(", ") : "없음"} · ${pool.backfill ? "backfill" : "엄격한 우선순위"}`
    : "지금 실행 중인 작업 없음";
  $("updated").textContent = "갱신 " + new Date(state.generated_at * 1000).toLocaleTimeString("ko-KR");

  $("agents-sub").textContent = `최근 ${state.hours}시간`;
  $("agents").replaceChildren(...(state.agents.length ? state.agents.map(agentCard) : [empty("기록된 에이전트가 없습니다")]));

  const live = state.runs.filter((r) => r.live).length;
  $("runs-sub").textContent = `진행 중 ${live} · 종료 ${state.runs.length - live}`;
  $("runs").replaceChildren(...(state.runs.length ? state.runs.map(runCard)
    : [empty("최근 실행이 없습니다. tablely run <작업 파일>로 시작하세요")]));

  $("sessions-section").hidden = !state.repo;
  $("sessions-sub").textContent = state.repo ? state.repo.replace(/\/+$/, "").split("/").pop() : "";
  $("sessions-sub").title = state.repo || "";
  $("sessions").replaceChildren(...(state.sessions.length ? state.sessions.map(sessionCard) : [empty("기록된 세션이 없습니다")]));

  $("notes").replaceChildren(...(state.notes.length ? state.notes.map((n) => el("li", {},
    el("span", { class: "who" }, n.agent || "?"), el("span", { class: "when" }, clock(n.t)), n.detail))
    : [el("li", { class: "empty" }, "메모가 없습니다")]));
}

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    render(await response.json());
  } catch (err) {
    $("updated").textContent = "연결 끊김 — 다시 시도 중";
  }
}

async function resume(kind, id, label, button) {
  const dialog = $("resume-dialog");
  $("resume-title").textContent = `재개 요약 — ${label}`;
  $("resume-status").className = "";
  $("resume-status").textContent = "지금까지 한 일을 정리하는 중…";
  $("resume-text").value = "";
  if (!dialog.open) dialog.showModal();
  button.disabled = true;
  try {
    const response = await fetch("/api/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Tablely-Token": TOKEN },
      body: JSON.stringify({ kind, id }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    $("resume-text").value = result.prompt;
    let status = `저장됨: ${result.prompt_file}`;
    if (result.launched) status = `AI를 시작했습니다: ${result.command} (pid ${result.pid}, 출력 ${result.log}) · ` + status;
    else if (result.error) { status = `AI를 시작하지 못했습니다: ${result.error} · ` + status; $("resume-status").className = "error"; }
    else status = "아래 요약을 복사해서 AI에게 붙여넣으세요 · " + status;
    $("resume-status").textContent = status;
  } catch (err) {
    $("resume-status").className = "error";
    $("resume-status").textContent = "요약을 만들지 못했습니다: " + err.message;
  } finally {
    button.disabled = false;
  }
}

$("close-btn").addEventListener("click", () => $("resume-dialog").close());
$("copy-btn").addEventListener("click", async () => {
  const text = $("resume-text").value;
  try { await navigator.clipboard.writeText(text); }
  catch { $("resume-text").select(); document.execCommand("copy"); }
  $("copy-btn").textContent = "복사됨";
  setTimeout(() => { $("copy-btn").textContent = "복사"; }, 1500);
});

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
