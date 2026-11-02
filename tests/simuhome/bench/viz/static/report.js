/* The report page: tabs (overview, one per config, episodes), inline-SVG charts, and the
   episode viewer. Reads the aggregates from <script id="data"> and each record from its
   own <script class="rec"> block, parsed only when opened. No external code. */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const CONFIGS = DATA.configs;
  const PHASES = DATA.phases;
  const CAT_NAMES = DATA.categories.map(c => c[0]);
  const CAT_HOW = Object.fromEntries(DATA.categories);
  const app = document.getElementById("app");
  const recordCache = new Map();
  let currentList = [];            // episode ids of the list the panel was opened from

  // ---------------------------------------------------------------- helpers
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v);
    }
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return el;
  }
  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVGNS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
    for (const c of children.flat(Infinity)) if (c != null) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return el;
  }
  const isNum = v => typeof v === "number" && isFinite(v);
  const pct = v => isNum(v) ? Math.round(v * 100) + "%" : "–";
  const pct1 = v => isNum(v) ? (v * 100).toFixed(1) + "%" : "–";
  const num = (v, d = 1) => isNum(v) ? v.toFixed(d).replace(/\.0+$/, "") : (v == null ? "–" : String(v));
  const int = v => isNum(v) ? Math.round(v).toLocaleString("en-US") : (v == null ? "–" : String(v));
  const secs = v => isNum(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " s" : "–";
  const compact = v => !isNum(v) ? "–" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 10000 ? (v / 1e3).toFixed(0) + "K" : v >= 1000 ? (v / 1e3).toFixed(1) + "K" : v.toFixed(v < 10 ? 1 : 0);
  const clip = (t, n = 140) => { t = String(t == null ? "" : t); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
  const colorOf = cfg => cfg.slot < 0 ? "var(--ref)" : `var(--s${cfg.slot + 1})`;
  const phaseColor = i => `var(--s${(i % 8) + 1})`;
  const cfgLabel = cfg => cfg.title + (cfg.reference && !/reference/i.test(cfg.title) ? " (reference)" : "");
  const gpuShort = cfg => (cfg.gpu && cfg.gpu.gpu) ? cfg.gpu.gpu.replace(/^NVIDIA\s+/, "") : (cfg.reference ? "API" : "GPU n/a");
  const groupRow = (summary, type, kase) => summary.find(r => r.type === type && r.case === kase);
  const esc = t => String(t == null ? "" : t);
  function jsonPretty(v) { try { return JSON.stringify(v, null, 2); } catch (e) { return String(v); } }

  function tryStore(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } }
  function tryLoad(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

  // Tooltip: one element, follows the pointer, text only via textContent.
  const tip = h("div", { id: "tip" });
  document.body.appendChild(tip);
  function showTip(ev, rows) {
    tip.replaceChildren(...rows.map(r => h("div", null,
      r.color ? h("span", { class: "k", style: { background: r.color } }) : null,
      h("span", { class: "v" }, r.value), r.label ? ` ${r.label}` : "")));
    tip.style.display = "block";
    moveTip(ev);
  }
  function moveTip(ev) {
    const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.style.display = "none"; }
  function hover(el, rowsFn) {
    el.addEventListener("pointerenter", ev => showTip(ev, rowsFn()));
    el.addEventListener("pointermove", moveTip);
    el.addEventListener("pointerleave", hideTip);
    el.setAttribute("tabindex", "0");
    el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.left + r.width / 2, clientY: r.top }, rowsFn()); });
    el.addEventListener("blur", hideTip);
  }

  // ---------------------------------------------------------------- tables
  // columns: [{key, label, fmt, num, cls}] ; rows: objects. opts.onRow(row) makes rows clickable.
  function table(columns, rows, opts = {}) {
    const state = { key: opts.sortKey || null, dir: opts.sortDir || 1 };
    const wrap = h("div", { class: "tbl-wrap" });
    function render() {
      let data = rows.slice();
      if (state.key) {
        const col = columns.find(c => c.key === state.key);
        data.sort((a, b) => {
          let x = a[state.key], y = b[state.key];
          if (col && col.sortValue) { x = col.sortValue(a); y = col.sortValue(b); }
          if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
          return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })) * state.dir;
        });
      }
      const thead = h("thead", null, h("tr", null, columns.map(c => {
        const th = h("th", { class: (c.num ? "num " : "") + (opts.sortable ? "sortable" : "") }, c.label + (state.key === c.key ? (state.dir > 0 ? " ▲" : " ▼") : ""));
        if (opts.sortable) th.addEventListener("click", () => { if (state.key === c.key) state.dir = -state.dir; else { state.key = c.key; state.dir = 1; } render(); });
        return th;
      })));
      const tbody = h("tbody", null, data.map(r => {
        const tr = h("tr", { class: [(opts.rowClass ? opts.rowClass(r) : ""), opts.onRow ? "clickable" : ""].join(" ") },
          columns.map(c => {
            const v = c.fmt ? c.fmt(r[c.key], r) : r[c.key];
            const td = h("td", { class: (c.num ? "num " : "") + (c.cls || "") });
            if (v instanceof Node) td.appendChild(v); else td.textContent = v == null ? "" : String(v);
            return td;
          }));
        if (opts.onRow) tr.addEventListener("click", () => opts.onRow(r));
        return tr;
      }));
      wrap.replaceChildren(h("table", { class: "tbl" }, thead, tbody));
    }
    render();
    return wrap;
  }
  // A summary.md display table (rows as {header: value}), as stats.tables() emits them.
  function mdTable(rows) {
    if (!rows || !rows.length) return h("p", { class: "note" }, "(none)");
    const keys = [...new Set(rows.flatMap(r => Object.keys(r)))];
    const columns = keys.map(k => ({ key: k, label: k, num: k !== "group" && k !== "bucket", fmt: v => v == null ? "" : String(v) }));
    return table(columns, rows, { rowClass: r => (r.group && /^(all |overall)/.test(r.group)) || r.bucket === "all" ? "total" : (r.group && !/ (feasible|infeasible|mixed)$/.test(r.group) ? "group-sep" : "") });
  }

  // ---------------------------------------------------------------- charts
  function figure(title, sub, svg, legendSeries, tableNode) {
    const fig = h("figure", { class: "chart" });
    const tools = h("div", { class: "fig-tools" });
    const btn = h("button", { type: "button", title: "Show the same numbers as a table" }, "Table");
    btn.addEventListener("click", () => { fig.classList.toggle("show-table"); btn.textContent = fig.classList.contains("show-table") ? "Chart" : "Table"; });
    tools.appendChild(btn);
    fig.appendChild(h("div", { class: "fig-head" }, h("span", { class: "fig-title" }, title), sub ? h("span", { class: "fig-sub" }, sub) : null, tools));
    if (legendSeries && legendSeries.length > 1) {
      fig.appendChild(h("div", { class: "legend" }, legendSeries.map(se => h("span", null, h("span", { class: "key" + (se.line ? " line" : ""), style: { background: se.color } }), se.name))));
    }
    fig.appendChild(svg);
    const tw = h("div", { class: "chart-table" }, tableNode);
    fig.appendChild(tw);
    return fig;
  }
  function niceMax(v) {
    if (!isNum(v) || v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const m = v / p;
    const step = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
    return step * p;
  }
  function ticks(max, n = 4) { const out = []; for (let i = 0; i <= n; i++) out.push(max * i / n); return out; }

  // Grouped columns: groups [{label, values: {seriesKey: v}}], series [{key, name, color}].
  function groupedColumns(opts) {
    const { groups, series, fmt = pct, max: forcedMax, title, sub, tableColumns } = opts;
    const W = 760, left = 44, right = 12, top = 10, bottom = 34;
    const H = 230, ph = H - top - bottom, pw = W - left - right;
    const gw = pw / Math.max(groups.length, 1);
    const bw = Math.min(24, Math.max(6, (gw - 12) / Math.max(series.length, 1) - 2));
    const maxV = forcedMax != null ? forcedMax : niceMax(Math.max(0, ...groups.flatMap(g => series.map(se => g.values[se.key] || 0))));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    for (const t of ticks(maxV)) {
      const y = top + ph - (t / maxV) * ph;
      svg.appendChild(s("line", { class: "grid", x1: left, x2: W - right, y1: y, y2: y }));
      svg.appendChild(s("text", { class: "axis-label", x: left - 6, y: y + 3.5, "text-anchor": "end" }, fmt(t)));
    }
    svg.appendChild(s("line", { class: "axis", x1: left, x2: W - right, y1: top + ph, y2: top + ph }));
    groups.forEach((g, gi) => {
      const gx = left + gi * gw;
      const total = series.length * (bw + 2) - 2;
      let x = gx + (gw - total) / 2;
      svg.appendChild(s("text", { class: "axis-label", x: gx + gw / 2, y: top + ph + 16, "text-anchor": "middle" }, g.label));
      if (g.sub) svg.appendChild(s("text", { class: "axis-label", x: gx + gw / 2, y: top + ph + 28, "text-anchor": "middle" }, g.sub));
      series.forEach(se => {
        const v = g.values[se.key];
        if (isNum(v)) {
          const bh = Math.max(0, (v / maxV) * ph);
          const y = top + ph - bh;
          const r = Math.min(4, bh / 2);
          const d = `M${x},${top + ph} V${y + r} a${r},${r} 0 0 1 ${r},-${r} h${bw - 2 * r} a${r},${r} 0 0 1 ${r},${r} V${top + ph} Z`;
          const mark = s("path", { class: "mark", d, style: `fill:${se.color}` });
          svg.appendChild(mark);
          const hit = s("rect", { class: "hit", x: x - 1, y: top, width: bw + 2, height: ph });
          hover(hit, () => [{ value: fmt(v), label: se.name, color: se.color }, { value: g.label + (g.sub ? " " + g.sub : ""), label: g.n != null && g.n[se.key] != null ? `(n=${g.n[se.key]})` : "" }]);
          svg.appendChild(hit);
        } else {
          svg.appendChild(s("text", { class: "axis-label", x: x + bw / 2, y: top + ph - 4, "text-anchor": "middle" }, "–"));
        }
        x += bw + 2;
      });
    });
    const tcols = [{ key: "label", label: tableColumns && tableColumns.groupLabel || "group" }, ...series.map(se => ({ key: se.key, label: se.name, num: true, fmt: v => fmt(v) }))];
    const trows = groups.map(g => ({ label: g.label + (g.sub ? " " + g.sub : ""), ...g.values }));
    return figure(title, sub, svg, series, table(tcols, trows));
  }

  // Horizontal stacked bars: rows [{label, sub, values: {key: v}}], series [{key, name, color}].
  function stackedBarsH(opts) {
    const { rows, series, fmt = secs, title, sub, totalLabel } = opts;
    const W = 760, left = 230, right = 70, top = 8, rowH = 30, bottom = 26;
    const H = top + rows.length * rowH + bottom, pw = W - left - right;
    const totals = rows.map(r => series.reduce((a, se) => a + (r.values[se.key] || 0), 0));
    const maxV = niceMax(Math.max(0, ...totals));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    for (const t of ticks(maxV)) {
      const x = left + (t / maxV) * pw;
      svg.appendChild(s("line", { class: "grid", x1: x, x2: x, y1: top, y2: top + rows.length * rowH }));
      svg.appendChild(s("text", { class: "axis-label", x, y: H - 8, "text-anchor": "middle" }, fmt(t)));
    }
    rows.forEach((r, ri) => {
      const y = top + ri * rowH + 3;
      svg.appendChild(s("text", { class: "row-label", x: left - 10, y: y + (r.sub ? 10 : 15), "text-anchor": "end" }, clip(r.label, 34)));
      if (r.sub) svg.appendChild(s("text", { class: "row-sub", x: left - 10, y: y + 21, "text-anchor": "end" }, clip(r.sub, 40)));
      let x = left;
      series.forEach(se => {
        const v = r.values[se.key] || 0;
        const w = (v / maxV) * pw;
        if (w > 0.5) {
          const rect = s("rect", { class: "mark", x, y, width: Math.max(0, w - 2), height: 24, rx: 0, style: `fill:${se.color}` });
          hover(rect, () => [{ value: fmt(v), label: se.name, color: se.color }, { value: r.label, label: `total ${fmt(totals[ri])}` }, ...(totals[ri] ? [{ value: pct(v / totals[ri]), label: "of total" }] : [])]);
          svg.appendChild(rect);
        }
        x += w;
      });
      svg.appendChild(s("text", { class: "val-label", x: left + (totals[ri] / maxV) * pw + 4, y: y + 16 }, fmt(totals[ri])));
    });
    const tcols = [{ key: "label", label: "config" }, ...series.map(se => ({ key: se.key, label: se.name, num: true, fmt: v => fmt(v || 0) })), { key: "_total", label: totalLabel || "total", num: true, fmt: v => fmt(v) }];
    const trows = rows.map((r, i) => ({ label: r.label + (r.sub ? ` (${r.sub})` : ""), ...r.values, _total: totals[i] }));
    return figure(title, sub, svg, series, table(tcols, trows));
  }

  // Simple horizontal bars, one per row, each with its own colour (the row is the entity).
  function barsH(opts) {
    const { rows, fmt = num, title, sub } = opts;
    const W = 760, left = 230, right = 70, top = 8, rowH = 28, bottom = 26;
    const H = top + rows.length * rowH + bottom, pw = W - left - right;
    const maxV = niceMax(Math.max(0, ...rows.map(r => r.value || 0)));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    for (const t of ticks(maxV)) {
      const x = left + (t / maxV) * pw;
      svg.appendChild(s("line", { class: "grid", x1: x, x2: x, y1: top, y2: top + rows.length * rowH }));
      svg.appendChild(s("text", { class: "axis-label", x, y: H - 8, "text-anchor": "middle" }, fmt(t)));
    }
    rows.forEach((r, ri) => {
      const y = top + ri * rowH + 3;
      svg.appendChild(s("text", { class: "row-label", x: left - 10, y: y + (r.sub ? 9 : 14), "text-anchor": "end" }, clip(r.label, 34)));
      if (r.sub) svg.appendChild(s("text", { class: "row-sub", x: left - 10, y: y + 20, "text-anchor": "end" }, clip(r.sub, 40)));
      if (isNum(r.value)) {
        const w = (r.value / maxV) * pw, rr = Math.min(4, w / 2);
        const d = `M${left},${y} h${Math.max(0, w - rr)} a${rr},${rr} 0 0 1 ${rr},${rr} v${22 - 2 * rr} a${rr},${rr} 0 0 1 -${rr},${rr} H${left} Z`;
        const mark = s("path", { class: "mark", d, style: `fill:${r.color}` });
        hover(mark, () => [{ value: fmt(r.value), label: r.label, color: r.color }, ...(r.extra || [])]);
        svg.appendChild(mark);
        svg.appendChild(s("text", { class: "val-label", x: left + w + 4, y: y + 15 }, fmt(r.value)));
      }
    });
    const tcols = [{ key: "label", label: "config" }, { key: "value", label: title, num: true, fmt: v => fmt(v) }];
    return figure(title, sub, svg, null, table(tcols, rows.map(r => ({ label: r.label + (r.sub ? ` (${r.sub})` : ""), value: r.value }))));
  }

  // Dot plot: rows [{label, sub, points: {key: v}}], series [{key, name, color}] on one axis.
  function dotPlot(opts) {
    const { rows, series, fmt = secs, title, sub } = opts;
    const W = 760, left = 230, right = 40, top = 8, rowH = 30, bottom = 26;
    const H = top + rows.length * rowH + bottom, pw = W - left - right;
    const maxV = niceMax(Math.max(0, ...rows.flatMap(r => series.map(se => r.points[se.key] || 0))));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    for (const t of ticks(maxV)) {
      const x = left + (t / maxV) * pw;
      svg.appendChild(s("line", { class: "grid", x1: x, x2: x, y1: top, y2: top + rows.length * rowH }));
      svg.appendChild(s("text", { class: "axis-label", x, y: H - 8, "text-anchor": "middle" }, fmt(t)));
    }
    rows.forEach((r, ri) => {
      const cy = top + ri * rowH + 15;
      svg.appendChild(s("text", { class: "row-label", x: left - 10, y: cy + (r.sub ? -2 : 4), "text-anchor": "end" }, clip(r.label, 34)));
      if (r.sub) svg.appendChild(s("text", { class: "row-sub", x: left - 10, y: cy + 10, "text-anchor": "end" }, clip(r.sub, 40)));
      const xs = series.map(se => isNum(r.points[se.key]) ? left + (r.points[se.key] / maxV) * pw : null).filter(x => x != null);
      if (xs.length > 1) svg.appendChild(s("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: cy, y2: cy, style: "stroke:var(--axis);stroke-width:2" }));
      series.forEach(se => {
        const v = r.points[se.key];
        if (!isNum(v)) return;
        const cx = left + (v / maxV) * pw;
        const c = s("circle", { class: "pt mark", cx, cy, r: 5.5, style: `fill:${se.color}` });
        const hit = s("circle", { class: "hit", cx, cy, r: 12 });
        hover(hit, () => [{ value: fmt(v), label: se.name, color: se.color }, { value: r.label, label: "" }]);
        svg.appendChild(c); svg.appendChild(hit);
      });
    });
    const tcols = [{ key: "label", label: "config" }, ...series.map(se => ({ key: se.key, label: se.name, num: true, fmt: v => fmt(v) }))];
    return figure(title, sub, svg, series, table(tcols, rows.map(r => ({ label: r.label + (r.sub ? ` (${r.sub})` : ""), ...r.points }))));
  }

  // Heat table: rows × cols of counts on a one-hue ramp; text stays legible per step.
  function heatTable(opts) {
    const { rowLabels, colLabels, cell, rowTotal, colTotal, fmt = v => v ? String(v) : "", title, sub, onCell, rowTitle } = opts;
    let max = 0;
    rowLabels.forEach(r => colLabels.forEach(c => { const v = cell(r, c); if (isNum(v)) max = Math.max(max, v); }));
    const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";
    function shade(v) {
      if (!isNum(v) || v <= 0) return null;
      const step = Math.min(7, Math.max(1, Math.ceil((v / max) * 7)));
      // light: dark text up to step 3; dark mode ramp is inverted (light steps are high).
      const textDark = dark ? step >= 5 : step <= 3;
      return { bg: `var(--seq-${step})`, cls: textDark ? "l" : "d" };
    }
    const tbl = h("table", { class: "tbl heat" });
    tbl.appendChild(h("thead", null, h("tr", null, h("th", null, rowTitle || ""), colLabels.map(c => h("th", { class: "num" }, c.label || c)), rowTotal ? h("th", { class: "num" }, "total") : null)));
    const body = h("tbody");
    rowLabels.forEach(r => {
      const tr = h("tr", null, h("td", { title: r.title || "" }, r.label || r));
      colLabels.forEach(c => {
        const v = cell(r, c), sh = shade(v);
        const td = h("td", { class: "cell " + (sh ? sh.cls : ""), style: sh ? { background: sh.bg } : null }, fmt(v));
        if (onCell && v) { td.style.cursor = "pointer"; td.addEventListener("click", () => onCell(r, c)); }
        tr.appendChild(td);
      });
      if (rowTotal) tr.appendChild(h("td", { class: "num" }, fmt(rowTotal(r))));
      body.appendChild(tr);
    });
    if (colTotal) body.appendChild(h("tr", { class: "total" }, h("td", null, "total"), colLabels.map(c => h("td", { class: "cell" }, fmt(colTotal(c)))), rowTotal ? h("td", { class: "num" }, fmt(rowLabels.reduce((a, r) => a + (rowTotal(r) || 0), 0))) : null));
    tbl.appendChild(body);
    return h("div", null, title ? h("div", { class: "fig-head" }, h("span", { class: "fig-title" }, title), sub ? h("span", { class: "fig-sub" }, sub) : null) : null, h("div", { class: "tbl-wrap" }, tbl));
  }

  // ---------------------------------------------------------------- shared pieces
  function scoreBadge(score, errorType) {
    if (score === 1) return h("span", { class: "badge pass" }, "✓ pass");
    if (score === 0) return h("span", { class: "badge fail", title: errorType || "" }, "✗ fail");
    if (score === -1) return h("span", { class: "badge infra", title: errorType || "" }, "! infra");
    return h("span", { class: "badge" }, "unscored");
  }
  function configChip(cfg) {
    return h("span", null, h("span", { class: "swatch", style: { background: colorOf(cfg) } }), cfgLabel(cfg));
  }
  function shortProvider(p) { return String(p || "").replace(/^local https?:\/\//, "local ").replace(/\/v1\/?$/, ""); }
  function configCard(cfg, full) {
    const g = cfg.gpu || {};
    const attempts = cfg.attempts.map(a => `${a.label} (${a.n})`).join(", ");
    const kv = [
      ["GPU", g.gpu || (cfg.reference ? "– (API models)" : "not recorded (no gpu.txt)")],
      ["node / partition", [g.node, g.partition].filter(Boolean).join(" / ") || "–"],
      ["job", g.job || "–"],
      ["workers", cfg.workers != null ? cfg.workers : "–"],
      ["attempts found", `${cfg.attempts.length}: ${attempts}`],
      ["episodes", `${cfg.n} rows` + (cfg.planned ? ` (${cfg.planned} planned per attempt)` : "")],
      ["commit", cfg.commit || "–"], ["started", cfg.started || "–"],
    ];
    const settings = Object.entries(cfg.settings || {}).map(([k, v]) => `${k}=${v}`).join("  ");
    return h("div", { class: "card" },
      h("div", { class: "config-title" }, h("span", { class: "cfg-swatch", style: { background: colorOf(cfg) } }), h("a", { href: `#config/${encodeURIComponent(cfg.key)}` }, cfg.title), cfg.reference ? h("span", { class: "badge" }, "reference") : null),
      h("dl", { class: "kv" }, kv.map(([k, v]) => [h("dt", null, k), h("dd", null, v)])),
      settings ? h("div", { class: "note mono small" }, settings) : null,
      h("h4", null, "Models per phase"),
      full ? table([{ key: "phase", label: "phase" }, { key: "model", label: "model" }, { key: "provider", label: "provider / endpoint" }, { key: "api", label: "API" }, { key: "settings", label: "settings" }], cfg.models || [])
           : table([{ key: "phase", label: "phase" }, { key: "model", label: "model" }, { key: "provider", label: "provider", fmt: shortProvider }], cfg.models || []));
  }

  // ---------------------------------------------------------------- overview
  function overview() {
    const root = h("div");
    root.appendChild(h("p", { class: "note" }, `Built ${DATA.built} from ${DATA.roots.join(", ")}${DATA.pooled ? " + pooled " + DATA.pooled : ""}. ` +
      `${CONFIGS.length} configuration${CONFIGS.length === 1 ? "" : "s"}, ${CONFIGS.reduce((a, c) => a + c.n, 0)} episode rows. ` +
      "Rates leave out infrastructure failures (score −1), as summary.md does. Every chart has a Table view."));

    root.appendChild(h("h2", null, "Configurations"));
    root.appendChild(h("div", { class: "grid" }, CONFIGS.map(c => configCard(c, false))));

    // Success by type × case.
    const series = CONFIGS.map(c => ({ key: c.key, name: cfgLabel(c), color: colorOf(c) }));
    const types = [...new Set(CONFIGS.flatMap(c => c.summary.map(r => r.type)))].filter(t => t !== "all").sort(typeKey);
    const cases = [...new Set(CONFIGS.flatMap(c => c.summary.map(r => r.case)))].filter(c => c !== "all");
    root.appendChild(h("h2", null, "Success rate by request type"));
    root.appendChild(h("p", { class: "note" }, "SimuHome's all-or-nothing score per request, pooled over the attempts found. Hover a bar for n; the Table view lists every value."));
    const grid1 = h("div", { class: "grid charts" });
    for (const kase of [...cases, "all"]) {
      const groups = types.map(t => {
        const values = {}, n = {};
        CONFIGS.forEach(c => { const r = groupRow(c.summary, t, kase); values[c.key] = r ? r.success_rate : null; n[c.key] = r ? r.n_scored : null; });
        return { label: t, values, n };
      });
      const all = {}; CONFIGS.forEach(c => { const r = groupRow(c.summary, "all", kase); all[c.key] = r ? r.success_rate : null; });
      groups.push({ label: "all", values: all });
      grid1.appendChild(h("div", { class: "card" }, groupedColumns({ title: kase === "all" ? "Both cases" : kase, sub: "success", groups, series, max: 1 })));
    }
    root.appendChild(grid1);

    // Other rates.
    root.appendChild(h("h2", null, "Quality rates"));
    root.appendChild(h("p", { class: "note" }, "checks: mean share of expected device values right (feasible). goals ok: eval goals fully reached (feasible, pooled). plans ok: our plans run to SUCCESS. refused/request: goals answered impossible or with a question. exec check and judge: infeasible only."));
    const metrics = [["success_rate", "success"], ["check_rate", "checks"], ["goals_verified_rate", "goals ok"], ["plans_success_rate", "plans ok"], ["execution_check_rate", "exec check"], ["judge_rate", "judge"]];
    const mgroups = metrics.map(([k, label]) => { const values = {}; CONFIGS.forEach(c => { const r = groupRow(c.summary, "all", "all"); values[c.key] = r ? r[k] : null; }); return { label, values }; });
    root.appendChild(h("div", { class: "card" }, groupedColumns({ title: "Overall rates", sub: "all types, both cases", groups: mgroups, series, max: 1 })));
    const rateRows = [];
    for (const c of CONFIGS) for (const kase of ["all", ...cases]) {
      const r = groupRow(c.summary, "all", kase); if (!r) continue;
      rateRows.push({ config: cfgLabel(c), cfg: c, group: kase === "all" ? "overall" : kase, n: r.n, scored: r.n_scored, infra: r.n_infra, errors: r.n_error, success: r.success_rate, checks: r.check_rate, goals: r.goals_verified_rate, plans: r.plans_success_rate, refused: r.goals_refused_mean, exec: r.execution_check_rate, judge: r.judge_rate });
    }
    root.appendChild(h("div", { class: "card" }, table([
      { key: "config", label: "config", fmt: (v, r) => configChip(r.cfg) }, { key: "group", label: "case" }, { key: "n", label: "n", num: true }, { key: "scored", label: "scored", num: true }, { key: "infra", label: "infra", num: true }, { key: "errors", label: "errors", num: true },
      { key: "success", label: "success", num: true, fmt: pct }, { key: "checks", label: "checks", num: true, fmt: pct }, { key: "goals", label: "goals ok", num: true, fmt: pct }, { key: "plans", label: "plans ok", num: true, fmt: pct },
      { key: "refused", label: "refused/request", num: true, fmt: v => num(v, 2) }, { key: "exec", label: "exec check", num: true, fmt: pct }, { key: "judge", label: "judge", num: true, fmt: pct }], rateRows,
      { rowClass: r => r.group === "overall" ? "group-sep" : "" })));

    // Latency.
    root.appendChild(h("h2", null, "Latency per episode"));
    root.appendChild(h("p", { class: "note" }, "Seconds from the request to the reply (the phases' sum; discovery and the simulator reset are not counted), per configuration and the GPU it ran on. Infrastructure failures left out."));
    const lrows = CONFIGS.map(c => { const r = groupRow(c.summary, "all", "all") || {}; return { label: cfgLabel(c), sub: gpuShort(c) + (c.workers ? ` · ${c.workers} workers` : ""), points: { median: r.total_s_median, mean: r.total_s, p95: r.total_s_p95 } }; });
    const lseries = [{ key: "median", name: "median", color: "var(--s1)" }, { key: "mean", name: "mean", color: "var(--s2)" }, { key: "p95", name: "p95", color: "var(--s3)" }];
    const pseries = PHASES.map((p, i) => ({ key: p, name: p, color: phaseColor(i) }));
    const phaseRows = CONFIGS.map(c => { const r = groupRow(c.summary, "all", "all") || {}; const values = {}; PHASES.forEach(p => values[p] = r[`${p}_s`] || 0); return { label: cfgLabel(c), sub: gpuShort(c), values }; });
    root.appendChild(h("div", { class: "grid charts" },
      h("div", { class: "card" }, dotPlot({ title: "Total seconds per episode", sub: "median · mean · p95", rows: lrows, series: lseries })),
      h("div", { class: "card" }, stackedBarsH({ title: "Mean seconds per phase", sub: "stacked; profiling and guarding taken out of planning", rows: phaseRows, series: pseries }))));
    // Per type latency table.
    const latRows = [];
    for (const c of CONFIGS) for (const t of [...types, "all"]) { const r = groupRow(c.summary, t, "all"); if (!r) continue; latRows.push({ config: cfgLabel(c), cfg: c, gpu: gpuShort(c), type: t === "all" ? "overall" : t, n: r.n_costed, mean: r.total_s, median: r.total_s_median, p95: r.total_s_p95, calls: r.total_calls, tin: r.total_tok_in, tout: r.total_tok_out }); }
    root.appendChild(h("details", { class: "sec" }, h("summary", null, "Latency and cost per request type ", h("span", { class: "cnt" }, "(mean / median / p95 seconds, calls, tokens)")), h("div", { class: "body" }, table([
      { key: "config", label: "config", fmt: (v, r) => configChip(r.cfg) }, { key: "gpu", label: "GPU" }, { key: "type", label: "type" }, { key: "n", label: "n", num: true },
      { key: "mean", label: "mean s", num: true, fmt: v => num(v, 1) }, { key: "median", label: "median s", num: true, fmt: v => num(v, 1) }, { key: "p95", label: "p95 s", num: true, fmt: v => num(v, 1) },
      { key: "calls", label: "LLM calls", num: true, fmt: v => num(v, 1) }, { key: "tin", label: "tokens in", num: true, fmt: int }, { key: "tout", label: "tokens out", num: true, fmt: int }], latRows, { rowClass: r => r.type === "overall" ? "total" : "" }))));

    // Tokens and calls.
    root.appendChild(h("h2", null, "Tokens and LLM calls per episode"));
    const tinRows = CONFIGS.map(c => { const r = groupRow(c.summary, "all", "all") || {}; const values = {}; PHASES.forEach(p => values[p] = r[`${p}_tok_in`] || 0); return { label: cfgLabel(c), sub: gpuShort(c), values }; });
    const toutRows = CONFIGS.map(c => { const r = groupRow(c.summary, "all", "all") || {}; const values = {}; PHASES.forEach(p => values[p] = r[`${p}_tok_out`] || 0); return { label: cfgLabel(c), sub: gpuShort(c), values }; });
    const callRows = CONFIGS.map(c => { const r = groupRow(c.summary, "all", "all") || {}; return { label: cfgLabel(c), sub: gpuShort(c), value: r.total_calls, color: colorOf(c), extra: PHASES.filter(p => r[`${p}_calls`]).map(p => ({ value: num(r[`${p}_calls`], 1), label: p })) }; });
    root.appendChild(h("div", { class: "grid charts" },
      h("div", { class: "card" }, stackedBarsH({ title: "Mean input tokens per phase", sub: "prompt tokens", rows: tinRows, series: pseries, fmt: compact })),
      h("div", { class: "card" }, stackedBarsH({ title: "Mean output tokens per phase", sub: "completion tokens, reasoning included", rows: toutRows, series: pseries, fmt: compact })),
      h("div", { class: "card" }, barsH({ title: "Mean LLM calls per episode", sub: "hover for the split per phase", rows: callRows, fmt: v => num(v, 1) }))));

    // Failure categories across configs.
    root.appendChild(h("h2", null, "Failure categories across configurations"));
    root.appendChild(h("p", { class: "note" }, "Failed and errored episodes by category (one category per episode, first matching rule wins; the rules are listed on each config page). Click a cell to open those episodes."));
    const catsPresent = CAT_NAMES.filter(cat => CONFIGS.some(c => Object.values(c.failures.by_type[cat] || {}).some(v => v)));
    const total = (c, cat) => Object.values(c.failures.by_type[cat] || {}).reduce((a, b) => a + b, 0);
    root.appendChild(h("div", { class: "card" }, heatTable({
      rowTitle: "category", rowLabels: catsPresent.map(cat => ({ label: cat, title: CAT_HOW[cat], key: cat })), colLabels: CONFIGS.map(c => ({ label: cfgLabel(c), key: c.key })),
      cell: (r, c) => total(CONFIGS.find(x => x.key === c.key), r.key),
      fmt: v => v ? String(v) : "", rowTotal: r => CONFIGS.reduce((a, c) => a + total(c, r.key), 0), colTotal: c => catsPresent.reduce((a, cat) => a + total(CONFIGS.find(x => x.key === c.key), cat), 0),
      onCell: (r, c) => { location.hash = `#episodes?config=${encodeURIComponent(c.key)}&category=${encodeURIComponent(r.key)}`; },
    })));

    // Attempts.
    const multi = CONFIGS.filter(c => c.attempts.length > 1);
    if (multi.length) {
      root.appendChild(h("h2", null, "Attempts: per attempt and pooled"));
      root.appendChild(h("p", { class: "note" }, "Success per attempt for the repeated types (qt2, qt4-x), the between-attempt range and standard deviation, and the pooled rate (all attempts' episodes together)."));
      for (const c of multi) root.appendChild(h("div", { class: "card" }, h("div", { class: "config-title" }, configChip(c)), spreadTable(c), attemptsTable(c)));
    } else {
      root.appendChild(h("h2", null, "Attempts"));
      root.appendChild(h("p", { class: "note" }, "One attempt per configuration was found; pass several --roots (or --pooled) to see per-attempt numbers and the between-attempt spread."));
    }
    return root;
  }
  function typeKey(a, b) {
    const order = DATA.types;
    const ia = order.indexOf(a), ib = order.indexOf(b);
    if (ia >= 0 && ib >= 0) return ia - ib; if (ia >= 0) return -1; if (ib >= 0) return 1;
    return a.localeCompare(b, undefined, { numeric: true });
  }
  function spreadTable(c) {
    if (!c.spread || !c.spread.length) return h("p", { class: "note" }, "No repeated type with two or more scored attempts.");
    const labels = c.attempts.map(a => a.label);
    const cols = [{ key: "type", label: "type" }, { key: "case", label: "case" }, ...labels.map(l => ({ key: "a_" + l, label: "attempt " + l, num: true, fmt: pct })), { key: "range", label: "min – max", num: true }, { key: "sd", label: "sd", num: true, fmt: v => num(v, 3) }, { key: "pooled", label: "pooled", num: true, fmt: pct }];
    const rows = c.spread.map(sp => { const r = { type: sp.type, case: sp.case, range: `${pct(sp.min)} – ${pct(sp.max)}`, sd: sp.sd, pooled: sp.pooled }; labels.forEach(l => r["a_" + l] = sp.attempts[l]); return r; });
    return table(cols, rows, { rowClass: r => r.case === "all" ? "total" : "" });
  }
  function attemptsTable(c) {
    const rows = c.attempts.map(a => { const r = groupRow(a.summary, "all", "all") || {}; return { attempt: a.label, n: r.n, scored: r.n_scored, infra: r.n_infra, success: r.success_rate, checks: r.check_rate, goals: r.goals_verified_rate, exec: r.execution_check_rate, judge: r.judge_rate, s: r.total_s, p95: r.total_s_p95 }; });
    const p = groupRow(c.summary, "all", "all") || {};
    rows.push({ attempt: "pooled", n: p.n, scored: p.n_scored, infra: p.n_infra, success: p.success_rate, checks: p.check_rate, goals: p.goals_verified_rate, exec: p.execution_check_rate, judge: p.judge_rate, s: p.total_s, p95: p.total_s_p95 });
    return table([{ key: "attempt", label: "attempt" }, { key: "n", label: "n", num: true }, { key: "scored", label: "scored", num: true }, { key: "infra", label: "infra", num: true }, { key: "success", label: "success", num: true, fmt: pct }, { key: "checks", label: "checks", num: true, fmt: pct }, { key: "goals", label: "goals ok", num: true, fmt: pct }, { key: "exec", label: "exec check", num: true, fmt: pct }, { key: "judge", label: "judge", num: true, fmt: pct }, { key: "s", label: "mean s", num: true, fmt: v => num(v, 1) }, { key: "p95", label: "p95 s", num: true, fmt: v => num(v, 1) }], rows, { rowClass: r => r.attempt === "pooled" ? "total" : "" });
  }

  // ---------------------------------------------------------------- config page
  function configPage(cfg) {
    const root = h("div");
    root.appendChild(configCard(cfg, true));
    let view = "pooled";
    const tablesHost = h("div");
    function renderTables() {
      const src = view === "pooled" ? cfg : cfg.attempts.find(a => a.label === view);
      const t = src.tables;
      tablesHost.replaceChildren(h("div", null,
        h("h3", null, "Score and goals"), h("div", { class: "card" }, mdTable(t.score)),
        cfg.buckets && cfg.buckets.length && view === "pooled" ? [h("h3", null, "Official HomeBench metrics (per bucket)"), h("div", { class: "card" }, mdTable(cfg.buckets), (cfg.notes || []).map(n => h("p", { class: "note" }, n)))] : null,
        h("h3", null, "Seconds per phase (mean per episode)"), h("div", { class: "card" }, mdTable(t.seconds)),
        h("h3", null, "Tokens per phase (mean per episode, input / output)"), h("div", { class: "card" }, mdTable(t.tokens))));
    }
    const head = h("h2", null, "Statistics");
    if (cfg.attempts.length > 1) {
      const sel = h("select", null, h("option", { value: "pooled" }, `pooled (${cfg.n} rows)`), cfg.attempts.map(a => h("option", { value: a.label }, `attempt ${a.label} (${a.n})`)));
      sel.addEventListener("change", () => { view = sel.value; renderTables(); });
      head.appendChild(h("span", { class: "fig-sub", style: { marginLeft: "12px" } }, " view: ", sel));
    }
    root.appendChild(head);
    root.appendChild(h("details", { class: "sec" }, h("summary", null, "Column legend"), h("div", { class: "body" }, h("pre", null, DATA.legend))));
    renderTables();
    root.appendChild(tablesHost);
    if (cfg.attempts.length > 1) { root.appendChild(h("h3", null, "Attempts")); root.appendChild(h("div", { class: "card" }, spreadTable(cfg), attemptsTable(cfg))); }

    // Failure catalogue.
    root.appendChild(h("h2", null, "Failure catalogue"));
    const f = cfg.failures;
    const cats = CAT_NAMES.filter(cat => f.by_type[cat]);
    const types = [...new Set(f.cases.map(c => c.type))].sort(typeKey);
    const casesPresent = [...new Set(f.cases.map(c => c.case))];
    root.appendChild(h("p", { class: "note" }, `${f.cases.length} failed or errored episode rows of ${cfg.n}. Counts per category and request type; click a cell to filter the list below.`));
    const listHost = h("div");
    let filterCat = null, filterType = null;
    function renderList() {
      let rows = f.cases;
      if (filterCat) rows = rows.filter(c => c.category === filterCat);
      if (filterType) rows = rows.filter(c => c.type === filterType);
      currentList = rows.map(r => r.id);
      listHost.replaceChildren(
        h("div", { class: "filters" }, h("span", { class: "note" }, `${rows.length} case${rows.length === 1 ? "" : "s"}`), filterCat || filterType ? h("button", { type: "button", onclick: () => { filterCat = filterType = null; renderList(); } }, "clear filter") : null, filterCat ? h("span", { class: "badge cat" }, filterCat) : null, filterType ? h("span", { class: "badge" }, filterType) : null),
        table([
          { key: "episode", label: "episode", fmt: (v, r) => h("a", { href: `#episode/${encodeURIComponent(r.id)}`, class: "mono" }, v) }, cfg.attempts.length > 1 ? { key: "attempt", label: "attempt" } : null,
          { key: "type", label: "type" }, { key: "case", label: "case" }, { key: "score", label: "score", fmt: (v, r) => scoreBadge(v, r.error_type) }, { key: "outcome", label: "outcome" },
          { key: "category", label: "category", fmt: v => h("span", { class: "badge cat", title: CAT_HOW[v] || "" }, v) }, { key: "explain", label: "error, clarification or failed check", cls: "wrap" }].filter(Boolean), rows, { sortable: true, onRow: r => openEpisode(r.id) }));
    }
    root.appendChild(h("div", { class: "card" }, heatTable({
      rowTitle: "category", rowLabels: cats.map(cat => ({ label: cat, title: CAT_HOW[cat], key: cat })), colLabels: types.map(t => ({ label: t, key: t })),
      cell: (r, c) => (f.by_type[r.key] || {})[c.key] || 0, rowTotal: r => Object.values(f.by_type[r.key] || {}).reduce((a, b) => a + b, 0), colTotal: c => cats.reduce((a, cat) => a + ((f.by_type[cat] || {})[c.key] || 0), 0),
      onCell: (r, c) => { filterCat = r.key; filterType = c.key; renderList(); listHost.scrollIntoView({ behavior: "smooth", block: "start" }); },
    }), casesPresent.length > 1 ? table([{ key: "category", label: "category" }, ...casesPresent.map(k => ({ key: k, label: k, num: true })), { key: "total", label: "total", num: true }],
      cats.map(cat => ({ category: cat, ...Object.fromEntries(casesPresent.map(k => [k, (f.by_case[cat] || {})[k] || 0])), total: Object.values(f.by_case[cat] || {}).reduce((a, b) => a + b, 0) }))) : null));
    root.appendChild(h("details", { class: "sec" }, h("summary", null, "How the categories are detected ", h("span", { class: "cnt" }, "(precedence order)")), h("div", { class: "body" }, table([{ key: "cat", label: "category" }, { key: "how", label: "rule", cls: "wrap" }], DATA.categories.map(([cat, how]) => ({ cat, how }))))));
    root.appendChild(h("h3", null, "Per-case analysis"));
    renderList();
    root.appendChild(h("div", { class: "card" }, listHost));
    return root;
  }

  // ---------------------------------------------------------------- episodes tab
  function allEpisodes() {
    const out = [];
    for (const c of CONFIGS) for (const e of c.episodes) out.push({ ...e, cfg: c, config: c.key });
    return out;
  }
  function episodesPage(params) {
    const root = h("div");
    const rows = allEpisodes();
    const uniq = k => [...new Set(rows.map(r => r[k]).filter(v => v != null))];
    const state = { config: params.get("config") || "", type: params.get("type") || "", case: params.get("case") || "", score: params.get("score") || "", category: params.get("category") || "", outcome: params.get("outcome") || "", q: params.get("q") || "" };
    const sel = (key, options, labels) => { const el = h("select", null, h("option", { value: "" }, "all"), options.map((o, i) => h("option", { value: o, selected: String(state[key]) === String(o) }, labels ? labels[i] : o))); el.addEventListener("change", () => { state[key] = el.value; render(); }); return el; };
    const search = h("input", { type: "search", placeholder: "episode, explanation…", value: state.q });
    search.addEventListener("input", () => { state.q = search.value; render(); });
    const host = h("div");
    const count = h("span", { class: "note" });
    root.appendChild(h("div", { class: "filters" },
      h("label", null, "config ", sel("config", CONFIGS.map(c => c.key), CONFIGS.map(cfgLabel))),
      h("label", null, "type ", sel("type", uniq("type").sort(typeKey))), h("label", null, "case ", sel("case", uniq("case"))),
      h("label", null, "score ", sel("score", ["1", "0", "-1", "none"], ["pass", "fail", "infra", "unscored"])),
      h("label", null, "category ", sel("category", CAT_NAMES.filter(c => rows.some(r => r.category === c)))),
      h("label", null, "outcome ", sel("outcome", uniq("outcome"))), h("label", null, "search ", search), count));
    function render() {
      let data = rows;
      if (state.config) data = data.filter(r => r.config === state.config);
      if (state.type) data = data.filter(r => r.type === state.type);
      if (state.case) data = data.filter(r => r.case === state.case);
      if (state.score) data = data.filter(r => state.score === "none" ? r.score == null : String(r.score) === state.score);
      if (state.category) data = data.filter(r => r.category === state.category);
      if (state.outcome) data = data.filter(r => r.outcome === state.outcome);
      if (state.q) { const q = state.q.toLowerCase(); data = data.filter(r => (r.episode + " " + (r.explain || "") + " " + (r.error_type || "")).toLowerCase().includes(q)); }
      currentList = data.map(r => r.id);
      count.textContent = `${data.length} of ${rows.length}`;
      host.replaceChildren(table([
        { key: "config", label: "config", fmt: (v, r) => configChip(r.cfg) }, { key: "attempt", label: "att." }, { key: "episode", label: "episode", fmt: (v, r) => h("a", { href: `#episode/${encodeURIComponent(r.id)}`, class: "mono", onclick: ev => ev.preventDefault() }, v) },
        { key: "type", label: "type" }, { key: "case", label: "case" }, { key: "score", label: "score", fmt: (v, r) => scoreBadge(v, r.error_type) }, { key: "outcome", label: "outcome", fmt: (v, r) => v + (r.error_step ? ` @${r.error_step}` : "") },
        { key: "category", label: "category", fmt: v => v ? h("span", { class: "badge cat", title: CAT_HOW[v] || "" }, v) : "" }, { key: "explain", label: "explanation", cls: "wrap", fmt: v => clip(v, 180) },
        { key: "total_s", label: "s", num: true, fmt: v => num(v, 0) }, { key: "total_calls", label: "calls", num: true, fmt: v => num(v, 0) }, { key: "total_tok_in", label: "tok in", num: true, fmt: compact }, { key: "total_tok_out", label: "tok out", num: true, fmt: compact }],
        data, { sortable: true, onRow: r => openEpisode(r.id) }));
    }
    render();
    root.appendChild(h("div", { class: "card" }, host));
    return root;
  }

  // ---------------------------------------------------------------- episode viewer
  function loadRecord(id) {
    if (recordCache.has(id)) return recordCache.get(id);
    const el = document.querySelector(`script.rec[data-id="${cssEscape(id)}"]`);
    if (!el) return null;
    let rec = null;
    try { rec = JSON.parse(el.textContent); } catch (e) { rec = { _parse_error: String(e) }; }
    recordCache.set(id, rec);
    return rec;
  }
  function cssEscape(v) { return window.CSS && CSS.escape ? CSS.escape(v) : v.replace(/["\\]/g, "\\$&"); }
  function findEpisodeRow(id) { for (const c of CONFIGS) { const e = c.episodes.find(x => x.id === id); if (e) return { ...e, cfg: c }; } return null; }

  const scrim = h("div", { id: "scrim" }); const panel = h("div", { id: "panel", role: "dialog", "aria-modal": "true" });
  document.body.appendChild(scrim); document.body.appendChild(panel);
  scrim.addEventListener("click", closePanel);
  document.addEventListener("keydown", ev => { if (ev.key === "Escape" && panel.classList.contains("open")) closePanel(); });
  function closePanel() {
    panel.classList.remove("open"); scrim.classList.remove("open"); document.body.style.overflow = "";
    if (location.hash.startsWith("#episode/")) history.replaceState(null, "", lastPageHash || "#overview");
  }
  function openEpisode(id) { if (location.hash !== `#episode/${encodeURIComponent(id)}`) location.hash = `#episode/${encodeURIComponent(id)}`; else showEpisode(id); }
  function sec(title, count, body, open) {
    return h("details", { class: "sec", open: open ? "" : null }, h("summary", null, title, count != null ? h("span", { class: "cnt" }, `(${count})`) : null), h("div", { class: "body" }, body));
  }
  function jsonDetails(label, value) {
    const d = h("details", { class: "json" }, h("summary", null, label || "JSON"));
    d.addEventListener("toggle", () => { if (d.open && d.children.length === 1) d.appendChild(h("pre", null, jsonPretty(value))); }, { once: false });
    return d;
  }
  function kvList(pairs) { return h("dl", { class: "kv" }, pairs.filter(p => p[1] != null && p[1] !== "").map(([k, v]) => [h("dt", null, k), h("dd", null, v instanceof Node ? v : String(v))])); }
  function btTree(node, root = true) {
    if (!node || typeof node !== "object") return h("span", { class: "muted" }, "(no tree)");
    const details = [];
    if (node.parameters && Object.keys(node.parameters).length) details.push(JSON.stringify(node.parameters));
    if (node.operator) details.push(`${node.operator} ${JSON.stringify(node.expected_value)}`);
    if (node.seconds != null) details.push(`${node.seconds} s`);
    if (node.time_of_day) details.push(`at ${node.time_of_day}`);
    if (node.policy) details.push(node.policy + (node.synchronise ? ", synchronised" : ""));
    if (node.memory) details.push("memory");
    if (node.op) details.push(`${node.op}(${(node.inputs || []).join(", ")}) → ${node.output}`);
    if (node.output && node.type === "read") details.push(`→ ${node.output}`);
    if (node.timeout_seconds != null) details.push(`timeout ${node.timeout_seconds} s`);
    if (node.checks) details.push(`${node.checks.length} check(s)`);
    const li = h("li", null, h("span", { class: "nt " + (node.type || "") }, node.type || "?"), h("span", null, node.name || node.artifact || ""), details.length ? h("span", { class: "nd" }, " — " + details.join(" · ")) : null);
    const kids = (node.children || []).concat(node.child ? [node.child] : []);
    if (kids.length) li.appendChild(h("ul", { class: "bt" }, kids.map(k => btTree(k, false))));
    return root ? h("ul", { class: "bt root" }, li) : li;
  }
  function checkClassBadge(cls) {
    const map = { ok: "pass", missing: "fail", wrong_value: "fail", wrong_timing: "fail" };
    return h("span", { class: "badge " + (map[cls] || "") }, cls || "?");
  }
  function warnings(rec, ev) {
    const out = [];
    if (rec.error) out.push(`error (${rec.error_step || "?"}): ${rec.error}`);
    (rec.warnings || []).forEach(w => out.push(typeof w === "string" ? w : JSON.stringify(w)));
    (rec.structures || []).forEach((st, i) => { if (st.error) out.push(`structuring of intent ${i + 1}: ${st.error}`); if (st.context_unavailable) out.push(`structuring of intent ${i + 1}: context unavailable`); });
    (rec.plans || []).forEach((p, i) => { if (p.error) out.push(`plan entry ${i + 1}: ${p.error} — ${p.explanation || ""}`); });
    (((rec.execution || {}).plans) || []).forEach(p => { if (p.error) out.push(`execution of plan ${p.plan}: ${p.error}`); else if (p.status && p.status !== "SUCCESS") out.push(`execution of plan ${p.plan}: ${p.status}`); });
    (((rec.execution || {}).actions) || []).forEach(a => { if (a.success === false) out.push(`action ${a.name} failed (HTTP ${a.status_code})${a.error ? ": " + a.error : ""}`); });
    (((rec.meter || {}).calls) || []).forEach(c => { if (c.ok === false) out.push(`LLM call in ${c.step} (${c.model}) failed${c.error ? ": " + c.error : ""}`); });
    (rec.tool_log || []).forEach(t => { if (t.outcome && t.outcome.ok === false) out.push(`tool ${t.tool} failed: ${t.outcome.error_type || t.outcome.status_code || ""}`); });
    if (ev && ev.score === -1) out.push(`infrastructure: ${ev.detail || ev.error_type}`);
    if (rec._stripped && Object.keys(rec._stripped).length) out.push("viewer note: " + Object.entries(rec._stripped).map(([k, v]) => `${k} reduced (${(v.bytes / 1e3).toFixed(0)} KB${v.count ? `, ${v.count} snapshots` : ""} in the original record)`).join("; "));
    return out;
  }
  let lastPageHash = "#overview";
  function showEpisode(id) {
    const rec = loadRecord(id);
    const row = findEpisodeRow(id);
    panel.replaceChildren();
    const idx = currentList.indexOf(id);
    const nav = h("span", null,
      h("button", { type: "button", disabled: idx <= 0 ? "" : null, onclick: () => openEpisode(currentList[idx - 1]) }, "◂ prev"), " ",
      h("button", { type: "button", disabled: idx < 0 || idx >= currentList.length - 1 ? "" : null, onclick: () => openEpisode(currentList[idx + 1]) }, "next ▸"));
    const head = h("div", { class: "panel-head" }, h("h2", { class: "mono" }, row ? row.episode : id), row ? configChip(row.cfg) : null, row && row.cfg.attempts.length > 1 ? h("span", { class: "badge" }, "attempt " + row.attempt) : null,
      row ? scoreBadge(row.score, row.error_type) : null, row && row.category ? h("span", { class: "badge cat", title: CAT_HOW[row.category] || "" }, row.category) : null,
      h("span", { class: "spacer" }), nav, " ", h("button", { type: "button", onclick: closePanel }, "✕ close"));
    panel.appendChild(head);
    if (!rec) { panel.appendChild(h("p", { class: "note" }, DATA.records_embedded ? "Record not found in this page." : "This report was built with --no-records; rebuild without it to see trajectories.")); open(); return; }
    const ev = rec._eval || {};
    const stages = (rec.meter || {}).stages || {};
    const totalS = Object.values(stages).reduce((a, st) => a + (st.seconds || 0), 0);
    const totalCalls = Object.values(stages).reduce((a, st) => a + (st.calls || 0), 0);
    const tin = Object.values(stages).reduce((a, st) => a + (st.tokens_in || 0), 0), tout = Object.values(stages).reduce((a, st) => a + (st.tokens_out || 0), 0);
    panel.appendChild(h("div", { class: "stat-row" },
      stat("type / case", `${rec.query_type} ${rec.case}`), stat("outcome", (rec.outcome || "–") + (rec.error_step ? ` @ ${rec.error_step}` : "")),
      stat("eval", ev.error_type || (ev.score === 1 ? "pass" : ev.score == null ? "unscored" : "fail")), stat("seconds", Object.keys(stages).length ? secs(totalS) : "–"), stat("wall", secs(rec.wall_seconds)),
      stat("LLM calls", totalCalls), stat("tokens in / out", `${int(tin)} / ${int(tout)}`)));
    if (row && row.explain) panel.appendChild(h("div", { class: "quote" }, h("b", null, "Why it failed: "), row.explain));

    // 1. Query and ground truth.
    const truth = [];
    const checks = (ev.breakdown || {}).checks || [];
    if (rec.ground_truth && rec.ground_truth.eval) truth.push(h("h4", null, "Answer key (benchmark file)"), h("pre", null, jsonPretty(rec.ground_truth.eval)));
    if (checks.length) truth.push(h("h4", null, "Expected values (the evaluation's checks)"), table([
      { key: "goal", label: "goal", num: true }, { key: "where", label: "device / room", fmt: (v, c) => c.device || c.room || "" }, { key: "attribute", label: "attribute" }, { key: "description", label: "description" },
      { key: "expected", label: "expected", fmt: v => typeof v === "object" ? JSON.stringify(v) : String(v) }, { key: "class", label: "verdict", fmt: checkClassBadge },
      { key: "detail", label: "detail", cls: "wrap", fmt: (v, c) => c.class === "wrong_value" ? `seen ${JSON.stringify(c.values)}` : c.class === "wrong_timing" ? `reached at tick ${c.reached_at}` : c.class === "missing" ? (c.attempted ? "an action was attempted" : "never attempted") : (c.matched || "") }], checks));
    const ec = ev.execution_check;
    if (ec && ec.infeasible_part) truth.push(h("h4", null, "Infeasible part"), kvList([["kind", ec.infeasible_part.kind], ["devices", (ec.infeasible_part.devices || []).join(", ")], ["stand-ins", (ec.infeasible_part.stand_ins || []).join(", ")]]));
    if (rec.meta) truth.push(h("h4", null, "Episode meta"), h("span", { class: "mono small" }, Object.entries(rec.meta).map(([k, v]) => `${k}=${v}`).join("  ")));
    panel.appendChild(sec("Query and ground truth", null, [h("div", { class: "quote" }, rec.input || rec.query || ""), rec.input && rec.query && rec.input !== rec.query ? h("p", { class: "note" }, "Raw query: ", rec.query) : null, truth], true));

    // 2. Segmentation.
    const intents = rec.intents || [];
    panel.appendChild(sec("Segmentation: intents", intents.length, intents.length ? table([{ key: "i", label: "#", num: true }, { key: "text", label: "text", cls: "wrap" }, { key: "type", label: "type" }, { key: "qualifiers", label: "qualifiers", fmt: v => (v || []).join(", ") }, { key: "reason", label: "reason", cls: "wrap" }], intents.map((it, i) => ({ i: i + 1, ...it }))) : h("p", { class: "note" }, "No intents."), true));

    // 3. Structuring.
    const structures = rec.structures || [];
    panel.appendChild(sec("Structuring", structures.length, structures.length ? structures.map((st, i) => {
      const goals = (st.structure || {}).goals || {};
      const preds = (st.structure || {}).predicates || {};
      return h("div", { class: "card" }, h("div", null, h("b", null, `Intent ${i + 1}: `), st.text),
        st.error ? h("div", { class: "quote" }, h("span", { class: "badge fail" }, "error"), " ", st.error) : null,
        st.clarification || st.question ? h("div", { class: "quote" }, h("span", { class: "badge warn" }, "question"), " ", st.clarification || st.question) : null,
        st.structure ? kvList([["structure", st.structure.structure], ["qualifiers", (st.structure.qualifiers || []).join(", ")], ["device references", st.device_references ? JSON.stringify(st.device_references) : null], ["context unavailable", st.context_unavailable ? "yes" : null]]) : null,
        Object.keys(goals).length ? table([{ key: "id", label: "goal" }, { key: "intent_text", label: "intent text", cls: "wrap" }, { key: "goal_kind", label: "kind" }, { key: "goal_specificity", label: "specificity" }, { key: "location_class", label: "location" }, { key: "device_class", label: "device class" }, { key: "rest", label: "other fields", cls: "wrap", fmt: (v, g) => Object.entries(g).filter(([k]) => !["id", "intent_text", "goal_kind", "goal_specificity", "location_class", "device_class"].includes(k) && g[k] != null && g[k] !== "" && !(Array.isArray(g[k]) && !g[k].length)).map(([k, x]) => `${k}=${typeof x === "object" ? JSON.stringify(x) : x}`).join("; ") }],
          Object.entries(goals).map(([id, g]) => ({ id, ...g }))) : null,
        Object.keys(preds).length ? h("div", null, h("h4", null, "Predicates"), h("pre", null, jsonPretty(preds))) : null,
        jsonDetails("structure JSON", st));
    }) : h("p", { class: "note" }, "No structures."), true));

    // 3b. State queries (qt1).
    const queries = rec.queries || [];
    if (queries.length) panel.appendChild(sec("State queries", queries.length, queries.map((q, i) => h("div", { class: "card" }, h("div", null, h("b", null, `Query ${i + 1} (${q.kind || "state"}): `), q.text), q.answer ? h("div", { class: "quote" }, q.answer) : null, q.error ? h("div", { class: "quote" }, h("span", { class: "badge fail" }, "error"), " ", q.error) : null, jsonDetails("query JSON", q)))));

    // 4. Planning.
    const plans = rec.plans || [];
    panel.appendChild(sec("Planning: plan entries", plans.length, plans.length ? plans.map((p, i) => {
      const badge = p.outcome === "plan" ? "pass" : p.outcome === "error" ? "fail" : "warn";
      return h("div", { class: "card" }, h("div", null, h("b", null, `Entry ${i + 1} `), h("span", { class: "badge " + badge }, p.outcome || "?"), " ", h("span", { class: "muted small" }, [p.route ? `route ${p.route}` : null, p.source ? `source ${p.source}` : null, p.execution_mode ? `mode ${p.execution_mode}` : null, p.long_running ? "long-running" : null].filter(Boolean).join(" · "))),
        p.intent && p.intent.text_intent ? h("p", { class: "note" }, p.intent.text_intent) : null,
        p.error ? h("div", { class: "quote" }, h("b", null, p.error), p.explanation ? ": " + p.explanation : "") : (p.explanation ? h("div", { class: "quote" }, p.explanation) : null),
        p.devices && p.devices.length ? h("p", { class: "small" }, "devices: ", p.devices.join(", ")) : null,
        p.tree ? h("div", null, h("h4", null, "Behaviour tree"), btTree(p.tree)) : null,
        p.profile && Object.keys(p.profile).length ? h("div", null, h("h4", null, "Profile"), h("pre", null, jsonPretty(p.profile))) : null,
        p.runtime && ((p.runtime.reads || []).length || (p.runtime.derived || []).length || p.runtime.clock) ? h("p", { class: "small" }, "runtime: ", JSON.stringify(p.runtime)) : null,
        jsonDetails("plan entry JSON", p));
    }) : h("p", { class: "note" }, "No plan entries."), true));

    // 5. Execution.
    const ex = rec.execution || {};
    const actions = ex.actions || [];
    const exPlans = ex.plans || [];
    const exBody = [];
    if (exPlans.length) exBody.push(table([{ key: "plan", label: "plan", num: true }, { key: "source", label: "source" }, { key: "status", label: "status", fmt: v => h("span", { class: "badge " + (v === "SUCCESS" ? "pass" : v === "FAILURE" ? "fail" : "warn") }, v || "?") }, { key: "ticks", label: "ticks", num: true }, { key: "error", label: "error", cls: "wrap" }], exPlans));
    exBody.push(kvList([["mode", ex.mode], ["real seconds", isNum(ex.real_seconds) ? ex.real_seconds.toFixed(2) : null], ["virtual seconds", ex.virtual_seconds], ["steps", ex.steps], ["horizon / end tick", ex.horizon_tick != null ? `${ex.horizon_tick} / ${ex.end_tick}` : null], ["details", ex.details ? Object.entries(ex.details).filter(([k]) => k !== "workflows").map(([k, v]) => `${k}=${v}`).join("  ") : null]]));
    if (actions.length) exBody.push(h("h4", null, `Action timeline (${actions.length})`), table([{ key: "time", label: "virtual time" }, { key: "tick", label: "tick", num: true }, { key: "plan", label: "plan", num: true }, { key: "name", label: "action", cls: "mono" }, { key: "parameters", label: "parameters", fmt: v => v ? JSON.stringify(v) : "" }, { key: "how", label: "how" }, { key: "success", label: "result", fmt: (v, a) => h("span", { class: "badge " + (v ? "pass" : "fail") }, (v ? "ok" : "failed") + (a.status_code ? ` ${a.status_code}` : "")) }], actions));
    else exBody.push(h("p", { class: "note" }, "No action was sent."));
    const wfs = (ex.details || {}).workflows || [];
    if (wfs.length) exBody.push(h("h4", null, `Scheduled workflows (${wfs.length})`), wfs.map(w => h("div", { class: "card" }, kvList([["workflow", w.workflow_id], ["plan", w.plan], ["start", w.start_time], ["status", w.status]]), h("pre", null, jsonPretty(w.steps)))));
    const timeline = rec.timeline || [];
    if (timeline.length) { const keys = [...new Set(timeline.flatMap(t => Object.keys(t.values || {})))]; exBody.push(h("h4", null, `Tracked values (${timeline.length} ticks)`), table([{ key: "tick", label: "tick", num: true }, ...keys.map(k => ({ key: k, label: k, fmt: v => v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v) }))], timeline.map(t => ({ tick: t.tick, ...(t.values || {}) })))); }
    const snaps = rec.snapshots || [];
    if (snaps.length) exBody.push(h("details", { class: "json" }, h("summary", null, `Room readings per snapshot (${snaps.length}; devices reduced to the checked ones)`), h("pre", null, jsonPretty(snaps))));
    if (rec.final_home_state) exBody.push(jsonDetails("Final home state (rooms, checked devices)", rec.final_home_state));
    panel.appendChild(sec("Execution", actions.length ? `${actions.length} actions` : null, exBody, true));

    // 6. Tool log.
    const tools = rec.tool_log && rec.tool_log.length ? rec.tool_log : (rec.tools_invoked || []);
    panel.appendChild(sec("Tool log", tools.length, tools.length ? table([{ key: "tool", label: "tool" }, { key: "params", label: "params", fmt: v => v ? JSON.stringify(v) : "" }, { key: "source", label: "source" }, { key: "outcome", label: "outcome", fmt: v => v ? h("span", { class: "badge " + (v.ok ? "pass" : "fail") }, (v.ok ? "ok" : "failed") + (v.status_code ? ` ${v.status_code}` : "") + (v.error_type ? ` ${v.error_type}` : "")) : "" }], tools) : h("p", { class: "note" }, "No tool calls recorded.")));

    // 7. Response.
    panel.appendChild(sec("Reply to the user", null, h("div", { class: "quote" }, rec.response || h("span", { class: "muted" }, "(no reply)")), true));

    // 8. Meter.
    const meterRows = PHASES.filter(p => stages[p]).map((p, i) => ({ phase: p, color: phaseColor(PHASES.indexOf(p)), ...stages[p] }));
    const bar = h("div", { class: "meter-bar" }, meterRows.filter(r => r.seconds > 0).map(r => h("span", { style: { width: (100 * r.seconds / Math.max(totalS, 1e-9)) + "%", background: r.color }, title: `${r.phase} ${secs(r.seconds)}` })));
    const calls = (rec.meter || {}).calls || [];
    panel.appendChild(sec("Meter: seconds, calls and tokens per phase", null, Object.keys(stages).length ? [bar, table([{ key: "phase", label: "phase", fmt: (v, r) => h("span", null, h("span", { class: "dot", style: { background: r.color } }), v) }, { key: "seconds", label: "seconds", num: true, fmt: v => num(v, 2) }, { key: "llm_seconds", label: "LLM s", num: true, fmt: v => num(v, 2) }, { key: "calls", label: "calls", num: true }, { key: "tokens_in", label: "tokens in", num: true, fmt: int }, { key: "tokens_out", label: "tokens out", num: true, fmt: int }], [...meterRows, { phase: "total", color: "transparent", seconds: totalS, llm_seconds: meterRows.reduce((a, r) => a + (r.llm_seconds || 0), 0), calls: totalCalls, tokens_in: tin, tokens_out: tout }], { rowClass: r => r.phase === "total" ? "total" : "" }),
      calls.length ? h("details", { class: "json" }, h("summary", null, `LLM calls (${calls.length})`), table([{ key: "step", label: "step" }, { key: "model", label: "model" }, { key: "ok", label: "ok", fmt: v => v ? "✓" : "✗" }, { key: "seconds", label: "s", num: true, fmt: v => num(v, 2) }, { key: "tokens_in", label: "in", num: true, fmt: int }, { key: "tokens_out", label: "out", num: true, fmt: int }, { key: "error", label: "error", cls: "wrap" }], calls)) : null,
      rec.discovery_seconds != null ? h("p", { class: "note" }, `Discovery took ${num(rec.discovery_seconds, 2)} s (not counted).`) : null] : h("p", { class: "note" }, "No meter stages in this record (written before the meter kept them)."), true));

    // 9. Evaluation.
    const evBody = [];
    if (!rec._eval) evBody.push(h("p", { class: "note" }, "Not scored yet (no .eval.json)."));
    else {
      evBody.push(kvList([["score", scoreBadge(ev.score, ev.error_type)], ["error type", ev.error_type], ["check", ev.check], ["detail", ev.detail]]));
      if (ec) evBody.push(h("h4", null, "Execution check (infeasible part)"), kvList([["score", scoreBadge(ec.score, ec.error_type)], ["error type", ec.error_type], ["answered impossible or asked", ec.answered_impossible_or_clarify == null ? null : (ec.answered_impossible_or_clarify ? "yes" : "no")], ["offending actions", (ec.offending_actions || []).length ? JSON.stringify(ec.offending_actions) : "none"]]));
      if (ev.judge) evBody.push(h("h4", null, "Judge panel"), kvList([["score", scoreBadge(ev.judge.score, ev.judge.error_type)], ["votes", (ev.judge.judge || []).join(" / ") + " (A = acceptable, B = not)"], ["error type", ev.judge.error_type]]));
      const req = ev.required_actions || (ev.judge || {}).required_actions || (ec || {}).required_actions;
      if (req && req.length) evBody.push(h("h4", null, "Required tool calls"), table([{ key: "tool", label: "tool" }, { key: "params", label: "params", fmt: v => JSON.stringify(v) }, { key: "invoked", label: "invoked", fmt: v => h("span", { class: "badge " + (v ? "pass" : "fail") }, v ? "yes" : "no") }], req));
      if (ev.targets_eval) evBody.push(h("h4", null, "SimuHome targets"), table([{ key: "device_id", label: "device" }, { key: "ok", label: "ok", fmt: v => h("span", { class: "badge " + (v ? "pass" : "fail") }, v ? "yes" : "no") }, { key: "diffs", label: "diffs", cls: "wrap", fmt: v => (v || []).map(d => `${d.attribute}: expected ${JSON.stringify(d.expected)}, actual ${JSON.stringify(d.actual)}`).join("; ") }], ev.targets_eval));
      if (ev.goals && !checks.length) evBody.push(h("h4", null, "Goals"), h("pre", null, jsonPretty(ev.goals)));
      if (ev.homebench) evBody.push(h("h4", null, "HomeBench"), h("pre", null, jsonPretty(ev.homebench)));
      evBody.push(jsonDetails("evaluation JSON", ev));
    }
    panel.appendChild(sec("Evaluation verdict", null, evBody, true));

    // 10. Warnings.
    const warns = warnings(rec, rec._eval);
    panel.appendChild(sec("Warnings and anomalies", warns.length, warns.length ? h("ul", null, warns.map(w => h("li", null, w))) : h("p", { class: "note" }, "Nothing flagged."), warns.length > 0));

    // 11. Trace.
    const steps = rec.steps || [];
    panel.appendChild(sec("Reasoning trace (steps)", steps.length, steps.length ? steps.map(st => h("details", { class: "json" }, h("summary", null, `${st.step}. ${st.action}`, h("span", { class: "muted" }, ` — ${clip(st.thought, 110)}`)), h("h4", null, "input"), h("pre", null, typeof st.action_input === "string" ? st.action_input : jsonPretty(st.action_input)), h("h4", null, "observation"), h("pre", null, prettyMaybeJson(st.observation)))) : h("p", { class: "note" }, "No trace.")));

    // 12. Raw.
    panel.appendChild(sec("Raw record (compacted)", null, jsonDetails("record JSON", rec)));
    open();
    function open() { panel.classList.add("open"); scrim.classList.add("open"); document.body.style.overflow = "hidden"; panel.scrollTop = 0; }
  }
  function prettyMaybeJson(v) {
    if (v == null) return "";
    if (typeof v !== "string") return jsonPretty(v);
    const t = v.trim();
    if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) { try { return jsonPretty(JSON.parse(t)); } catch (e) { /* plain text */ } }
    return v;
  }
  function stat(label, value) { return h("div", { class: "stat" }, h("div", { class: "l" }, label), h("div", { class: "v" }, String(value))); }

  // ---------------------------------------------------------------- shell and routing
  function themeButton() {
    const modes = ["auto", "light", "dark"];
    let mode = tryLoad("report-theme") || "auto";
    const btn = h("button", { type: "button", title: "Colour theme" });
    function apply() { if (mode === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = mode; btn.textContent = `theme: ${mode}`; }
    btn.addEventListener("click", () => { mode = modes[(modes.indexOf(mode) + 1) % modes.length]; tryStore("report-theme", mode); apply(); route(); });
    apply();
    return btn;
  }
  const header = h("header", { class: "top" }, h("div", null, h("h1", null, DATA.title), h("div", { class: "sub" }, `${CONFIGS.length} configs · ${CONFIGS.reduce((a, c) => a + c.n, 0)} episode rows · built ${DATA.built}`)), h("nav", { class: "tabs" }), themeButton());
  const main = h("main");
  app.replaceChildren(header, main);
  function renderTabs(active) {
    const nav = header.querySelector("nav.tabs");
    const tabs = [["#overview", "Overview"], ...CONFIGS.map(c => [`#config/${encodeURIComponent(c.key)}`, h("span", null, h("span", { class: "swatch", style: { background: colorOf(c) } }), c.title)]), ["#episodes", "Episodes"]];
    nav.replaceChildren(...tabs.map(([href, label]) => h("a", { href, class: href === active ? "active" : "" }, label)));
  }
  function route() {
    const hash = location.hash || "#overview";
    if (hash.startsWith("#episode/")) { const id = decodeURIComponent(hash.slice(9)); if (!main.children.length) renderPage(lastPageHash); showEpisode(id); return; }
    closePanelSilent();
    lastPageHash = hash;
    renderPage(hash);
  }
  function closePanelSilent() { panel.classList.remove("open"); scrim.classList.remove("open"); document.body.style.overflow = ""; }
  function renderPage(hash) {
    hideTip();
    const [path, query] = hash.split("?");
    const params = new URLSearchParams(query || "");
    let page;
    if (path.startsWith("#config/")) { const key = decodeURIComponent(path.slice(8)); const cfg = CONFIGS.find(c => c.key === key); page = cfg ? configPage(cfg) : h("p", null, "Unknown configuration."); renderTabs(path); }
    else if (path === "#episodes") { page = episodesPage(params); renderTabs(path); }
    else { page = overview(); renderTabs("#overview"); }
    main.replaceChildren(page);
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);
  route();
})();
