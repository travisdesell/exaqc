/*
 * EXAQC Artifacts viewer.
 *
 * A dependency-free single-page app over the viewer's read-only JSON API
 * (src/utils/artifact_viewer/server.py). Charts are drawn with the vendored
 * uPlot. Pages are chosen by the URL hash:
 *
 *   #/                          the run list
 *   #/groups                    compare runs (grouped by --groups)
 *   #/run/<run>                 a run: progress/genealogy chart, genome table
 *   #/run/<run>/genome/<n>      ... with genome <n> open in the detail panel
 *   #/run/<run>/compare/<a>/<b> two genomes side by side
 */
"use strict";

(() => {
  /** How often live pages check for new genomes. */
  const POLL_INTERVAL_MS = 5000;
  /** Genome table rows loaded at a time, as the table is scrolled. */
  const PAGE_SIZE = 50;
  /** More genome rows load when the bottom of the table comes within this many pixels of the window. */
  const LOAD_MARGIN_PX = 600;
  /** Gap left above the genome table (below the sticky top bar) when a re-sort or filter scrolls back to it. */
  const TABLE_SCROLL_OFFSET_PX = 64;
  /** The chart column's default share of the run page's width, in percent (matches the run-body grid in app.css). */
  const CHART_WIDTH_DEFAULT_PERCENT = 38;
  /** Narrowest the chart column can be dragged (matches the run-body grid in app.css). */
  const MIN_CHART_COLUMN_PX = 260;
  /** Narrowest the genome table can be dragged (matches the run-body grid in app.css). */
  const MIN_TABLE_COLUMN_PX = 320;
  /** Width of the draggable divider between the chart and the table (matches app.css). */
  const COLUMN_RESIZER_PX = 14;
  /** Arrow-key step when resizing the columns from the keyboard, in percent (Shift: STEP_LARGE). */
  const COLUMN_RESIZE_STEP = 2;
  const COLUMN_RESIZE_STEP_LARGE = 10;
  /** Where this browser remembers the chart column's width. */
  const CHART_WIDTH_STORAGE_KEY = "exaqc-artifacts.chartWidth";
  /** A run whose last genome was saved this recently (seconds) is shown as live. */
  const LIVE_WINDOW_S = 120;
  /** Radius (CSS px) within which hovering or clicking picks a chart point. */
  const HIT_RADIUS = 14;
  /** Children listed in the lineage tab before the rest are summarized. */
  const MAX_CHILDREN_SHOWN = 200;
  /** Ancestors drawn per generation in the ancestry graph. */
  const MAX_ANCESTORS_PER_GENERATION = 30;
  /** Categorical slots in the palette; later categories fold into "other". */
  const CATEGORICAL_SLOTS = 8;

  /**
   * Line weights for the genome chart's two views. Both draw the best-so-far
   * line and every parent->child link; Progress emphasizes the best-so-far
   * line, Genealogy the links. `linkAlpha` is [minimum, maximum, coverage]:
   * links fade as they crowd the plot, so the opacity is coverage divided by
   * how many times over the drawn links would cover the plot area, clamped to
   * the range. A few links stay at the maximum; tens of thousands stay legible
   * instead of filling the plot solid.
   */
  const CHART_EMPHASIS = {
    progress: { bestWidth: 3, bestInk: "--text-primary", linkWidth: 0.75, linkAlpha: [0.01, 0.2, 0.15], lineageWidth: 1.5, lineageInk: "--text-secondary" },
    genealogy: { bestWidth: 1, bestInk: "--text-secondary", linkWidth: 1.75, linkAlpha: [0.02, 0.8, 0.6], lineageWidth: 3, lineageInk: "--text-primary" },
  };

  /** Chart options that only restyle the chart, so changing them keeps the zoom. */
  const STYLE_ONLY_OPTIONS = new Set(["mode", "highlightLineage"]);

  /** Why the seed genome, a parent of every initial genome, has no page of its own. */
  const SEED_EXPLANATION =
    "The seed genome: the empty starting circuit every initial genome was mutated from. " +
    "It is never evaluated, so it has no fitness and is not stored in the archive.";

  /**
   * Height of the run page's genome chart: whatever is left of the window below
   * the chart's controls and legend, since the chart stays in view (sticky) while
   * the genome table beside it scrolls. When the columns stack (narrow windows)
   * a fixed height is used.
   */
  function genomeChartHeight(host) {
    const sticky = host.closest(".chart-sticky");
    if (!sticky || getComputedStyle(sticky).position !== "sticky") return 520;
    const offset = host.getBoundingClientRect().top - sticky.getBoundingClientRect().top;
    // 60px: where the chart sticks below the top bar; 44px: the panel's padding and a margin below
    return Math.max(320, window.innerHeight - 60 - offset - 44);
  }

  /** The chart column width (percent) this browser last chose, or null if none (or storage is unavailable). */
  function readChartWidth() {
    try {
      const saved = localStorage.getItem(CHART_WIDTH_STORAGE_KEY);
      const value = saved === null ? NaN : Number(saved);
      return value >= 10 && value <= 90 ? value : null;
    } catch {
      return null;
    }
  }

  /** Remembers the chart column width (percent) in this browser, or forgets it when `percent` is null. */
  function storeChartWidth(percent) {
    try {
      if (percent === null) localStorage.removeItem(CHART_WIDTH_STORAGE_KEY);
      else localStorage.setItem(CHART_WIDTH_STORAGE_KEY, percent.toFixed(2));
    } catch {
      // storage can be unavailable (private windows, blocked site data): the split just isn't remembered
    }
  }

  /**
   * Makes the divider between the run page's chart and table columns. Dragging
   * it (or focusing its grip and pressing the arrow keys, Home or End) trades
   * width between the columns, and double-clicking it restores the default split.
   *
   * The chart column's share of the width is the run body's `--chart-width`
   * custom property, a percentage so the split holds as the window resizes; the
   * grid in app.css keeps both columns above their minimum widths whatever it is
   * set to. A chosen width is remembered in this browser for later run pages.
   *
   * @param {HTMLElement} runBody The grid holding the two columns.
   * @param {HTMLElement} chartPanel The chart column, whose width is measured.
   * @returns {{node: HTMLElement, destroy: Function}} The divider element, and a
   *   function that ends a drag in progress when the page goes away.
   */
  function createColumnResizer(runBody, chartPanel) {
    // The whole divider is the drag target, but it runs the full height of the table, so
    // keyboard focus goes to its grip instead: the grip is always in view, so focusing
    // it never scrolls the page.
    const grip = h("span", {
      class: "column-resizer-grip",
      role: "separator",
      tabindex: 0,
      "aria-orientation": "vertical",
      "aria-label": "Resize the chart and genome table columns",
      "aria-valuemin": 0,
      "aria-valuemax": 100,
    });
    const node = h("div", { class: "column-resizer", title: "Drag to resize the chart and table · double-click to reset" }, grip);

    /** The chart column's current share of the run body's width, in percent. */
    const currentPercent = () => (chartPanel.getBoundingClientRect().width / Math.max(runBody.clientWidth, 1)) * 100;

    /** Sets the chart column's share of the width, clamped so both columns keep their minimum widths. */
    function setPercent(percent, { remember = false } = {}) {
      const width = Math.max(runBody.clientWidth, 1);
      const minimum = (MIN_CHART_COLUMN_PX / width) * 100;
      const maximum = Math.max(minimum, ((width - COLUMN_RESIZER_PX - MIN_TABLE_COLUMN_PX) / width) * 100);
      const clamped = Math.max(minimum, Math.min(maximum, percent));
      runBody.style.setProperty("--chart-width", `${clamped.toFixed(2)}%`);
      grip.setAttribute("aria-valuenow", Math.round(clamped));
      if (remember) storeChartWidth(clamped);
    }

    // a drag in progress: the pointer, where it started, the chart width then, and a pending animation frame
    let drag = null;
    const dragPercent = () => ((drag.startWidth + drag.x - drag.startX) / Math.max(runBody.clientWidth, 1)) * 100;

    node.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      node.setPointerCapture(event.pointerId);
      drag = { pointerId: event.pointerId, startX: event.clientX, x: event.clientX, startWidth: chartPanel.getBoundingClientRect().width, frame: 0 };
      document.body.classList.add("resizing-columns");
    });
    node.addEventListener("pointermove", (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      drag.x = event.clientX;
      // resize at most once a frame: the chart redraws (every genome and link) at each new width
      if (!drag.frame)
        drag.frame = requestAnimationFrame(() => {
          if (!drag) return;
          drag.frame = 0;
          setPercent(dragPercent());
        });
    });

    /** Finishes a drag at the pointer's last position, remembering the width if the divider moved. */
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      cancelAnimationFrame(drag.frame);
      if (drag.x !== drag.startX) setPercent(dragPercent(), { remember: true });
      drag = null;
      document.body.classList.remove("resizing-columns");
    }
    node.addEventListener("pointerup", endDrag);
    node.addEventListener("pointercancel", endDrag);
    node.addEventListener("lostpointercapture", endDrag);

    node.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? COLUMN_RESIZE_STEP_LARGE : COLUMN_RESIZE_STEP;
      const target = { ArrowLeft: currentPercent() - step, ArrowRight: currentPercent() + step, Home: 0, End: 100 }[event.key];
      if (target === undefined) return;
      event.preventDefault();
      setPercent(target, { remember: true });
    });
    node.addEventListener("dblclick", () => {
      runBody.style.removeProperty("--chart-width");
      grip.setAttribute("aria-valuenow", CHART_WIDTH_DEFAULT_PERCENT);
      storeChartWidth(null);
    });

    // start from the width this browser last chose (the grid keeps it within bounds until it is dragged)
    const saved = readChartWidth();
    if (saved !== null) runBody.style.setProperty("--chart-width", `${saved}%`);
    grip.setAttribute("aria-valuenow", Math.round(saved ?? CHART_WIDTH_DEFAULT_PERCENT));

    return {
      node,
      destroy() {
        if (drag) cancelAnimationFrame(drag.frame);
        drag = null;
        document.body.classList.remove("resizing-columns");
      },
    };
  }

  const app = document.getElementById("app");
  const breadcrumbs = document.getElementById("breadcrumbs");

  /** The current page: an object with a `kind` and a `destroy()` method. */
  let currentPage = null;

  // ---------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------

  /** Creates an HTML element with attributes, event handlers and children. */
  function h(tag, attributes, ...children) {
    const node = document.createElement(tag);
    applyAttributes(node, attributes);
    appendChildren(node, children);
    return node;
  }

  /** Creates an SVG element with attributes and children. */
  function s(tag, attributes, ...children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    applyAttributes(node, attributes);
    appendChildren(node, children);
    return node;
  }

  /** Sets attributes, `text`, `class` and `on<event>` handlers on a node. */
  function applyAttributes(node, attributes) {
    for (const [key, value] of Object.entries(attributes || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "text") node.textContent = value;
      else if (key === "class") node.setAttribute("class", value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
  }

  /** Appends children, skipping empty values and turning others into text. */
  function appendChildren(node, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }

  /** Replaces a node's children, skipping empty values and turning others into text. */
  function setChildren(node, ...children) {
    node.replaceChildren();
    appendChildren(node, children);
  }

  /** Fetches a JSON API path, throwing the server's error message on failure. */
  async function api(path) {
    const response = await fetch(path, { cache: "no-store" });
    let body = null;
    try {
      body = await response.json();
    } catch (error) {
      body = null;
    }
    if (!response.ok) throw new Error((body && body.error) || `${response.status} ${response.statusText}`);
    return body;
  }

  /** Reads a CSS custom property (palette token). */
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /** The color of categorical palette slot `slot` (1-8). */
  function slotColor(slot) {
    return token(`--series-${slot}`);
  }

  /** A hex color with an alpha channel, as an rgba() string. */
  function withAlpha(hex, alpha) {
    const value = hex.replace("#", "");
    const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
    const number = parseInt(full, 16);
    return `rgba(${(number >> 16) & 255}, ${(number >> 8) & 255}, ${number & 255}, ${alpha})`;
  }

  /** Mixes two hex colors; `t` = 0 gives `a`, 1 gives `b`. */
  function mixColors(a, b, t) {
    const parse = (hex) => {
      const n = parseInt(hex.replace("#", ""), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    };
    const [ca, cb] = [parse(a), parse(b)];
    const mixed = ca.map((channel, i) => Math.round(channel + (cb[i] - channel) * t));
    return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
  }

  function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  /** Whether smaller values of a fitness key are better (loss-like keys). */
  function lowerIsBetter(key) {
    return /loss/i.test(key);
  }

  /** Formats a number compactly for tables and tooltips. */
  function formatNumber(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number") return typeof value === "object" ? JSON.stringify(value) : String(value);
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return value.toLocaleString();
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)) return value.toExponential(3);
    return value.toFixed(magnitude >= 100 ? 2 : 4);
  }

  function formatTime(seconds) {
    return isNumber(seconds) ? new Date(seconds * 1000).toLocaleString() : "—";
  }

  function formatAgo(seconds) {
    if (!isNumber(seconds)) return "—";
    const elapsed = Math.max(0, Date.now() / 1000 - seconds);
    if (elapsed < 60) return `${Math.round(elapsed)}s ago`;
    if (elapsed < 3600) return `${Math.round(elapsed / 60)}m ago`;
    if (elapsed < 86400) return `${Math.round(elapsed / 3600)}h ago`;
    return `${Math.round(elapsed / 86400)}d ago`;
  }

  function isLive(summary) {
    return isNumber(summary.last_saved_at) && Date.now() / 1000 - summary.last_saved_at < LIVE_WINDOW_S;
  }

  function label(text) {
    return String(text ?? "unknown").replace(/_/g, " ");
  }

  function formatQubits(qubits) {
    return (qubits || []).map((qubit) => (Array.isArray(qubit) ? `${qubit[0]}[${qubit[1]}]` : String(qubit))).join(", ");
  }

  function formatParameters(parameters) {
    const entries = Object.entries(parameters || {});
    return entries.length ? entries.map(([name, value]) => `${name}=${formatNumber(value)}`).join(", ") : "—";
  }

  function genomeHref(run, number) {
    return `#/run/${run}/genome/${number}`;
  }

  function notice(message, isError = false) {
    return h("p", { class: isError ? "notice error" : "notice", text: message });
  }

  function setBreadcrumbs(items) {
    setChildren(breadcrumbs,
      ...items.flatMap((item, i) => [
        ...(i > 0 ? [h("span", { class: "separator", "aria-hidden": "true" })] : []),
        item.href ? h("a", { href: item.href, text: item.label }) : h("span", { text: item.label }),
      ])
    );
  }

  function copyButton(text) {
    const button = h("button", { type: "button", text: "Copy" });
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch (error) {
        button.textContent = "Copy failed";
      }
      setTimeout(() => (button.textContent = "Copy"), 1500);
    });
    return button;
  }

  function select(options, value, onChange, ariaLabel) {
    const node = h(
      "select",
      { "aria-label": ariaLabel, onchange: (event) => onChange(event.target.value) },
      options.map(([optionValue, optionLabel]) => h("option", { value: optionValue, text: optionLabel }))
    );
    node.value = value;
    return node;
  }

  function segmented(options, value, onChange) {
    return h(
      "span",
      { class: "segmented", role: "group" },
      options.map(([optionValue, optionLabel]) =>
        h("button", {
          type: "button",
          "aria-pressed": String(optionValue === value),
          text: optionLabel,
          onclick: () => onChange(optionValue),
        })
      )
    );
  }

  function checkbox(text, checked, onChange) {
    const input = h("input", { type: "checkbox", onchange: (event) => onChange(event.target.checked) });
    input.checked = checked;
    return h("label", {}, input, text);
  }

  // ---------------------------------------------------------------------------
  // Color encodings
  // ---------------------------------------------------------------------------

  /** Folds a generating operator into its family, so colors stay within the palette. */
  function operatorFamily(operator) {
    if (operator === "binary_crossover") return "binary crossover";
    if (operator === "n_ary_crossover") return "n-ary crossover";
    if (operator === "exponential_crossover") return "exponential crossover";
    return operator ? "mutation" : "unknown";
  }

  const FAMILY_SLOTS = { mutation: 1, "binary crossover": 2, "n-ary crossover": 3, "exponential crossover": 4 };

  function familyColor(family) {
    return FAMILY_SLOTS[family] ? slotColor(FAMILY_SLOTS[family]) : token("--text-muted");
  }

  /**
   * The categories a color encoding splits genomes into, each with a fixed
   * palette color: insert type (global best stands out; discarded recedes),
   * operator family, or island.
   */
  function categoriesFor(encoding, points) {
    const muted = token("--text-muted");
    let categories;
    if (encoding === "insert_type") {
      categories = [
        { key: "global_best", label: "global best", color: slotColor(2) },
        { key: "local_best", label: "local best", color: slotColor(3) },
        { key: "inserted", label: "inserted", color: slotColor(1) },
        { key: "discarded", label: "discarded", color: muted },
      ];
    } else if (encoding === "family") {
      categories = Object.keys(FAMILY_SLOTS).map((family) => ({ key: family, label: family, color: familyColor(family) }));
    } else {
      const islands = [...new Set(points.island.filter((island) => island !== null))].sort((a, b) => a - b);
      categories = islands.slice(0, CATEGORICAL_SLOTS - 1).map((island, i) => ({ key: String(island), label: `island ${island}`, color: slotColor(i + 1) }));
      if (islands.length > CATEGORICAL_SLOTS - 1) categories.push({ key: "other", label: "other islands", color: muted });
    }
    categories.push({ key: "unknown", label: encoding === "island" ? "no island" : "unknown", color: muted });
    return categories;
  }

  function categoryKey(encoding, points, i, known) {
    let key;
    if (encoding === "insert_type") key = points.insert_type[i];
    else if (encoding === "family") key = operatorFamily(points.operator[i]);
    else key = points.island[i] === null ? null : String(points.island[i]);
    if (key === null || key === undefined) return "unknown";
    if (known.has(key)) return key;
    return encoding === "island" ? "other" : "unknown";
  }

  function passesFilters(points, i, filters) {
    if (filters.insert_type && points.insert_type[i] !== filters.insert_type) return false;
    if (filters.crossover_type && points.crossover_type[i] !== filters.crossover_type) return false;
    if (filters.island !== "" && filters.island !== undefined && String(points.island[i]) !== String(filters.island)) return false;
    if (filters.generated_by && !(points.generated_by[i] || []).includes(filters.generated_by)) return false;
    return true;
  }

  function legendItem(color, text, line = false) {
    return h("span", {}, h("span", { class: line ? "swatch line" : "swatch", style: `background:${color}` }), text);
  }

  function axisStyle(labelText) {
    return {
      label: labelText,
      stroke: token("--text-secondary"),
      grid: { stroke: token("--gridline"), width: 1 },
      ticks: { stroke: token("--baseline"), width: 1 },
    };
  }

  // ---------------------------------------------------------------------------
  // The genome chart (progress scatter and whole-run genealogy)
  // ---------------------------------------------------------------------------

  /**
   * Creates the run page's chart: every genome as a point, with genome number
   * running down the chart and a fitness key across it, every parent->child link
   * drawn beneath the points, and a best-so-far line joining the genomes that
   * improved the best value. `config.mode`
   * ("progress" or "genealogy") decides which of the line and the links is
   * emphasized (see CHART_EMPHASIS).
   */
  function createGenomeChart(container, config) {
    const legend = h("div", { class: "chart-legend" });
    const host = h("div", { class: "chart-host" });
    const tooltip = h("div", { class: "chart-tooltip", hidden: true });
    const empty = notice("");
    empty.hidden = true;
    setChildren(container, legend, host, empty);

    let points = null;
    let links = null;
    let selected = null;
    let plot = null;
    let model = null;
    let zoomed = false;
    let lineageCache = { genome: null, value: null };

    function buildModel() {
      const count = points.genome_number.length;
      const xs = points.genome_number;
      const ys = points.y.map((value) => (isNumber(value) ? value : null));
      const categories = categoriesFor(config.encoding, points);
      const known = new Set(categories.map((category) => category.key));
      const categoryIndex = new Map(categories.map((category, i) => [category.key, i]));
      const category = new Int32Array(count);
      const visible = new Uint8Array(count);
      const indexOf = new Map();
      const better = lowerIsBetter(config.yKey) ? (a, b) => a < b : (a, b) => a > b;

      let bestIndex = -1;
      for (let i = 0; i < count; i++) {
        indexOf.set(xs[i], i);
        category[i] = categoryIndex.get(categoryKey(config.encoding, points, i, known));
        visible[i] = ys[i] !== null && passesFilters(points, i, config.filters) ? 1 : 0;
        if (visible[i] && (bestIndex < 0 || better(ys[i], ys[bestIndex]))) bestIndex = i;
      }

      let parentsOf = null;
      let childrenOf = null;
      if (links) {
        parentsOf = new Map();
        childrenOf = new Map();
        for (let k = 0; k < links.child.length; k++) {
          const child = links.child[k];
          const parent = links.parent[k];
          if (!parentsOf.has(child)) parentsOf.set(child, []);
          parentsOf.get(child).push(parent);
          if (!childrenOf.has(parent)) childrenOf.set(parent, []);
          childrenOf.get(parent).push(child);
        }
        if (config.hideDeadEnds) {
          for (let i = 0; i < count; i++) {
            if (visible[i] && !childrenOf.has(xs[i]) && xs[i] !== selected && i !== bestIndex) visible[i] = 0;
          }
        }
      }

      // The best-so-far line only has values at the genomes that improved on
      // every earlier one, so it connects those genomes with straight lines.
      const best = new Array(count).fill(null);
      let current = null;
      let visibleCount = 0;
      for (let i = 0; i < count; i++) {
        if (!visible[i]) continue;
        visibleCount++;
        if (current === null || better(ys[i], current)) {
          current = ys[i];
          best[i] = current;
        }
      }

      const data = [xs];
      categories.forEach((_, categoryNumber) => {
        data.push(ys.map((y, i) => (visible[i] && category[i] === categoryNumber ? y : null)));
      });
      data.push(best);

      let edgesByFamily = null;
      if (links) {
        edgesByFamily = new Map();
        for (let k = 0; k < links.child.length; k++) {
          const childIndex = indexOf.get(links.child[k]);
          const parentIndex = indexOf.get(links.parent[k]);
          if (childIndex === undefined || parentIndex === undefined || !visible[childIndex] || !visible[parentIndex]) continue;
          const family = operatorFamily(points.operator[childIndex]);
          if (!edgesByFamily.has(family)) edgesByFamily.set(family, []);
          edgesByFamily.get(family).push(parentIndex, childIndex);
        }
      }

      return { count, xs, ys, categories, category, visible, visibleCount, indexOf, parentsOf, childrenOf, edgesByFamily, data };
    }

    /** The selected genome's ancestors and descendants, and the links between them. */
    function lineage() {
      // a genome that isn't plotted (such as the seed, whose lineage is the whole run) highlights nothing
      if (!links || selected === null || !config.highlightLineage || !model.indexOf.has(selected)) return null;
      if (lineageCache.genome === selected) return lineageCache.value;
      const edges = [];
      const walk = (start, next, towardsParents) => {
        const seen = new Set([start]);
        const queue = [start];
        while (queue.length) {
          const genome = queue.shift();
          for (const other of next.get(genome) || []) {
            edges.push(towardsParents ? [other, genome] : [genome, other]);
            if (!seen.has(other)) {
              seen.add(other);
              queue.push(other);
            }
          }
        }
      };
      walk(selected, model.parentsOf, true);
      walk(selected, model.childrenOf, false);
      lineageCache = { genome: selected, value: edges };
      return edges;
    }

    /**
     * Where point `i` is drawn, as [left, top]: genome number runs down the
     * chart and the fitness value across it. Canvas pixels when `canvas` is
     * set, otherwise CSS pixels within the plotting area.
     */
    function position(u, i, canvas = false) {
      return [u.valToPos(model.ys[i], "y", canvas), u.valToPos(model.xs[i], "x", canvas)];
    }

    function emphasis() {
      return CHART_EMPHASIS[config.mode] || CHART_EMPHASIS.progress;
    }

    /**
     * Draws every parent->child link, faded by how crowded the plot is (see
     * CHART_EMPHASIS): the links' canvas positions are computed first, so their
     * visible length decides the opacity they are all stroked with.
     */
    function drawLinks(u) {
      if (!model.edgesByFamily) return;
      const style = emphasis();
      const ctx = u.ctx;
      const { left, top, width, height } = u.bbox;
      const lineWidth = style.linkWidth * uPlot.pxRatio;
      const clampX = (x) => Math.max(left, Math.min(left + width, x));
      const clampY = (y) => Math.max(top, Math.min(top + height, y));

      // segment end points per family, as [x0, y0, x1, y1, ...], and the length they draw inside the plot
      const segmentsByFamily = new Map();
      let visibleLength = 0;
      for (const [family, pairs] of model.edgesByFamily) {
        const segments = new Float64Array(pairs.length * 2);
        for (let k = 0; k < pairs.length; k += 2) {
          const [x0, y0] = position(u, pairs[k], true);
          const [x1, y1] = position(u, pairs[k + 1], true);
          segments.set([x0, y0, x1, y1], k * 2);
          visibleLength += Math.hypot(clampX(x1) - clampX(x0), clampY(y1) - clampY(y0));
        }
        segmentsByFamily.set(family, segments);
      }

      const [minimumAlpha, maximumAlpha, coverage] = style.linkAlpha;
      const timesCovered = (visibleLength * lineWidth) / Math.max(width * height, 1);
      let alpha = Math.max(minimumAlpha, Math.min(maximumAlpha, coverage / Math.max(timesCovered, 1e-9)));
      if (lineage()) alpha *= 0.5;

      ctx.save();
      ctx.beginPath();
      ctx.rect(left, top, width, height);
      ctx.clip();
      ctx.lineWidth = lineWidth;
      for (const [family, segments] of segmentsByFamily) {
        ctx.strokeStyle = withAlpha(familyColor(family), alpha);
        ctx.beginPath();
        for (let k = 0; k < segments.length; k += 4) {
          ctx.moveTo(segments[k], segments[k + 1]);
          ctx.lineTo(segments[k + 2], segments[k + 3]);
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawOverlay(u) {
      const style = emphasis();
      const ctx = u.ctx;
      const { left, top, width, height } = u.bbox;
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, top, width, height);
      ctx.clip();

      const edges = lineage();
      if (edges) {
        ctx.strokeStyle = token(style.lineageInk);
        ctx.lineWidth = style.lineageWidth * uPlot.pxRatio;
        ctx.beginPath();
        for (const [parent, child] of edges) {
          const p = model.indexOf.get(parent);
          const c = model.indexOf.get(child);
          if (p === undefined || c === undefined || model.ys[p] === null || model.ys[c] === null) continue;
          ctx.moveTo(...position(u, p, true));
          ctx.lineTo(...position(u, c, true));
        }
        ctx.stroke();
      }

      const i = model.indexOf.get(selected);
      if (i !== undefined && model.ys[i] !== null) {
        ctx.strokeStyle = token("--text-primary");
        ctx.lineWidth = 2 * uPlot.pxRatio;
        ctx.beginPath();
        ctx.arc(...position(u, i, true), 8 * uPlot.pxRatio, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.restore();
    }

    /** Finds the visible point nearest a cursor position, within the hit radius. */
    function nearest(u, left, top) {
      if (left === null || left === undefined || left < 0 || top < 0) return -1;
      // genome number runs down the chart, so the cursor's height picks the genomes to check
      const edgeA = u.posToVal(top - HIT_RADIUS, "x");
      const edgeB = u.posToVal(top + HIT_RADIUS, "x");
      const xLow = Math.min(edgeA, edgeB);
      const xHigh = Math.max(edgeA, edgeB);
      let low = 0;
      let high = model.count;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (model.xs[middle] < xLow) low = middle + 1;
        else high = middle;
      }
      let best = -1;
      let bestDistance = HIT_RADIUS * HIT_RADIUS;
      for (let i = low; i < model.count && model.xs[i] <= xHigh; i++) {
        if (!model.visible[i]) continue;
        const [pointLeft, pointTop] = position(u, i);
        const dx = pointLeft - left;
        const dy = pointTop - top;
        const distance = dx * dx + dy * dy;
        if (distance <= bestDistance) {
          bestDistance = distance;
          best = i;
        }
      }
      return best;
    }

    function updateTooltip(u) {
      const i = nearest(u, u.cursor.left, u.cursor.top);
      if (i < 0) {
        tooltip.hidden = true;
        return;
      }
      const categoryInfo = model.categories[model.category[i]];
      const parents = model.parentsOf ? model.parentsOf.get(model.xs[i]) || [] : null;
      setChildren(tooltip,
        h("div", {}, h("b", { text: `Genome ${model.xs[i]}` })),
        h("div", {}, h("span", { class: "muted", text: `${config.yKey}: ` }), formatNumber(model.ys[i])),
        h("div", {}, legendItem(categoryInfo.color, categoryInfo.label)),
        h("div", { class: "muted", text: (points.generated_by[i] || []).map(label).join(", ") || "no operators recorded" }),
        parents && parents.length
          ? h("div", { class: "muted", text: `parents: ${parents.map((parent) => (config.seeds && config.seeds.has(parent) ? `${parent} (seed)` : parent)).join(", ")}` })
          : null
      );
      tooltip.hidden = false;
      const overBox = u.over.getBoundingClientRect();
      const hostBox = host.getBoundingClientRect();
      const [pointLeft, pointTop] = position(u, i);
      const x = overBox.left - hostBox.left + pointLeft;
      const y = overBox.top - hostBox.top + pointTop;
      const flip = x + tooltip.offsetWidth + 20 > hostBox.width;
      tooltip.style.left = `${flip ? x - tooltip.offsetWidth - 14 : x + 14}px`;
      tooltip.style.top = `${Math.max(0, y - tooltip.offsetHeight / 2)}px`;
    }

    function renderLegend() {
      const counts = new Array(model.categories.length).fill(0);
      for (let i = 0; i < model.count; i++) if (model.visible[i]) counts[model.category[i]]++;
      const items = model.categories
        .map((category, i) => (counts[i] ? legendItem(category.color, `${category.label} (${counts[i].toLocaleString()})`) : null))
        .filter(Boolean);
      const style = emphasis();
      items.push(legendItem(token(style.bestInk), `best ${config.yKey} so far`, true));
      if (model.edgesByFamily && config.highlightLineage) items.push(legendItem(token(style.lineageInk), "selected lineage", true));
      if (model.edgesByFamily && config.encoding !== "family") {
        items.push(h("span", { class: "muted", text: "links:" }));
        for (const family of Object.keys(FAMILY_SLOTS)) {
          if (model.edgesByFamily.has(family)) items.push(legendItem(familyColor(family), family, true));
        }
      }
      setChildren(legend, ...items);
    }

    function render() {
      const previous = plot && zoomed ? { x: { ...plot.scales.x }, y: { ...plot.scales.y } } : null;
      if (plot) {
        plot.destroy();
        plot = null;
      }
      model = buildModel();
      lineageCache = { genome: null, value: null };
      renderLegend();

      empty.hidden = model.visibleCount > 0;
      host.hidden = model.visibleCount === 0;
      if (!model.visibleCount) {
        empty.textContent = `No genomes have a numeric ${config.yKey} to plot with the current filters.`;
        return;
      }

      const markerSize = model.visibleCount > 5000 ? 5 : 8;
      const surface = token("--surface-1");
      const style = emphasis();
      const options = {
        width: Math.max(280, host.clientWidth),
        height: genomeChartHeight(host),
        // Genome number runs down the chart (earliest at the top). The value runs
        // across with better values to the right: loss-like keys decrease left to
        // right, everything else (e.g. target_metric) increases.
        scales: { x: { time: false, ori: 1, dir: -1 }, y: { ori: 0, dir: lowerIsBetter(config.yKey) ? -1 : 1 } },
        axes: [
          { ...axisStyle("Genome number"), side: 3, size: 60 },
          // wide tick spacing, so the value labels don't collide in the narrow column
          { ...axisStyle(config.yKey), side: 0, size: 50, space: 70 },
        ],
        series: [
          { label: "Genome" },
          ...model.categories.map((category) => ({
            label: category.label,
            stroke: category.color,
            width: 0,
            paths: () => null,
            points: { show: true, size: markerSize, width: 1, stroke: surface, fill: category.color },
          })),
          // straight lines between the genomes that improved the best value
          { label: "Best so far", stroke: token(style.bestInk), width: style.bestWidth, points: { show: false }, spanGaps: true },
        ],
        legend: { show: false },
        cursor: { drag: { x: true, y: true }, points: { show: false } },
        hooks: {
          drawAxes: [drawLinks],
          draw: [drawOverlay],
          setCursor: [updateTooltip],
          setSelect: [() => (zoomed = true)],
        },
      };

      plot = new uPlot(options, model.data, host);
      host.append(tooltip);

      if (previous) {
        plot.setScale("x", { min: previous.x.min, max: previous.x.max });
        plot.setScale("y", { min: previous.y.min, max: previous.y.max });
      }

      let pressed = null;
      plot.over.addEventListener("mousedown", (event) => (pressed = [event.clientX, event.clientY]));
      // A click on empty space clears the selection, but only once it is clear the
      // click is not the start of a double-click (which resets the zoom instead).
      let backgroundClick = null;
      plot.over.addEventListener("click", (event) => {
        if (pressed && Math.hypot(event.clientX - pressed[0], event.clientY - pressed[1]) > 4) return;
        const i = nearest(plot, plot.cursor.left, plot.cursor.top);
        clearTimeout(backgroundClick);
        if (i >= 0) config.onSelect(model.xs[i]);
        else backgroundClick = setTimeout(() => config.onClearSelection(), 300);
      });
      plot.over.addEventListener("dblclick", () => {
        clearTimeout(backgroundClick);
        zoomed = false;
      });
      plot.over.addEventListener("mouseleave", () => (tooltip.hidden = true));
    }

    const resize = () => {
      if (!plot || !host.clientWidth) return;
      const height = genomeChartHeight(host);
      if (Math.abs(plot.width - host.clientWidth) > 2 || Math.abs(plot.height - height) > 2) plot.setSize({ width: host.clientWidth, height });
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    window.addEventListener("resize", resize);

    /** Applies option changes, resetting the zoom unless they only restyle the chart. */
    function applyOptions(changes) {
      if (Object.keys(changes).some((key) => !STYLE_ONLY_OPTIONS.has(key))) zoomed = false;
      Object.assign(config, changes);
    }

    return {
      setData(newPoints, newLinks, changes = {}) {
        applyOptions(changes);
        points = newPoints;
        links = newLinks;
        render();
      },
      setOptions(changes) {
        applyOptions(changes);
        if (points) render();
      },
      setSelected(genome) {
        const changedVisibility = config.hideDeadEnds && links;
        selected = genome;
        lineageCache = { genome: null, value: null };
        if (changedVisibility && points) render();
        else if (plot) plot.redraw(false, false);
      },
      destroy() {
        resizeObserver.disconnect();
        window.removeEventListener("resize", resize);
        if (plot) plot.destroy();
      },
    };
  }

  /** A line chart of several series over a shared x axis, with an optional band per series. */
  function createLineChart(container, { xLabel, yLabel, x, series, height = 300 }) {
    const legend = h(
      "div",
      { class: "chart-legend" },
      series.map((line) => legendItem(line.color, line.label, true))
    );
    const host = h("div", { class: "chart-host" });
    setChildren(container, legend, host);

    const data = [x];
    const uplotSeries = [{ label: xLabel }];
    const bands = [];
    for (const line of series) {
      data.push(line.values);
      uplotSeries.push({ label: line.label, stroke: line.color, width: 2, points: { show: false }, spanGaps: true });
      if (line.low && line.high) {
        const highIndex = data.push(line.high) - 1;
        uplotSeries.push({ label: `${line.label} high`, stroke: "transparent", width: 0, points: { show: false }, spanGaps: true });
        const lowIndex = data.push(line.low) - 1;
        uplotSeries.push({ label: `${line.label} low`, stroke: "transparent", width: 0, points: { show: false }, spanGaps: true });
        bands.push({ series: [highIndex, lowIndex], fill: withAlpha(line.color, 0.15) });
      }
    }

    const plot = new uPlot(
      {
        width: Math.max(320, host.clientWidth),
        height,
        scales: { x: { time: false } },
        axes: [axisStyle(xLabel), { ...axisStyle(yLabel), size: 70 }],
        series: uplotSeries,
        bands,
        legend: { show: false },
        cursor: { drag: { x: true, y: false } },
      },
      data,
      host
    );

    const resizeObserver = new ResizeObserver(() => {
      if (host.clientWidth && Math.abs(plot.width - host.clientWidth) > 2) plot.setSize({ width: host.clientWidth, height });
    });
    resizeObserver.observe(host);

    return {
      destroy() {
        resizeObserver.disconnect();
        plot.destroy();
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Runs page
  // ---------------------------------------------------------------------------

  async function showRunsPage() {
    setBreadcrumbs([{ label: "Runs" }]);
    const content = h("div", {}, notice("Loading runs…"));
    setChildren(app, content);

    let timer = null;
    let destroyed = false;

    async function refresh() {
      const payload = await api("/api/runs");
      if (destroyed) return;
      const runs = payload.runs;
      const header = h(
        "div",
        { class: "page-header" },
        h("h1", { text: "Runs" }),
        h("span", { class: "meta", text: `${runs.length} run${runs.length === 1 ? "" : "s"}` }),
        h("span", { class: "spacer" }),
        runs.length > 1 || payload.groups.length ? h("a", { href: "#/groups", text: "Compare runs →" }) : null
      );

      const rows = runs.map((run) =>
        h(
          "tr",
          { class: "clickable", tabindex: "0", onclick: () => (location.hash = `#/run/${run.index}`), onkeydown: (event) => event.key === "Enter" && (location.hash = `#/run/${run.index}`) },
          h("td", {}, h("span", { class: isLive(run) ? "live-dot" : "live-dot idle", title: isLive(run) ? "Still being written" : "Idle" }), h("a", { href: `#/run/${run.index}`, text: run.name })),
          h("td", { text: run.groups.join(", ") || "—" }),
          h("td", { text: run.error ? "unreadable" : `${label(run.task)} · ${run.task_target ?? "—"}` }),
          h("td", { text: run.population_strategy ?? "—" }),
          h("td", { class: "number", text: formatNumber(run.genomes) }),
          h("td", { class: "number" }, run.best_loss ? [formatNumber(run.best_loss.value), h("span", { class: "meta", text: ` #${run.best_loss.genome_number}` })] : "—"),
          h("td", { class: "number" }, run.best_target_metric ? [formatNumber(run.best_target_metric.value), h("span", { class: "meta", text: ` #${run.best_target_metric.genome_number}` })] : "—"),
          h("td", { text: formatAgo(run.last_saved_at), title: formatTime(run.last_saved_at) })
        )
      );

      setChildren(content,
        header,
        h(
          "div",
          { class: "card table-wrap" },
          h(
            "table",
            {},
            h(
              "thead",
              {},
              h(
                "tr",
                {},
                ["Run", "Groups", "Task", "Strategy", "Genomes", "Best loss", "Best target_metric", "Updated"].map((name, i) =>
                  h("th", { class: i >= 4 && i <= 6 ? "number" : null, text: name })
                )
              )
            ),
            h("tbody", {}, rows)
          )
        )
      );
    }

    await refresh();
    timer = setInterval(() => refresh().catch(() => {}), POLL_INTERVAL_MS);

    return {
      kind: "runs",
      destroy() {
        destroyed = true;
        clearInterval(timer);
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Run page
  // ---------------------------------------------------------------------------

  async function showRunPage(index, initialGenome) {
    const run = await api(`/api/runs/${index}`);
    setBreadcrumbs([{ label: "Runs", href: "#/" }, { label: run.name }]);

    const keys = run.fitness_keys || [];
    const state = {
      mode: "progress",
      yKey: keys.includes("loss") ? "loss" : keys[0] || "n_gates",
      encoding: "insert_type",
      hideDeadEnds: false,
      highlightLineage: true,
      showHistory: false,
      sort: keys.includes("loss") ? "loss" : "genome_number",
      desc: !keys.includes("loss"),
      filters: { insert_type: "", generated_by: "", crossover_type: "", island: "" },
      selected: null,
      tab: "diagram",
      knownGenomes: run.genomes || 0,
      newGenomes: 0,
      points: null,
      links: null,
      summary: run,
      // parents that aren't in the archive: the seed genome
      seeds: new Set(run.unarchived_parents || []),
    };

    let destroyed = false;
    let timer = null;
    let historyChart = null;
    let detailRequest = 0;

    const headerNode = h("div", { class: "page-header" });
    const filterRow = h("div", { class: "toolbar" });
    const chartControls = h("div", { class: "toolbar" });
    const chartNode = h("div");
    const historyNode = h("div");
    historyNode.hidden = true;
    const tableToolbar = h("div", { class: "toolbar" });
    const tableNode = h("div", { class: "table-wrap" });
    // below the table: how many genomes are shown; reaching it loads more
    const tableStatus = h("div", { class: "table-status" });
    // shown above the genome table only while a genome is selected
    const detailNode = h("section", { class: "card detail-panel", "aria-label": "Genome details" });
    detailNode.hidden = true;

    // The two columns always share one height; as the table grows the chart panel
    // stretches with it, and the chart stays in view inside it. The divider
    // between them trades width between the columns.
    const chartPanel = h("section", { class: "card chart-panel", "aria-label": "Genome chart" }, h("div", { class: "chart-sticky" }, chartControls, chartNode, historyNode));
    const runBody = h("div", { class: "run-body" }, chartPanel);
    const columnResizer = createColumnResizer(runBody, chartPanel);
    runBody.append(columnResizer.node, h("div", { class: "run-main" }, detailNode, h("section", { class: "card table-card", "aria-label": "Genome table" }, tableToolbar, tableNode, tableStatus)));
    setChildren(app, headerNode, h("div", { class: "card" }, filterRow), runBody);

    const chart = createGenomeChart(chartNode, {
      mode: state.mode,
      seeds: state.seeds,
      yKey: state.yKey,
      encoding: state.encoding,
      filters: state.filters,
      hideDeadEnds: state.hideDeadEnds,
      highlightLineage: state.highlightLineage,
      onSelect: (genome) => toggleGenome(genome),
      onClearSelection: () => {
        if (state.selected !== null && state.selected !== undefined) location.hash = `#/run/${index}`;
      },
    });

    function renderHeader() {
      const summary = state.summary;
      setChildren(headerNode,
        h("h1", {}, h("span", { class: isLive(summary) ? "live-dot" : "live-dot idle", title: isLive(summary) ? "Still being written" : "Idle" }), run.name),
        h(
          "div",
          { class: "meta" },
          h("span", {}, "task ", h("b", { text: `${label(summary.task)} · ${summary.task_target ?? "—"}` })),
          h("span", {}, "backend ", h("b", { text: summary.target ?? "—" })),
          h("span", {}, "strategy ", h("b", { text: summary.population_strategy ?? "—" })),
          h("span", {}, h("b", { text: formatNumber(summary.genomes) }), " genomes"),
          h("span", { title: formatTime(summary.last_saved_at) }, "updated ", h("b", { text: formatAgo(summary.last_saved_at) })),
          h("span", { text: `started ${formatTime(summary.start_time)}` })
        ),
        summary.command_line ? h("details", {}, h("summary", { text: "Command line" }), h("div", { class: "command" }, h("pre", { text: summary.command_line }), copyButton(summary.command_line))) : null
      );
    }

    function renderFilters() {
      const options = run.filter_options || {};
      const all = (values, labeler = label) => [["", "All"], ...(values || []).map((value) => [String(value), labeler(value)])];
      const setFilter = (name) => (value) => {
        state.filters[name] = value;
        chart.setOptions({ filters: state.filters });
        resetTable();
      };
      setChildren(filterRow,
        h("strong", { text: "Filter genomes" }),
        h("label", {}, "insert type", select(all(options.insert_type), state.filters.insert_type, setFilter("insert_type"), "Insert type filter")),
        h("label", {}, "operator", select(all(options.generated_by), state.filters.generated_by, setFilter("generated_by"), "Operator filter")),
        options.crossover_type && options.crossover_type.length
          ? h("label", {}, "crossover", select(all(options.crossover_type), state.filters.crossover_type, setFilter("crossover_type"), "Crossover type filter"))
          : null,
        options.island && options.island.length
          ? h("label", {}, "island", select(all(options.island, (island) => `island ${island}`), state.filters.island, setFilter("island"), "Island filter"))
          : null,
        h("span", { class: "spacer" }),
        h(
          "form",
          {
            onsubmit: (event) => {
              event.preventDefault();
              const value = Number(new FormData(event.target).get("genome"));
              if (Number.isInteger(value)) location.hash = genomeHref(index, value);
            },
          },
          h("label", {}, "go to genome", h("input", { name: "genome", type: "number", min: "0", style: "width:7em", "aria-label": "Genome number" }))
        )
      );
    }

    function renderChartControls() {
      const yOptions = [...keys, "n_gates", "n_enabled_gates", "n_parameters"].map((key) => [key, key]);
      setChildren(chartControls,
        segmented(
          [
            ["progress", "Progress"],
            ["genealogy", "Genealogy"],
          ],
          state.mode,
          (mode) => {
            state.mode = mode;
            renderChartControls();
            chart.setOptions({ mode });
          }
        ),
        h(
          "label",
          {},
          "value",
          select(
            yOptions,
            state.yKey,
            (key) => {
              state.yKey = key;
              loadChart({ yKey: key });
            },
            "Value plotted across the chart"
          )
        ),
        h(
          "label",
          {},
          "color by",
          select(
            [
              ["insert_type", "insert type"],
              ["family", "operator family"],
              ...(run.filter_options && run.filter_options.island && run.filter_options.island.length ? [["island", "island"]] : []),
            ],
            state.encoding,
            (encoding) => ((state.encoding = encoding), chart.setOptions({ encoding })),
            "Color by"
          )
        ),
        checkbox("hide dead ends", state.hideDeadEnds, (value) => ((state.hideDeadEnds = value), chart.setOptions({ hideDeadEnds: value }))),
        checkbox("highlight selected lineage", state.highlightLineage, (value) => ((state.highlightLineage = value), chart.setOptions({ highlightLineage: value }))),
        run.has_history ? checkbox("search history", state.showHistory, (value) => ((state.showHistory = value), loadHistory())) : null,
        h("span", { class: "meta", text: "drag to zoom · double-click to reset · click a point to open or close it · click empty space to clear" })
      );
    }

    /**
     * Loads every genome's point and parent links for the chart.
     *
     * @param {object} changes Chart options to apply along with the new data.
     */
    async function loadChart(changes = {}) {
      try {
        const payload = await api(`/api/runs/${index}/genealogy?y=${encodeURIComponent(state.yKey)}`);
        state.points = payload.points;
        state.links = payload.links;
        if (!destroyed) chart.setData(state.points, state.links, changes);
      } catch (error) {
        setChildren(chartNode, notice(`Could not load the chart: ${error.message}`, true));
      }
    }

    async function loadHistory() {
      if (historyChart) {
        historyChart.destroy();
        historyChart = null;
      }
      historyNode.hidden = !state.showHistory;
      if (!state.showHistory) return;
      try {
        const { columns } = await api(`/api/runs/${index}/history`);
        const lines = [
          ["best", "best", 1],
          ["top5_mean", "top-k mean", 2],
          ["pop_mean", "population mean", 3],
        ].filter(([column]) => columns[column]);
        setChildren(historyNode, h("h3", { text: "Search history (population fitness as recorded by the profiler, per insertion)" }));
        const chartHost = h("div");
        historyNode.append(chartHost);
        historyChart = createLineChart(chartHost, {
          xLabel: "Insertion",
          yLabel: "fitness",
          x: columns.step,
          series: lines.map(([column, name, slot]) => ({ label: name, color: slotColor(slot), values: columns[column] })),
          height: 240,
        });
      } catch (error) {
        setChildren(historyNode, notice(`Could not load the search history: ${error.message}`, true));
      }
    }

    // -------------------------- genome table (loads more as it is scrolled) --------------------------

    let tableRequest = 0;
    let tableBody = null;
    let loadedRows = 0;
    let totalRows = null;
    let loadingRows = false;
    // the highest genome number the first page included: later pages keep to it, so genomes
    // a live run saves meanwhile don't shift the rows (they wait for a refresh)
    let tableSnapshot = null;

    const rowObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) loadMoreRows();
      },
      { rootMargin: `0px 0px ${LOAD_MARGIN_PX}px 0px` }
    );
    rowObserver.observe(tableStatus);

    /**
     * Empties the genome table and loads its first rows (on page load, after a
     * sort or filter change, or a refresh).
     *
     * @param {object} options `scrollToTable`: bring the top of the table into
     *   view if the page is scrolled past it, so a re-sorted or re-filtered
     *   table is read from its first row (and doesn't load rows just to refill
     *   the scrolled-down window).
     */
    function resetTable({ scrollToTable = true } = {}) {
      tableRequest++;
      loadedRows = 0;
      totalRows = null;
      loadingRows = false;
      tableSnapshot = null;
      // scroll before emptying the table: once its rows are gone the page is shorter,
      // the browser clamps the scroll position and the table top can't be measured
      const card = tableNode.closest(".table-card");
      if (scrollToTable && card) {
        const top = card.getBoundingClientRect().top - TABLE_SCROLL_OFFSET_PX;
        if (top < 0) window.scrollTo({ top: window.scrollY + top, behavior: "instant" });
      }
      tableBody = h("tbody");
      setChildren(tableNode, h("table", {}, h("thead", {}, tableHeader(tableColumns())), tableBody));
      return loadMoreRows();
    }

    /** Appends the next rows to the genome table, until every matching genome is shown. */
    async function loadMoreRows() {
      if (tableBody === null || loadingRows || (totalRows !== null && loadedRows >= totalRows)) return;
      loadingRows = true;
      const request = tableRequest;
      renderTableStatus();

      const parameters = new URLSearchParams({ sort: state.sort, desc: state.desc ? "1" : "0", offset: loadedRows, limit: PAGE_SIZE });
      for (const [name, value] of Object.entries(state.filters)) if (value !== "") parameters.set(name, value);
      if (tableSnapshot !== null) parameters.set("max_genome", tableSnapshot);

      try {
        const page = await api(`/api/runs/${index}/genomes?${parameters}`);
        if (destroyed || request !== tableRequest) return;
        if (tableSnapshot === null) tableSnapshot = page.max_genome_number ?? null;
        const columns = tableColumns();
        tableBody.append(...page.rows.map((row) => tableRow(row, columns)));
        loadedRows += page.rows.length;
        totalRows = page.total;
        loadingRows = false;
        renderTableStatus();
        // keep loading while the table still doesn't reach (nearly) the bottom of the window
        if (page.rows.length && tableStatus.getBoundingClientRect().top < window.innerHeight + LOAD_MARGIN_PX) loadMoreRows();
      } catch (error) {
        if (request !== tableRequest) return;
        loadingRows = false;
        setChildren(tableStatus, notice(`Could not load genomes: ${error.message}`, true));
      }
    }

    /** Shows below the table whether genomes are loading, how many are shown, or that none match. */
    function renderTableStatus() {
      if (totalRows === null) {
        setChildren(tableStatus, notice("Loading genomes…"));
      } else if (totalRows === 0) {
        setChildren(tableStatus, notice("No genomes match the current filters."));
      } else if (loadedRows >= totalRows) {
        setChildren(tableStatus, h("span", { class: "meta", text: `All ${totalRows.toLocaleString()} genomes shown` }));
      } else {
        const more = loadingRows ? "loading more…" : "scroll for more";
        setChildren(tableStatus, h("span", { class: "meta", text: `Showing ${loadedRows.toLocaleString()} of ${totalRows.toLocaleString()} genomes · ${more}` }));
      }
    }

    /** The genome table's columns: each with a sort key (or null), a label and a cell builder. */
    function tableColumns() {
      const insertColors = new Map(categoriesFor("insert_type", { island: [] }).map((category) => [category.key, category.color]));
      return [
        { key: "genome_number", label: "#", number: true, cell: (row) => h("a", { class: "genome-link", href: genomeHref(index, row.genome_number), text: row.genome_number }) },
        ...keys.map((key) => ({ key, label: key, number: true, cell: (row) => formatNumber(row.fitness ? row.fitness[key] : null) })),
        { key: "insert_type", label: "insert type", cell: (row) => legendItem(insertColors.get(row.insert_type) || token("--text-muted"), label(row.insert_type)) },
        { key: null, label: "generated by", cell: (row) => row.generated_by.map(label).join(", ") || "—" },
        { key: null, label: "parents", cell: (row) => (row.parents.length ? parentLinks(row.parents) : "—") },
        ...(run.filter_options && run.filter_options.island && run.filter_options.island.length ? [{ key: "island", label: "island", number: true, cell: (row) => formatNumber(row.island) }] : []),
        { key: "n_enabled_gates", label: "gates", number: true, cell: (row) => `${row.n_enabled_gates}/${row.n_gates}` },
        { key: "n_parameters", label: "params", number: true, cell: (row) => formatNumber(row.n_parameters) },
      ];
    }

    /** The genome table's header row; clicking a sortable column re-sorts (and reloads) the table. */
    function tableHeader(columns) {
      return h(
        "tr",
        {},
        columns.map((column) => {
          if (!column.key) return h("th", { text: column.label });
          const active = state.sort === column.key;
          return h("th", {
            class: `sortable${column.number ? " number" : ""}`,
            "aria-sort": active ? (state.desc ? "descending" : "ascending") : "none",
            text: `${column.label}${active ? (state.desc ? " ▼" : " ▲") : ""}`,
            onclick: () => {
              state.desc = active ? !state.desc : column.key === "genome_number" || (keys.includes(column.key) && !lowerIsBetter(column.key));
              state.sort = column.key;
              resetTable();
            },
          });
        })
      );
    }

    /** One genome's table row; clicking it selects the genome, or clears the selection if it is selected. */
    function tableRow(row, columns) {
      return h(
        "tr",
        {
          class: `clickable${row.genome_number === state.selected ? " selected" : ""}`,
          tabindex: "0",
          "data-genome": row.genome_number,
          onclick: (event) => {
            // links to other genomes (such as parents) navigate as usual
            if (event.target.closest("a:not(.genome-link)")) return;
            event.preventDefault();
            toggleGenome(row.genome_number);
          },
          // a focused link turns Enter into a click itself
          onkeydown: (event) => event.key === "Enter" && !event.target.closest("a") && toggleGenome(row.genome_number),
        },
        columns.map((column) => h("td", { class: column.number ? "number" : null }, column.cell(row)))
      );
    }

    function renderTableToolbar() {
      setChildren(tableToolbar,
        h("h2", { text: "Genomes" }),
        h("span", { class: "spacer" }),
        state.newGenomes > 0
          ? h(
              "button",
              {
                type: "button",
                class: "badge",
                onclick: () => {
                  state.newGenomes = 0;
                  renderTableToolbar();
                  resetTable();
                },
              },
              `${state.newGenomes.toLocaleString()} new genome${state.newGenomes === 1 ? "" : "s"} · refresh`
            )
          : null
      );
    }

    /** Links to a genome's parents, labelling the (unstored) seed genome rather than linking to it. */
    function parentLinks(parents) {
      return parents.map((parent, i) => [
        i ? ", " : "",
        state.seeds.has(parent)
          ? h("span", { class: "seed", title: SEED_EXPLANATION, text: `${parent} (seed)` })
          : h("a", { href: genomeHref(index, parent), text: parent }),
      ]);
    }

    /** Selects a genome, or clears the selection if it is already the selected one. */
    function toggleGenome(genome) {
      location.hash = genome === state.selected ? `#/run/${index}` : genomeHref(index, genome);
    }

    function highlightTableRow() {
      for (const row of tableNode.querySelectorAll("tr[data-genome]")) {
        row.classList.toggle("selected", Number(row.dataset.genome) === state.selected);
      }
    }

    // -------------------------- genome detail panel --------------------------

    async function selectGenome(genome) {
      state.selected = genome;
      chart.setSelected(genome);
      highlightTableRow();
      if (genome === null || genome === undefined) {
        detailRequest++;
        detailNode.hidden = true;
        setChildren(detailNode);
        return;
      }
      if (state.seeds.has(genome)) {
        // e.g. "go to genome 1": explain the seed rather than report a missing genome
        detailRequest++;
        detailNode.hidden = false;
        setChildren(
          detailNode,
          h(
            "div",
            { class: "page-header" },
            h("h2", { text: `Genome ${genome} (seed)` }),
            h("span", { class: "spacer" }),
            h("button", { type: "button", text: "Close", onclick: () => (location.hash = `#/run/${index}`) })
          ),
          notice(SEED_EXPLANATION)
        );
        return;
      }
      const request = ++detailRequest;
      detailNode.hidden = false;
      detailNode.classList.add("is-loading");
      try {
        const payload = await api(`/api/runs/${index}/genomes/${genome}`);
        if (request !== detailRequest || destroyed) return;
        renderDetail(payload);
        // the panel sits above the genome table, so bring it into view if it is off screen
        const box = detailNode.getBoundingClientRect();
        if (box.top > window.innerHeight - 80 || box.bottom < 60) detailNode.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        if (request === detailRequest) setChildren(detailNode, notice(`Could not load genome ${genome}: ${error.message}`, true));
      } finally {
        detailNode.classList.remove("is-loading");
      }
    }

    function renderDetail(payload) {
      const { summary, genome, children, commands } = payload;
      const number = genome.genome_number;
      const tabContent = h("div");

      const tabs = [
        ["diagram", "Diagram"],
        ["training", "Training"],
        ["fitness", "Fitness"],
        ["gates", `Gates (${summary.n_enabled_gates}/${summary.n_gates})`],
        ["lineage", "Lineage"],
        ["commands", "Commands"],
      ];

      function showTab(tab) {
        state.tab = tab;
        for (const button of tabBar.querySelectorAll("button")) button.setAttribute("aria-selected", String(button.dataset.tab === tab));
        setChildren(tabContent, ...renderTab(tab));
      }

      function renderTab(tab) {
        if (tab === "diagram" || tab === "training") {
          const frame = h("div", { class: "image-frame" }, notice(tab === "diagram" ? "Drawing the architecture diagram…" : "Drawing the training plot…"));
          const image = h("img", { alt: `Genome ${number} ${tab === "diagram" ? "architecture diagram" : "training plot"}` });
          image.addEventListener("load", () => setChildren(frame, image));
          image.addEventListener("error", async () => {
            let message = "The image could not be drawn.";
            try {
              await api(`/api/runs/${index}/genomes/${number}/${tab}.png`);
            } catch (error) {
              message = error.message;
            }
            setChildren(frame, notice(message));
          });
          image.src = `/api/runs/${index}/genomes/${number}/${tab}.png`;
          return [frame];
        }
        if (tab === "fitness") return renderFitness(summary, genome);
        if (tab === "gates") return renderGates(genome);
        if (tab === "lineage") return renderLineage(summary, genome, children);
        return renderCommands(number, commands);
      }

      const tabBar = h(
        "div",
        { class: "tabs", role: "tablist" },
        tabs.map(([tab, name]) => h("button", { type: "button", role: "tab", "data-tab": tab, text: name, onclick: () => showTab(tab) }))
      );

      const compareForm = h(
        "form",
        {
          class: "toolbar",
          onsubmit: (event) => {
            event.preventDefault();
            const other = Number(new FormData(event.target).get("other"));
            if (Number.isInteger(other)) location.hash = `#/run/${index}/compare/${number}/${other}`;
          },
        },
        h("label", {}, "compare with genome", h("input", { name: "other", type: "number", min: "0", style: "width:7em", required: true, value: summary.parents.find((parent) => !state.seeds.has(parent)) ?? "" })),
        h("button", { type: "submit", text: "Compare" })
      );

      const insertColor = new Map(categoriesFor("insert_type", { island: [] }).map((category) => [category.key, category.color])).get(summary.insert_type) || token("--text-muted");

      setChildren(detailNode,
        h(
          "div",
          { class: "page-header" },
          h("h2", { text: `Genome ${number}` }),
          legendItem(insertColor, label(summary.insert_type)),
          h("span", { class: "spacer" }),
          h("button", { type: "button", text: "Close", onclick: () => (location.hash = `#/run/${index}`) })
        ),
        h(
          "div",
          { class: "meta" },
          (run.fitness_keys || []).slice(0, 4).map((key) => h("span", {}, `${key} `, h("b", { text: formatNumber(genome.fitness ? genome.fitness[key] : null) })))
        ),
        compareForm,
        tabBar,
        tabContent
      );
      showTab(state.tab);
    }

    function factsList(entries) {
      return h(
        "dl",
        { class: "facts" },
        entries.flatMap(([name, value]) => [h("dt", { text: name }), h("dd", {}, value)])
      );
    }

    function renderFitness(summary, genome) {
      const valueEntries = (values) => Object.entries(values || {}).map(([key, value]) => [key, formatNumber(value)]);
      return [
        h("h3", { text: "Fitness" }),
        factsList(valueEntries(genome.fitness)),
        h("h3", { text: "Search" }),
        factsList([
          ["task", `${label(genome.task)} · ${genome.task_target ?? "—"}`],
          ["backend", genome.target],
          ["insertion", formatNumber(summary.insertion)],
          ["saved", formatTime(summary.saved_at)],
          ["generated by", summary.generated_by.map(label).join(", ") || "—"],
          ["crossover type", summary.crossover_type ?? "—"],
          ["island", formatNumber(summary.island)],
          ["gates", `${summary.n_enabled_gates} enabled of ${summary.n_gates}`],
          ["gate parameters", formatNumber(summary.n_parameters)],
          ["input qubits", formatQubits(genome.input_qubits)],
          ["output qubits", formatQubits(genome.output_qubits)],
          ["encoder", genome.encoder ? genome.encoder.class : "none"],
          ["decoder", genome.decoder ? genome.decoder.class : "none"],
        ]),
        h("h3", { text: "Hyperparameters" }),
        factsList(valueEntries(genome.hyperparameters)),
      ];
    }

    function renderGates(genome) {
      const gates = [...(genome.gates || [])].sort((a, b) => a.depth - b.depth);
      if (!gates.length) return [notice("This genome has no gates.")];
      return [
        h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["innovation", "depth", "gate", "qubits", "parameters", "enabled"].map((name, i) => h("th", { class: i < 2 ? "number" : null, text: name })))),
            h(
              "tbody",
              {},
              gates.map((gate) =>
                h(
                  "tr",
                  {},
                  h("td", { class: "number", text: gate.innovation_number }),
                  h("td", { class: "number", text: formatNumber(gate.depth) }),
                  h("td", { class: "mono", text: gate.method_name }),
                  h("td", { class: "mono", text: formatQubits(gate.qubits) }),
                  h("td", { class: "mono", text: formatParameters(gate.parameters) }),
                  h("td", { text: gate.enabled === false ? "no" : "yes" })
                )
              )
            )
          )
        ),
      ];
    }

    function renderLineage(summary, genome, children) {
      const links = (numbers) => numbers.map((other, i) => [i ? ", " : "", h("a", { href: genomeHref(index, other), text: other })]);
      const depthSelect = select(
        [1, 2, 3, 5, 8, 12, 20].map((depth) => [String(depth), `${depth} generation${depth === 1 ? "" : "s"}`]),
        "5",
        (depth) => loadAncestry(Number(depth)),
        "Ancestry depth"
      );
      const graphNode = h("div", { class: "ancestry" }, notice("Loading ancestry…"));

      async function loadAncestry(depth) {
        try {
          const payload = await api(`/api/runs/${index}/genomes/${genome.genome_number}/ancestry?depth=${depth}`);
          setChildren(graphNode, renderAncestry(payload));
        } catch (error) {
          setChildren(graphNode, notice(`Could not load the ancestry: ${error.message}`, true));
        }
      }
      loadAncestry(5);

      return [
        factsList([
          ["generated by", summary.generated_by.map(label).join(", ") || "—"],
          ["crossover type", summary.crossover_type ?? "—"],
          ["parents", summary.parents.length ? parentLinks(summary.parents) : "—"],
          [
            `children (${children.length.toLocaleString()})`,
            children.length ? [links(children.slice(0, MAX_CHILDREN_SHOWN)), children.length > MAX_CHILDREN_SHOWN ? ` … and ${(children.length - MAX_CHILDREN_SHOWN).toLocaleString()} more` : ""] : "none",
          ],
        ]),
        h("div", { class: "toolbar", style: "margin-top:12px" }, h("h3", { text: "Ancestry", style: "margin:0" }), depthSelect, h("span", { class: "meta", text: `darker = better ${state.yKey}; click a genome to open it` })),
        graphNode,
      ];
    }

    /** Draws a genome's ancestors as columns by generation, oldest on the left. */
    function renderAncestry(payload) {
      const byGeneration = new Map();
      for (const node of payload.nodes) {
        if (!byGeneration.has(node.generation)) byGeneration.set(node.generation, []);
        byGeneration.get(node.generation).push(node);
      }
      const maxGeneration = Math.max(0, ...byGeneration.keys());
      const columnWidth = 96;
      const rowHeight = 30;
      const position = new Map();
      let tallest = 1;
      for (const [generation, nodes] of byGeneration) {
        nodes.sort((a, b) => a.genome_number - b.genome_number);
        const shown = nodes.slice(0, MAX_ANCESTORS_PER_GENERATION);
        tallest = Math.max(tallest, shown.length + (nodes.length > shown.length ? 1 : 0));
        shown.forEach((node, i) => position.set(node.genome_number, { x: 40 + (maxGeneration - generation) * columnWidth, y: 24 + i * rowHeight, node }));
      }

      const values = payload.nodes.map((node) => (node.fitness ? node.fitness[state.yKey] : null)).filter(isNumber);
      const low = Math.min(...values);
      const high = Math.max(...values);
      const light = token("--sequential-light");
      const dark = token("--sequential-dark");
      const colorOf = (node) => {
        const value = node.fitness ? node.fitness[state.yKey] : null;
        if (!isNumber(value)) return token("--surface-2");
        const t = high > low ? (value - low) / (high - low) : 1;
        return mixColors(light, dark, lowerIsBetter(state.yKey) ? 1 - t : t);
      };

      const width = 80 + maxGeneration * columnWidth + 40;
      const height = 24 + tallest * rowHeight + 10;
      const graph = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `Ancestry of genome ${payload.nodes.find((node) => node.generation === 0)?.genome_number}` });

      for (const edge of payload.edges) {
        const from = position.get(edge.parent);
        const to = position.get(edge.child);
        if (!from || !to) continue;
        graph.append(s("line", { x1: from.x, y1: from.y, x2: to.x, y2: to.y, stroke: token("--baseline"), "stroke-width": 1 }));
      }

      for (const [generation, nodes] of byGeneration) {
        if (nodes.length > MAX_ANCESTORS_PER_GENERATION) {
          graph.append(s("text", { x: 40 + (maxGeneration - generation) * columnWidth - 14, y: 24 + MAX_ANCESTORS_PER_GENERATION * rowHeight + 4, text: `+${nodes.length - MAX_ANCESTORS_PER_GENERATION} more` }));
        }
      }

      for (const { x, y, node } of position.values()) {
        const value = node.fitness ? node.fitness[state.yKey] : null;
        const title = node.in_archive
          ? `Genome ${node.genome_number} · generation ${node.generation} back · ${state.yKey}: ${formatNumber(value)} · ${label(node.insert_type)} · ${(node.generated_by || []).map(label).join(", ")}`
          : `Genome ${node.genome_number}. ${SEED_EXPLANATION}`;
        const group = s(
          "g",
          { class: node.in_archive ? "node" : null, tabindex: node.in_archive ? "0" : null },
          s("title", { text: title }),
          s("circle", {
            cx: x,
            cy: y,
            r: 8,
            fill: node.in_archive ? colorOf(node) : "none",
            stroke: node.generation === 0 ? token("--text-primary") : node.in_archive ? token("--surface-1") : token("--text-muted"),
            "stroke-width": 2,
          }),
          s("text", { x: x + 11, y: y + 4, text: node.in_archive ? node.genome_number : "seed" })
        );
        if (node.in_archive) {
          group.addEventListener("click", () => (location.hash = genomeHref(index, node.genome_number)));
          group.addEventListener("keydown", (event) => event.key === "Enter" && (location.hash = genomeHref(index, node.genome_number)));
        }
        graph.append(group);
      }
      return graph;
    }

    function renderCommands(number, commands) {
      return [
        ...Object.entries(commands).map(([name, command]) => [h("h3", { text: name }), h("div", { class: "command" }, h("pre", { text: command }), copyButton(command))]).flat(),
        h("h3", { text: "Genome JSON" }),
        h("a", { href: `/api/runs/${index}/genomes/${number}.json`, download: `genome_${number}.json`, text: `Download genome_${number}.json` }),
      ];
    }

    // -------------------------- live updates --------------------------

    async function poll() {
      const summary = await api(`/api/runs/${index}`);
      if (destroyed) return;
      state.summary = summary;
      renderHeader();
      const added = (summary.genomes || 0) - state.knownGenomes;
      if (added > 0) {
        state.knownGenomes = summary.genomes;
        state.newGenomes += added;
        renderTableToolbar();
        await loadChart();
        if (state.showHistory) loadHistory();
      }
    }

    renderHeader();
    renderFilters();
    renderChartControls();
    renderTableToolbar();
    await Promise.all([loadChart(), resetTable({ scrollToTable: false })]);
    if (initialGenome !== null && initialGenome !== undefined) selectGenome(initialGenome);
    timer = setInterval(() => poll().catch(() => {}), POLL_INTERVAL_MS);

    return {
      kind: "run",
      index,
      select: selectGenome,
      destroy() {
        destroyed = true;
        clearInterval(timer);
        rowObserver.disconnect();
        columnResizer.destroy();
        chart.destroy();
        if (historyChart) historyChart.destroy();
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Compare two genomes
  // ---------------------------------------------------------------------------

  async function showComparePage(index, a, b) {
    const [run, payload] = await Promise.all([api(`/api/runs/${index}`), api(`/api/runs/${index}/compare?a=${a}&b=${b}`)]);
    setBreadcrumbs([{ label: "Runs", href: "#/" }, { label: run.name, href: `#/run/${index}` }, { label: `Compare ${a} and ${b}` }]);

    const numeric = (value) => (isNumber(value) ? value : null);
    const valueTable = (rows, showDifference) =>
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          {},
          h("thead", {}, h("tr", {}, h("th", { text: "key" }), h("th", { class: "number", text: `genome ${a}` }), h("th", { class: "number", text: `genome ${b}` }), showDifference ? h("th", { class: "number", text: `${b} − ${a}` }) : null)),
          h(
            "tbody",
            {},
            rows.map((row) =>
              h(
                "tr",
                { class: row.same ? null : "different" },
                h("td", { text: row.key }),
                h("td", { class: "number", text: formatNumber(row.a) }),
                h("td", { class: "number", text: formatNumber(row.b) }),
                showDifference ? h("td", { class: "number", text: numeric(row.a) !== null && numeric(row.b) !== null ? formatNumber(row.b - row.a) : "—" }) : null
              )
            )
          )
        )
      );

    const gateRows = (gates) =>
      gates.map((gate) =>
        h("tr", {}, h("td", { class: "number", text: gate.innovation_number }), h("td", { class: "mono", text: gate.method_name }), h("td", { class: "mono", text: formatQubits(gate.qubits) }), h("td", { class: "mono", text: formatParameters(gate.parameters) }), h("td", { text: gate.enabled === false ? "no" : "yes" }))
      );
    const gateTable = (title, gates) =>
      gates.length
        ? [
            h("h3", { text: `${title} (${gates.length})` }),
            h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["innovation", "gate", "qubits", "parameters", "enabled"].map((name, i) => h("th", { class: i === 0 ? "number" : null, text: name })))), h("tbody", {}, gateRows(gates)))),
          ]
        : [];

    const form = h(
      "form",
      {
        class: "toolbar",
        onsubmit: (event) => {
          event.preventDefault();
          const data = new FormData(event.target);
          location.hash = `#/run/${index}/compare/${Number(data.get("a"))}/${Number(data.get("b"))}`;
        },
      },
      h("h1", { text: "Compare genomes" }),
      h("input", { name: "a", type: "number", min: "0", value: a, style: "width:7em", "aria-label": "First genome" }),
      h("span", { text: "and" }),
      h("input", { name: "b", type: "number", min: "0", value: b, style: "width:7em", "aria-label": "Second genome" }),
      h("button", { type: "submit", text: "Compare" })
    );

    const column = (side, number) =>
      h(
        "section",
        { class: "card" },
        h("div", { class: "page-header" }, h("h2", {}, h("a", { href: genomeHref(index, number), text: `Genome ${number}` })), h("span", { class: "meta", text: `${label(side.summary.insert_type)} · ${side.summary.generated_by.map(label).join(", ") || "no operators"} · ${side.summary.n_enabled_gates}/${side.summary.n_gates} gates` })),
        h("div", { class: "image-frame" }, h("img", { src: `/api/runs/${index}/genomes/${number}/diagram.png`, alt: `Genome ${number} architecture diagram` }))
      );

    const gates = payload.gates;
    setChildren(app,
      form,
      h("div", { class: "compare-grid" }, column(payload.a, a), column(payload.b, b)),
      h("div", { class: "compare-grid" }, h("section", { class: "card" }, h("h2", { text: "Fitness" }), valueTable(payload.fitness, true)), h("section", { class: "card" }, h("h2", { text: "Hyperparameters" }), valueTable(payload.hyperparameters, false))),
      h(
        "section",
        { class: "card" },
        h("h2", { text: "Gates, matched by innovation number" }),
        h("p", { class: "meta", text: `${gates.unchanged} identical · ${gates.changed.length} changed · ${gates.only_a.length} only in ${a} · ${gates.only_b.length} only in ${b}` }),
        gates.changed.length
          ? [
              h("h3", { text: `Changed (${gates.changed.length})` }),
              h(
                "div",
                { class: "table-wrap" },
                h(
                  "table",
                  {},
                  h("thead", {}, h("tr", {}, h("th", { class: "number", text: "innovation" }), h("th", { text: "gate" }), h("th", { text: "field" }), h("th", { text: `genome ${a}` }), h("th", { text: `genome ${b}` }))),
                  h(
                    "tbody",
                    {},
                    gates.changed.flatMap((change) =>
                      Object.entries(change.differences).map(([field, [valueA, valueB]]) =>
                        h(
                          "tr",
                          {},
                          h("td", { class: "number", text: change.innovation_number }),
                          h("td", { class: "mono", text: change.method_name }),
                          h("td", { text: field }),
                          h("td", { class: "mono", text: field === "qubits" ? formatQubits(valueA) : field === "parameters" ? formatParameters(valueA) : formatNumber(valueA) }),
                          h("td", { class: "mono", text: field === "qubits" ? formatQubits(valueB) : field === "parameters" ? formatParameters(valueB) : formatNumber(valueB) })
                        )
                      )
                    )
                  )
                )
              ),
            ]
          : null,
        gateTable(`Only in genome ${a}`, gates.only_a),
        gateTable(`Only in genome ${b}`, gates.only_b)
      )
    );

    return { kind: "compare", destroy() {} };
  }

  // ---------------------------------------------------------------------------
  // Compare runs (groups)
  // ---------------------------------------------------------------------------

  async function showGroupsPage() {
    setBreadcrumbs([{ label: "Runs", href: "#/" }, { label: "Compare runs" }]);
    const settings = { metric: "best", conf: "std" };
    const content = h("div", {}, notice("Aggregating runs…"));
    setChildren(app, content);
    let chart = null;

    async function load() {
      const payload = await api(`/api/groups?metric=${encodeURIComponent(settings.metric)}&conf=${settings.conf}`);
      if (chart) {
        chart.destroy();
        chart = null;
      }

      const charted = payload.groups.filter((group) => group.history).slice(0, CATEGORICAL_SLOTS);
      const steps = [...new Set(charted.flatMap((group) => group.history.step))].sort((x, y) => x - y);
      const align = (group, field) => {
        const byStep = new Map(group.history.step.map((step, i) => [step, group.history[field][i]]));
        return steps.map((step) => (byStep.has(step) ? byStep.get(step) : null));
      };

      const stats = (values) => (values && values.n ? `${formatNumber(values.mean)} ± ${formatNumber(values.std)} (min ${formatNumber(values.min)}, max ${formatNumber(values.max)})` : "—");
      const insertTypes = [...new Set(payload.groups.flatMap((group) => Object.values(group.operators).flatMap((counts) => Object.keys(counts))))].sort();

      const controls = h(
        "div",
        { class: "toolbar" },
        h("h1", { text: "Compare runs" }),
        h("label", {}, "history metric", select((payload.metrics.length ? payload.metrics : [settings.metric]).map((metric) => [metric, metric]), settings.metric, (metric) => ((settings.metric = metric), load()), "History metric")),
        h(
          "label",
          {},
          "band",
          select(
            [
              ["std", "± 1 standard deviation"],
              ["95ci", "95% confidence interval"],
            ],
            settings.conf,
            (conf) => ((settings.conf = conf), load()),
            "Band"
          )
        )
      );

      const chartNode = h("div");
      const skipped = payload.groups.filter((group) => group.history).length - charted.length;

      setChildren(content,
        controls,
        h(
          "section",
          { class: "card" },
          h("h2", { text: `Mean ${settings.metric} per insertion across each group's runs` }),
          skipped > 0 ? notice(`The chart shows the first ${CATEGORICAL_SLOTS} groups; every group is in the table below.`) : null,
          charted.length ? chartNode : notice("None of these runs recorded a search history.")
        ),
        h(
          "section",
          { class: "card table-wrap" },
          h("h2", { text: "Best genome of each run, summarized per group" }),
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["group", "runs", "best loss (mean ± std)", "best target_metric (mean ± std)", "history"].map((name) => h("th", { text: name })))),
            h(
              "tbody",
              {},
              payload.groups.map((group) =>
                h(
                  "tr",
                  {},
                  h("td", {}, charted.includes(group) ? legendItem(slotColor(charted.indexOf(group) + 1), group.name, true) : group.name),
                  h("td", {}, group.runs.map((run, i) => [i ? ", " : "", h("a", { href: `#/run/${run.index}`, text: run.name })])),
                  h("td", { text: stats(group.best_loss) }),
                  h("td", { text: stats(group.best_target_metric) }),
                  h("td", { text: group.history ? `${group.history.n_runs} run(s)` : group.history_error || "—" })
                )
              )
            )
          )
        ),
        h(
          "section",
          { class: "card" },
          h("h2", { text: "How each operator's genomes were inserted" }),
          payload.groups.map((group) => {
            const operators = Object.entries(group.operators).sort(([x], [y]) => x.localeCompare(y));
            return h(
              "details",
              { open: payload.groups.length <= 3 },
              h("summary", { text: group.name }),
              operators.length
                ? h(
                    "div",
                    { class: "table-wrap" },
                    h(
                      "table",
                      {},
                      h("thead", {}, h("tr", {}, h("th", { text: "operator" }), h("th", { class: "number", text: "total" }), insertTypes.map((type) => h("th", { class: "number", text: label(type) })))),
                      h(
                        "tbody",
                        {},
                        operators.map(([operator, counts]) => {
                          const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
                          return h(
                            "tr",
                            {},
                            h("td", { text: label(operator) }),
                            h("td", { class: "number", text: total.toLocaleString() }),
                            insertTypes.map((type) => h("td", { class: "number", text: counts[type] ? `${counts[type].toLocaleString()} (${((100 * counts[type]) / total).toFixed(1)}%)` : "—" }))
                          );
                        })
                      )
                    )
                  )
                : notice("No genomes recorded.")
            );
          })
        )
      );

      if (charted.length) {
        chart = createLineChart(chartNode, {
          xLabel: "Insertion",
          yLabel: settings.metric,
          x: steps,
          series: charted.map((group, i) => ({ label: `${group.name} (n=${group.history.n_runs})`, color: slotColor(i + 1), values: align(group, "mean"), low: align(group, "low"), high: align(group, "high") })),
          height: 340,
        });
      }
    }

    await load();
    return {
      kind: "groups",
      destroy() {
        if (chart) chart.destroy();
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Routing
  // ---------------------------------------------------------------------------

  function parseRoute() {
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
    const number = (value) => (value !== undefined && /^\d+$/.test(value) ? Number(value) : null);
    if (parts[0] === "groups") return { page: "groups" };
    if (parts[0] === "run" && number(parts[1]) !== null) {
      const index = number(parts[1]);
      if (parts[2] === "compare" && number(parts[3]) !== null && number(parts[4]) !== null) return { page: "compare", index, a: number(parts[3]), b: number(parts[4]) };
      return { page: "run", index, genome: parts[2] === "genome" ? number(parts[3]) : null };
    }
    return { page: "runs" };
  }

  async function route() {
    const target = parseRoute();
    if (currentPage && currentPage.kind === "run" && target.page === "run" && currentPage.index === target.index) {
      currentPage.select(target.genome);
      return;
    }
    if (currentPage) currentPage.destroy();
    currentPage = null;
    try {
      if (target.page === "groups") currentPage = await showGroupsPage();
      else if (target.page === "run") currentPage = await showRunPage(target.index, target.genome);
      else if (target.page === "compare") currentPage = await showComparePage(target.index, target.a, target.b);
      else currentPage = await showRunsPage();
    } catch (error) {
      setChildren(app, notice(error.message, true), h("p", {}, h("a", { href: "#/", text: "Back to the runs" })));
    }
  }

  window.addEventListener("hashchange", route);
  route();
})();
