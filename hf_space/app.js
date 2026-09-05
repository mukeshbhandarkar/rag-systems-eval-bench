const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const num = value => value == null ? "—" : Number(value).toFixed(2);

async function loadData() {
  const [resultsResponse, failuresResponse] = await Promise.all([
    fetch("data/results.json"), fetch("data/failures.json")
  ]);
  if (!resultsResponse.ok || !failuresResponse.ok) throw new Error("Dashboard data files could not be loaded.");
  return [await resultsResponse.json(), await failuresResponse.json()];
}

function renderOverview(benchmark) {
  const values = [
    [benchmark.question_count, "Questions"], [benchmark.chunk_count, "Corpus chunks"],
    [benchmark.benchmark_version, "Benchmark version"], [Object.keys(benchmark.categories).length, "Categories"]
  ];
  document.querySelector("#overview").innerHTML = values.map(([value, label]) =>
    `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`).join("");
}

function renderRetrievers(retrievers) {
  const qualityKeys = ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"];
  const maxima = Object.fromEntries(qualityKeys.map(key => [key, Math.max(...retrievers.map(r => r.metrics[key]))]));
  document.querySelector("#retrieval-table").innerHTML = retrievers.map(retriever => {
    const metricCells = qualityKeys.map(key => `<td class="${retriever.metrics[key] === maxima[key] ? "best" : ""}">${pct(retriever.metrics[key])}</td>`).join("");
    return `<tr><td>${retriever.label}</td>${metricCells}<td>${num(retriever.elapsed_seconds)}s</td><td>${num(retriever.peak_rss_mb)} MB</td></tr>`;
  }).join("");
}

function renderCategories(results) {
  const categories = Object.keys(results.benchmark.categories).filter(category => category !== "unanswerable");
  const filters = document.querySelector("#category-filters");
  filters.innerHTML = categories.map((category, index) => `<button type="button" data-category="${category}" aria-pressed="${index === 0}">${category} · ${results.benchmark.categories[category]}</button>`).join("");
  const draw = category => {
    document.querySelector("#category-chart").innerHTML = results.retrievers.map(retriever => {
      const value = retriever.categories[category].recall_at_5;
      return `<div class="system-bar"><strong>${retriever.label}</strong><div class="track"><div class="fill" style="width:${value * 100}%"></div></div><span>${pct(value)}</span></div>`;
    }).join("");
  };
  filters.addEventListener("click", event => {
    const button = event.target.closest("button"); if (!button) return;
    filters.querySelectorAll("button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    draw(button.dataset.category);
  });
  draw(categories[0]);
}

function renderFailures(failures) {
  const select = document.querySelector("#failure-select");
  const detail = document.querySelector("#failure-detail");
  if (!failures.length) { select.hidden = true; detail.innerHTML = "<p>No incomplete top-5 cases were exported.</p>"; return; }
  select.innerHTML = failures.map((item, index) => `<option value="${index}">${item.question_id} · ${item.question}</option>`).join("");
  const draw = index => {
    const item = failures[index];
    const cards = Object.entries(item.retrievers).map(([name, data]) => `<div class="retriever-card"><strong>${name.toUpperCase()} · Recall@5 ${pct(data.recall_at_5)}</strong><ol>${data.results.map(result => `<li><code>${result.chunk_id}</code><br>${result.excerpt}…</li>`).join("")}</ol></div>`).join("");
    detail.innerHTML = `<p class="eyebrow">${item.category}</p><h3>${item.question}</h3><p><strong>Reference:</strong> ${item.reference_answers.join(" / ")}</p><p><strong>Relevant evidence:</strong> <code>${item.relevant_chunk_ids.join(", ")}</code></p><div class="retriever-grid">${cards}</div>`;
  };
  select.addEventListener("change", () => draw(Number(select.value))); draw(0);
}

function renderGeneration(generation) {
  const metrics = generation.metrics;
  const values = [
    [pct(metrics.exact_match), "Exact match"], [pct(metrics.token_f1), "Token F1"],
    [pct(metrics.semantic_similarity_mean), "Semantic similarity"], [metrics.success_count, "Exact successes"],
    [metrics.retrieval_failure_count, "Retrieval failures"], [metrics.generation_failure_count, "Generation failures"]
  ];
  document.querySelector("#generation-cards").innerHTML = values.map(([value, label]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`).join("");
}

loadData().then(([results, failures]) => {
  renderOverview(results.benchmark); renderRetrievers(results.retrievers); renderCategories(results);
  renderFailures(failures); renderGeneration(results.generation);
}).catch(error => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="error" role="alert">${error.message} Serve this directory through an HTTP server; file:// fetch is intentionally unsupported.</p>`); });
