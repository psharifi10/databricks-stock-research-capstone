"use strict";

const form = document.querySelector("#research-form");
const tickerInput = document.querySelector("#ticker");
const questionInput = document.querySelector("#question");
const researchButton = document.querySelector("#research-button");
const statusMessage = document.querySelector("#status");
const errorMessage = document.querySelector("#error");
const results = document.querySelector("#results");
const agentAnswer = document.querySelector("#agent-answer");
const companyContent = document.querySelector("#company-content");
const latestPrice = document.querySelector("#latest-price");
const priceRows = document.querySelector("#price-rows");
const priceEmpty = document.querySelector("#price-empty");
const newsList = document.querySelector("#news-list");
const evidenceList = document.querySelector("#evidence-list");
const AGENT_REQUEST_TIMEOUT_MS = 65000;
const RESEARCH_SERVICE_UNAVAILABLE_MESSAGE =
  "The research service is temporarily unavailable.";

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined && text !== null) {
    element.textContent = String(text);
  }
  return element;
}

function displayValue(value, fallback = "Not available") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return displayValue(value);
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2,
  }).format(number);
}

function formatMarketCap(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return displayValue(value);
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
    style: "currency",
    currency: "USD",
  }).format(number);
}

function formatDate(value) {
  if (!value) {
    return "Date unavailable";
  }
  const source = String(value);
  const date = /^\d{4}-\d{2}-\d{2}$/.test(source)
    ? new Date(`${source}T00:00:00`)
    : new Date(source);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(date);
}

function sourceLink(url, label = "View source") {
  if (!url) {
    return null;
  }
  try {
    const parsed = new URL(url, window.location.origin);
    if (!(["http:", "https:"].includes(parsed.protocol))) {
      return null;
    }
    const link = createElement("a", "source-link", label);
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch (_error) {
    return null;
  }
}

function fact(label, value) {
  const item = createElement("div", "fact");
  const term = createElement("dt", "", label);
  const definition = createElement("dd", "", displayValue(value));
  item.append(term, definition);
  return item;
}

function renderCompany(company, ticker) {
  companyContent.replaceChildren();
  if (!company) {
    companyContent.append(
      createElement("p", "empty-state", "No persisted company profile is available."),
    );
    return;
  }

  const title = createElement("div", "company-title");
  title.append(
    createElement("strong", "", displayValue(company.name, "Company name unavailable")),
    createElement("span", "ticker-chip", displayValue(company.ticker, ticker)),
  );

  const facts = createElement("dl", "fact-grid");
  facts.append(
    fact("Industry", company.industry),
    fact("Exchange", company.exchange),
    fact("Market cap", formatMarketCap(company.market_cap)),
  );

  const description = createElement(
    "p",
    "description",
    displayValue(company.description, "No company description is available."),
  );
  companyContent.append(title, facts, description);
}

function renderPrices(prices) {
  priceRows.replaceChildren();
  latestPrice.replaceChildren();
  const rows = Array.isArray(prices) ? prices : [];
  priceEmpty.hidden = rows.length > 0;

  if (rows.length === 0) {
    return;
  }

  const latest = rows[rows.length - 1];
  latestPrice.append(
    createElement("strong", "", `$${formatNumber(latest.close)}`),
    createElement("span", "", `Latest close · ${formatDate(latest.price_date)}`),
  );

  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    [
      formatDate(row.price_date),
      formatNumber(row.open),
      formatNumber(row.high),
      formatNumber(row.low),
      formatNumber(row.close),
    ].forEach((value) => {
      tableRow.append(createElement("td", "", value));
    });
    priceRows.append(tableRow);
  });
}

function renderNews(news) {
  newsList.replaceChildren();
  const rows = Array.isArray(news) ? news : [];
  if (rows.length === 0) {
    newsList.append(createElement("p", "empty-state", "No recent linked news is available."));
    return;
  }

  rows.forEach((article) => {
    const item = createElement("article", "item");
    item.append(createElement("h3", "", displayValue(article.title, "Untitled article")));

    const meta = createElement("div", "item-meta");
    meta.append(
      createElement("span", "", displayValue(article.publisher, "Publisher unavailable")),
      createElement("span", "", formatDate(article.published_at)),
    );
    if (article.sentiment) {
      meta.append(createElement("span", "sentiment", article.sentiment));
    }
    item.append(meta);

    const link = sourceLink(article.article_url);
    if (link) {
      item.append(link);
    }
    newsList.append(item);
  });
}

function renderEvidence(evidence) {
  evidenceList.replaceChildren();
  const rows = Array.isArray(evidence) ? evidence : [];
  if (rows.length === 0) {
    evidenceList.append(
      createElement("p", "empty-state", "No semantic evidence matched this question."),
    );
    return;
  }

  rows.forEach((result) => {
    const item = createElement("article", "item");
    item.append(createElement("h3", "", displayValue(result.title, "Untitled source")));

    const similarity = Number(result.similarity);
    const similarityLabel = Number.isFinite(similarity)
      ? `${Math.round(similarity * 100)}% similarity`
      : "Similarity unavailable";
    const meta = createElement("div", "item-meta");
    meta.append(createElement("span", "", similarityLabel));
    item.append(meta);

    const chunk = displayValue(result.chunk_text, "No passage preview is available.");
    const preview = chunk.length > 280 ? `${chunk.slice(0, 277)}…` : chunk;
    item.append(createElement("p", "item-preview", preview));

    const link = sourceLink(result.article_url);
    if (link) {
      item.append(link);
    }
    evidenceList.append(item);
  });
}

function renderResearch(context) {
  renderCompany(context.company, context.ticker);
  renderPrices(context.prices);
  renderNews(context.recent_news);
  renderEvidence(context.semantic_evidence);
  results.hidden = false;
}

function renderAgentSummary(answer, unavailable = false) {
  agentAnswer.textContent = answer;
  agentAnswer.classList.toggle("agent-answer--unavailable", unavailable);
}

async function requestResearch(path, body, signal) {
  const requestOptions = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  if (signal) {
    requestOptions.signal = signal;
  }
  const response = await fetch(path, requestOptions);
  const contentType = response.headers.get("Content-Type") || "";
  if (!contentType.toLowerCase().includes("application/json")) {
    throw new Error(RESEARCH_SERVICE_UNAVAILABLE_MESSAGE);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(RESEARCH_SERVICE_UNAVAILABLE_MESSAGE);
  }
  if (!response.ok || !payload?.ok) {
    throw new Error(
      payload?.error?.message || "Research could not be completed right now.",
    );
  }
  return payload.data;
}

async function requestAgentResearch(body) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    AGENT_REQUEST_TIMEOUT_MS,
  );
  try {
    return await requestResearch("/api/agent", body, controller.signal);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function setLoading(isLoading) {
  researchButton.disabled = isLoading;
  researchButton.setAttribute("aria-busy", String(isLoading));
  statusMessage.textContent = isLoading
    ? "Requesting an AI summary and grounded research evidence…"
    : "";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  results.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  results.hidden = true;

  const ticker = tickerInput.value.trim().toUpperCase();
  const question = questionInput.value.trim();
  tickerInput.value = ticker;
  if (!ticker || !question) {
    showError("Enter both a ticker and a research question.");
    return;
  }

  setLoading(true);
  try {
    const requestBody = { ticker, question };
    const [agentOutcome, researchOutcome] = await Promise.allSettled([
      requestAgentResearch(requestBody),
      requestResearch("/api/research", requestBody),
    ]);

    if (agentOutcome.status === "fulfilled") {
      renderAgentSummary(agentOutcome.value.answer);
    } else {
      renderAgentSummary(
        "AI summary is temporarily unavailable. Grounded research data is shown below.",
        true,
      );
    }

    if (researchOutcome.status === "rejected") {
      throw researchOutcome.reason;
    }
    renderResearch(researchOutcome.value);
  } catch (error) {
    showError(
      error instanceof Error
        ? error.message
        : "Research could not be completed right now.",
    );
  } finally {
    setLoading(false);
  }
});
