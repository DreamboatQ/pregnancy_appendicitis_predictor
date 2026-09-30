const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const resultCard = document.querySelector("#prediction-result");
const formError = document.querySelector("#form-error");
const warningsBox = document.querySelector("#result-warnings");

const fieldNames = [
  "age", "gestational_age", "body_temperature", "pain_duration", "wbc", "crp",
  "neutrophils", "diabetes", "hypertension", "placental_abnormalities", "fever",
  "rlq_effusion", "primiparity", "appendiceal_swelling", "abscess",
];

function setLoading(isLoading) {
  button.disabled = isLoading;
  button.querySelector(".button-label").hidden = isLoading;
  button.querySelector(".button-loading").hidden = !isLoading;
}

function readForm() {
  const data = Object.fromEntries(new FormData(form).entries());
  return Object.fromEntries(
    fieldNames.map((name) => [name, data[name] === "" ? null : Number(data[name])]),
  );
}

function populateForm(values) {
  for (const name of fieldNames) {
    if (!(name in values)) continue;
    form.elements[name].value = values[name] ?? "";
  }
}

function renderResult(result) {
  document.querySelector("#probability-value").textContent = `${Number(result.probability_percent).toFixed(2)}%`;
  document.querySelector("#classification-label").textContent = result.classification;
  document.querySelector("#imputed-count").textContent = `${result.missing_fields.length} of 15`;
  document.querySelector("#shap-plot").src = result.shap_plot;

  const banner = document.querySelector("#probability-banner");
  banner.classList.toggle("probability-banner--above", result.probability >= result.threshold);
  banner.classList.toggle("probability-banner--below", result.probability < result.threshold);

  const messages = [];
  if (result.missing_fields.length) {
    messages.push(`Imputed unavailable inputs: ${result.missing_fields.join(", ")}.`);
  }
  messages.push(...result.warnings);
  warningsBox.replaceChildren();
  if (messages.length) {
    const list = document.createElement("ul");
    for (const message of messages) {
      const item = document.createElement("li");
      item.textContent = message;
      list.append(item);
    }
    warningsBox.append(list);
    warningsBox.hidden = false;
  } else {
    warningsBox.hidden = true;
  }

  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runPrediction(values, { updateForm = false } = {}) {
  if (updateForm) populateForm(values);
  formError.hidden = true;
  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");
    renderResult(result);
    return result;
  } catch (error) {
    formError.textContent = error instanceof Error ? error.message : "Prediction failed.";
    formError.hidden = false;
    resultCard.hidden = true;
    throw error;
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await runPrediction(readForm());
  } catch {
    // The visible error region already contains the actionable message.
  }
});

function registerWebMCPTool() {
  const context = document.modelContext;
  if (!context?.registerTool) return;

  const nullableNumber = { type: ["number", "null"] };
  const nullableBinary = { type: ["number", "null"], enum: [0, 1, null] };
  const properties = {
    age: nullableNumber,
    gestational_age: nullableNumber,
    body_temperature: nullableNumber,
    pain_duration: nullableNumber,
    wbc: nullableNumber,
    crp: nullableNumber,
    neutrophils: nullableNumber,
    diabetes: nullableBinary,
    hypertension: nullableBinary,
    placental_abnormalities: nullableBinary,
    fever: nullableBinary,
    rlq_effusion: nullableBinary,
    primiparity: nullableBinary,
    appendiceal_swelling: nullableBinary,
    abscess: nullableBinary,
  };

  try {
    void Promise.resolve(context.registerTool({
      name: "predict_observed_surgery_probability",
      title: "Predict observed surgical-decision probability",
      description: "Fill the 15 patient variables, run the saved Random Forest ensemble, and display its pooled probability and individual SHAP explanation.",
      inputSchema: {
        type: "object",
        properties,
        required: fieldNames,
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      async execute(input) {
        const result = await runPrediction(input, { updateForm: true });
        return {
          probability: result.probability,
          probabilityPercent: result.probability_percent,
          classification: result.classification,
          threshold: result.threshold,
          imputedFields: result.missing_fields,
          topContributors: result.contributions.slice(0, 5),
        };
      },
    })).catch(() => {});
  } catch {
    // WebMCP is optional and does not affect the visible application.
  }
}

registerWebMCPTool();
