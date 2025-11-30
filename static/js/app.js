// static/js/app.js

function safeParseJSON(raw) {
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.error("Failed to parse JSON:", e, raw);
    return null;
  }
}

/**
 * Results charts for a single run
 */
function initResultsCharts(layerMetrics, canvasIdMetrics, canvasIdEce) {
  if (!layerMetrics || !layerMetrics.length) return;

  const metricsId = canvasIdMetrics || "metricsChart";
  const eceId = canvasIdEce || "eceChart";

  const metricsCanvas = document.getElementById(metricsId);
  const eceCanvas = document.getElementById(eceId);
  if (!metricsCanvas || !eceCanvas) return;

  const labels = layerMetrics.map(m =>
    typeof m.layer_index !== "undefined" ? m.layer_index : m.layer
  );
  const acc = layerMetrics.map(m => m.accuracy);
  const f1 = layerMetrics.map(m => m.f1_weighted);
  const ece = layerMetrics.map(m => m.ece);

  if (metricsCanvas._chartInstance) metricsCanvas._chartInstance.destroy();
  if (eceCanvas._chartInstance) eceCanvas._chartInstance.destroy();

  metricsCanvas._chartInstance = new Chart(metricsCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: labels.map(l => `Layer ${l}`),
      datasets: [
        { label: "Accuracy", data: acc, tension: 0.3, borderWidth: 2, pointRadius: 3 },
        { label: "Weighted F1", data: f1, tension: 0.3, borderWidth: 2, pointRadius: 3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          min: 0.0,
          max: 1.0,
          ticks: { callback: v => v.toFixed(2) }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`
          }
        },
        legend: { labels: { color: "#e5e7eb" } }
      }
    }
  });

  eceCanvas._chartInstance = new Chart(eceCanvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: labels.map(l => `Layer ${l}`),
      datasets: [{ label: "ECE", data: ece, borderWidth: 1 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.0,
          ticks: { callback: v => v.toFixed(2) }
        }
      },
      plugins: {
        tooltip: {
          callbacks: { label: ctx => `ECE: ${ctx.parsed.y.toFixed(3)}` }
        },
        legend: { labels: { color: "#e5e7eb" } }
      }
    }
  });
}

/**
 * Radar chart for information flow
 */
function initRadarChart(radarData) {
  if (!radarData || !radarData.labels || !radarData.labels.length) return;
  const canvas = document.getElementById("radarChart");
  if (!canvas) return;

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas.getContext("2d"), {
    type: "radar",
    data: {
      labels: radarData.labels,
      datasets: [
        {
          label: "Information flow signature",
          data: radarData.values,
          borderWidth: 2,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: { display: false }
        }
      },
      plugins: {
        legend: { labels: { color: "#e5e7eb" } }
      }
    }
  });
}

/**
 * Comparison chart: overlay F1 curves from two models
 */
function initCompareChart(metricsA, metricsB, labelA, labelB) {
  if (!metricsA || !metricsB) return;
  const canvas = document.getElementById("compareChart");
  if (!canvas) return;

  const labels = metricsA.map(m =>
    typeof m.layer_index !== "undefined" ? m.layer_index : m.layer
  );
  const f1A = metricsA.map(m => m.f1_weighted);
  const f1B = metricsB.map(m => m.f1_weighted);

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels: labels.map(l => `Layer ${l}`),
      datasets: [
        {
          label: labelA,
          data: f1A,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3
        },
        {
          label: labelB,
          data: f1B,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { min: 0, max: 1, ticks: { callback: v => v.toFixed(2) } }
      },
      plugins: {
        legend: { labels: { color: "#e5e7eb" } }
      }
    }
  });
}

/**
 * Histogram for dataset explorer
 */
function initLengthHistogram(bins, counts) {
  const canvas = document.getElementById("lengthChart");
  if (!canvas || !bins || !counts) return;
  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: bins,
      datasets: [{ label: "Examples", data: counts, borderWidth: 1 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#e5e7eb" } }
      }
    }
  });
}

// Loading overlay on run launch
document.addEventListener("DOMContentLoaded", () => {
  const runButton = document.querySelector("[data-run-button]");
  const loadingOverlay = document.querySelector("[data-loading-overlay]");

  if (runButton && loadingOverlay) {
    runButton.addEventListener("click", () => {
      loadingOverlay.classList.remove("hidden");
    });
  }
});

// Expose APIs globally
window.initResultsCharts = initResultsCharts;
window.initRadarChart = initRadarChart;
window.initCompareChart = initCompareChart;
window.initLengthHistogram = initLengthHistogram;
