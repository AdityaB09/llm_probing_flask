// app.js

// app.js

let loadingInterval = null;

function showLoadingOverlay() {
  const overlay = document.getElementById("loading-overlay");
  const msgEl = document.getElementById("loading-message");
  if (!overlay || !msgEl) return;

  overlay.classList.remove("d-none");

  const messages = [
    "1/5 Downloading / loading dataset (Kaggle or NLTK)…",
    "2/5 Building train / test splits…",
    "3/5 Extracting hidden states from every transformer layer…",
    "4/5 Training logistic regression probes layer by layer…",
    "5/5 Computing accuracy, F1, and calibration error…",
    "Almost there – rendering visualizations…",
  ];

  let idx = 0;
  msgEl.textContent = messages[idx];

  if (loadingInterval) clearInterval(loadingInterval);
  loadingInterval = setInterval(() => {
    idx = (idx + 1) % messages.length;
    msgEl.textContent = messages[idx];
  }, 2200);
}

// Stop any interval when the page has finished navigation
window.addEventListener("pageshow", () => {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
});


// History sparkline on index page
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("historySparkline");
  if (canvas) {
    fetch("/api/history_summary")
      .then((res) => res.json())
      .then((data) => {
        const labels = data.map((run) => run.task_name).reverse();
        const f1s = data.map((run) => run.best_f1).reverse();
        new Chart(canvas.getContext("2d"), {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Best F1 per run",
                data: f1s,
                tension: 0.4,
              },
            ],
          },
          options: {
            plugins: {
              legend: { display: false },
            },
            scales: {
              x: { display: false },
              y: { min: 0, max: 1 },
            },
          },
        });
      })
      .catch(() => {});
  }
});

// Called from results.html
function initResultsCharts(layerMetrics, canvasIdOverride) {
  const layers = layerMetrics.map((m) => m.layer_index);
  const acc = layerMetrics.map((m) => m.accuracy);
  const f1 = layerMetrics.map((m) => m.f1_weighted);
  const ece = layerMetrics.map((m) => m.ece);

  const metricsCanvas = document.getElementById(canvasIdOverride || "metricsChart");
  const eceCanvas = document.getElementById("eceChart"); // may be null on detail page

  if (metricsCanvas) {
    new Chart(metricsCanvas.getContext("2d"), {
      type: "line",
      data: {
        labels: layers,
        datasets: [
          { label: "Accuracy", data: acc, tension: 0.3 },
          { label: "F1 (weighted)", data: f1, tension: 0.3 },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#e2e8f0" } },
        },
        scales: {
          x: {
            title: { display: true, text: "Layer Index", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
          },
          y: {
            min: 0,
            max: 1,
            ticks: { color: "#94a3b8" },
          },
        },
      },
    });
  }

  if (eceCanvas) {
    new Chart(eceCanvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: layers,
        datasets: [
          { label: "ECE", data: ece },
        ],
      },
      options: {
        plugins: {
          legend: { labels: { color: "#e2e8f0" } },
        },
        scales: {
          x: {
            title: { display: true, text: "Layer Index", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
          },
          y: {
            min: 0,
            max: Math.max(...ece.concat([0.1])),
            ticks: { color: "#94a3b8" },
          },
        },
      },
    });
  }
}