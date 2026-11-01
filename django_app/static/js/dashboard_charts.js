// Client-side Chart.js visualization loader for analytics dashboard
document.addEventListener("DOMContentLoaded", function () {
  const chartCanvas = document.getElementById("statusPieChart");
  if (!chartCanvas) return;

  fetch("/analytics/api/charts/")
    .then((response) => response.json())
    .then((data) => {
      // 1. Loan Approval Status Doughnut Chart
      const ctxPie = document.getElementById("statusPieChart").getContext("2d");
      new Chart(ctxPie, {
        type: "doughnut",
        data: {
          labels: data.status_distribution.labels,
          datasets: [
            {
              data: data.status_distribution.data,
              backgroundColor: ["#10b981", "#ef4444"],
              borderWidth: 2,
              borderColor: "#ffffff",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });

      // 2. Property Area Distribution Bar Chart
      const propCanvas = document.getElementById("propertyAreaChart");
      if (propCanvas) {
        new Chart(propCanvas.getContext("2d"), {
          type: "bar",
          data: {
            labels: data.property_area.labels,
            datasets: [
              {
                label: "Applicants Count",
                data: data.property_area.data,
                backgroundColor: ["#3b82f6", "#60a5fa", "#93c5fd"],
                borderRadius: 6,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { beginAtZero: true },
            },
          },
        });
      }

      // 3. Credit History Impact Chart
      const creditCanvas = document.getElementById("creditHistoryChart");
      if (creditCanvas) {
        new Chart(creditCanvas.getContext("2d"), {
          type: "pie",
          data: {
            labels: data.credit_history.labels,
            datasets: [
              {
                data: data.credit_history.data,
                backgroundColor: ["#2563eb", "#cbd5e1"],
                borderWidth: 2,
                borderColor: "#ffffff",
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom" },
            },
          },
        });
      }

      // 4. Income Brackets Chart
      const incomeCanvas = document.getElementById("incomeBracketsChart");
      if (incomeCanvas) {
        new Chart(incomeCanvas.getContext("2d"), {
          type: "bar",
          data: {
            labels: data.income_brackets.labels,
            datasets: [
              {
                label: "Applicants",
                data: data.income_brackets.data,
                backgroundColor: "#8b5cf6",
                borderRadius: 6,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { beginAtZero: true },
            },
          },
        });
      }
    })
    .catch((err) => console.error("Error loading analytics data:", err));
});
