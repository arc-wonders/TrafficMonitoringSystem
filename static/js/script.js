document.addEventListener("DOMContentLoaded", function () {
    // Store chart instances to prevent duplicate canvas errors
    const charts = {};
  
    // Function to create or update a chart
    function createChart(chartId, type, data, options) {
      const canvas = document.getElementById(chartId);
  
      if (charts[chartId]) {
        charts[chartId].destroy(); // Destroy previous chart instance
      }
  
      const ctx = canvas.getContext("2d");
      charts[chartId] = new Chart(ctx, { type, data, options });
    }
  
    // Sample static data (to be replaced by live data)
    const sampleLineData = {
      labels: ["0h", "1h", "2h", "3h", "4h", "5h", "6h"],
      datasets: [{
        label: "Sample Data",
        data: [12, 19, 3, 5, 2, 3, 10],
        borderColor: "#39ff14",
        backgroundColor: "rgba(57, 255, 20, 0.2)",
        tension: 0.4
      }]
    };
  
    const sampleBarData = {
      labels: ["Cars", "Bikes", "Trucks", "Buses"],
      datasets: [{
        label: "Vehicle Types",
        data: [50, 20, 15, 10],
        backgroundColor: ["#ff073a", "#39ff14", "#0f52ba", "#ffb347"],
      }]
    };
  
    // Function to load charts dynamically when a section is viewed
    function loadChart(targetId) {
      if (targetId === "section-vehicle-count") {
        createChart("vehicleCountChart", "line", sampleLineData, { responsive: true });
      } else if (targetId === "section-vehicle-category") {
        createChart("vehicleCategoryChart", "pie", {
          labels: sampleBarData.labels,
          datasets: [{
            data: sampleBarData.datasets[0].data,
            backgroundColor: sampleBarData.datasets[0].backgroundColor,
          }]
        }, { responsive: true });
      } else if (targetId === "section-traffic-density") {
        createChart("trafficDensityChart", "bar", sampleLineData, { responsive: true });
      } else if (targetId === "section-high-speed") {
        createChart("highSpeedChart", "bar", sampleBarData, { responsive: true });
      } else if (targetId === "section-speed-distribution") {
        createChart("speedDistributionChart", "bar", sampleBarData, { responsive: true });
      }
    }
  
    // Sidebar Navigation Handling
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');
  
    navItems.forEach(item => {
      item.addEventListener('click', () => {
        // Remove active class from all nav items
        navItems.forEach(i => i.classList.remove('active'));
  
        // Hide all content sections
        contentSections.forEach(sec => sec.style.display = 'none');
  
        // Add active class to clicked item and show target section
        item.classList.add('active');
        const targetId = item.getAttribute('data-target');
        document.getElementById(targetId).style.display = 'block';
  
        // Load corresponding chart
        loadChart(targetId);
      });
    });
  
    // Load default chart when the page loads
    loadChart("section-vehicle-count");
  
    // Fetch Real-Time Data Using SSE (Server-Sent Events)
    const eventSource = new EventSource("/api/detect");
  
    eventSource.onmessage = function (event) {
      const data = JSON.parse(event.data);
  
      // Update charts with real-time data
      if (charts["vehicleCountChart"]) {
        charts["vehicleCountChart"].data.labels.push(new Date().toLocaleTimeString());
        charts["vehicleCountChart"].data.datasets[0].data.push(data.vehicle_count);
        charts["vehicleCountChart"].update();
      }
  
      if (charts["vehicleCategoryChart"]) {
        charts["vehicleCategoryChart"].data.datasets[0].data = [
          data.vehicle_types.car || 0,
          data.vehicle_types.bike || 0,
          data.vehicle_types.truck || 0,
          data.vehicle_types.bus || 0
        ];
        charts["vehicleCategoryChart"].update();
      }
  
      if (charts["highSpeedChart"]) {
        charts["highSpeedChart"].data.datasets[0].data = [data.high_speed_violations || 0];
        charts["highSpeedChart"].update();
      }
  
      if (charts["speedDistributionChart"]) {
        charts["speedDistributionChart"].data.datasets[0].data = data.speed_data.slice(-10);
        charts["speedDistributionChart"].update();
      }
    };
  
    eventSource.onerror = function (error) {
      console.error("Error with SSE:", error);
    };
  
    // Load Data Button Functionality
    document.getElementById('loadDataBtn').addEventListener('click', function () {
      alert('Live data is being fetched automatically.');
    });
  });
  