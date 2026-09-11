/**
 * LeafDoc Interactive Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const systemStatusBadge = document.getElementById("system-status-badge");
  const systemStatusText = document.getElementById("system-status-text");

  // Input Tabs & Panels
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewImage = document.getElementById("preview-image");
  const btnClearFile = document.getElementById("btn-clear-file");
  const btnDiagnoseFile = document.getElementById("btn-diagnose-file");
  const chkAutoCrop = document.getElementById("chk-auto-crop");
  const cropSelector = document.getElementById("crop-selector");

  // Webcam
  const webcamVideo = document.getElementById("webcam-video");
  const webcamCanvas = document.getElementById("webcam-canvas");
  const btnStartCamera = document.getElementById("btn-start-camera");
  const btnSnapPhoto = document.getElementById("btn-snap-photo");
  let mediaStream = null;

  // URL
  const imageUrlInput = document.getElementById("image-url-input");
  const btnDiagnoseUrl = document.getElementById("btn-diagnose-url");

  // Sample Gallery
  const samplePills = document.querySelectorAll(".sample-pill");

  // Status & Progress States
  const loadingSection = document.getElementById("loading-section");
  const loadingProgressText = document.getElementById("loading-progress-text");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const btnDismissError = document.getElementById("btn-dismiss-error");
  const resultsSection = document.getElementById("results-section");

  // Result Elements
  const resOverallStatusBadge = document.getElementById("res-overall-status-badge");
  const resSpeciesBadge = document.getElementById("res-species-badge");
  const resAutocropBadge = document.getElementById("res-autocrop-badge");
  const resCropFilterBadge = document.getElementById("res-crop-filter-badge");
  const resCropFilterName = document.getElementById("res-crop-filter-name");
  const resDiseaseName = document.getElementById("res-disease-name");
  const resPathogenName = document.getElementById("res-pathogen-name");
  const resConfidenceVal = document.getElementById("res-confidence-val");
  const resConfidenceBar = document.getElementById("res-confidence-bar");
  const resSeverityRating = document.getElementById("res-severity-rating");
  const resAffectedPct = document.getElementById("res-affected-pct");
  const resImmediateActionText = document.getElementById("res-immediate-action-text");
  const resUrgentBanner = document.getElementById("res-urgent-banner");
  const resDescription = document.getElementById("res-disease-description");

  // Visualizer
  const visMainImage = document.getElementById("vis-main-image");
  const vtabBtnCrop = document.getElementById("vtab-btn-crop");
  const viewerTabs = document.querySelectorAll(".vtab-btn");
  const meterProgressBar = document.getElementById("meter-progress-bar");
  const meterAffectedPctBadge = document.getElementById("meter-affected-pct-badge");
  const resTopkList = document.getElementById("res-topk-list");

  // Treatment Tabs
  const treatmentTabBtns = document.querySelectorAll(".ttab-btn");
  const treatmentPanels = document.querySelectorAll(".treatment-panel");
  const resCulturalList = document.getElementById("res-cultural-list");
  const resChemicalList = document.getElementById("res-chemical-list");
  const resBiologicalList = document.getElementById("res-biological-list");
  const resPreventiveList = document.getElementById("res-preventive-list");

  // Classes Modal
  const btnOpenClasses = document.getElementById("btn-open-classes");
  const classesModal = document.getElementById("classes-modal");
  const btnCloseClasses = document.getElementById("btn-close-classes");
  const classesGridContainer = document.getElementById("classes-grid-container");
  const classesSearchInput = document.getElementById("classes-search-input");
  let cachedClassesData = null;

  // Print Buttons
  const btnPrintReport = document.getElementById("btn-print-report");
  const btnPrintAction = document.getElementById("btn-print-action");

  // State
  let currentVisualizations = null;
  let selectedFile = null;

  // =========================================================================
  // 1. Initial Health Check
  // =========================================================================
  async function checkSystemHealth() {
    try {
      const resp = await fetch("/api/v1/health");
      if (!resp.ok) throw new Error("Health check failed");
      const data = await resp.json();

      const devName = data.device_name ? data.device_name : data.device.toUpperCase();
      systemStatusText.innerHTML = `Ready &bull; <strong>${devName}</strong> (${data.total_classes} Classes)`;
      systemStatusBadge.title = `PyTorch compute device: ${data.device}, Models verified.`;
    } catch (err) {
      console.warn("Backend not yet responding:", err);
      systemStatusText.innerText = "Engine Initializing...";
    }
  }
  checkSystemHealth();

  // =========================================================================
  // 2. Input Navigation Tabs
  // =========================================================================
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.add("d-none"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.remove("d-none");

      // Stop camera if leaving camera tab
      if (targetId !== "camera-panel" && mediaStream) {
        stopCamera();
      }
    });
  });

  // =========================================================================
  // 3. File Drag & Drop + Browse
  // =========================================================================
  dropzone.addEventListener("click", (e) => {
    if (e.target !== btnClearFile && !btnClearFile.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please select a valid image file (JPEG, PNG, WebP).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
      previewImage.src = ev.target.result;
      dropzonePreview.classList.remove("d-none");
      btnDiagnoseFile.classList.remove("d-none");
      // Scroll smoothly to button
      btnDiagnoseFile.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };
    reader.readAsDataURL(file);
  }

  btnClearFile.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    dropzonePreview.classList.add("d-none");
    btnDiagnoseFile.classList.add("d-none");
  });

  btnDiagnoseFile.addEventListener("click", () => {
    if (selectedFile) {
      runDiagnosisWithFile(selectedFile);
    }
  });

  // =========================================================================
  // 4. Live Webcam Scanner
  // =========================================================================
  btnStartCamera.addEventListener("click", async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      webcamVideo.srcObject = mediaStream;
      btnStartCamera.classList.add("d-none");
      btnSnapPhoto.classList.remove("d-none");
    } catch (err) {
      showError("Unable to access camera. Please allow camera permissions or upload an image file.");
      console.error(err);
    }
  });

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
      webcamVideo.srcObject = null;
      btnStartCamera.classList.remove("d-none");
      btnSnapPhoto.classList.add("d-none");
    }
  }

  btnSnapPhoto.addEventListener("click", () => {
    if (!webcamVideo.videoWidth) return;
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCanvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        stopCamera();
        runDiagnosisWithFile(file);
      }
    }, "image/jpeg", 0.95);
  });

  // =========================================================================
  // 5. Remote Image URL
  // =========================================================================
  btnDiagnoseUrl.addEventListener("click", () => {
    const url = imageUrlInput.value.trim();
    if (!url) {
      showError("Please enter a valid image URL.");
      return;
    }
    runDiagnosisWithUrl(url);
  });

  // =========================================================================
  // 6. Instant 1-Click Samples
  // =========================================================================
  samplePills.forEach((pill) => {
    pill.addEventListener("click", async () => {
      const src = pill.getAttribute("data-src");
      if (!src) return;

      showLoading(`Loading sample leaf: ${pill.getAttribute("data-title")}...`);
      try {
        const response = await fetch(src);
        if (!response.ok) throw new Error("Could not fetch sample leaf file");
        const blob = await response.blob();
        const filename = src.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });

        // Update preview in dropzone
        handleFileSelected(file);
        // Run diagnosis directly
        await runDiagnosisWithFile(file);
      } catch (err) {
        showError(`Sample loading failed: ${err.message}`);
      }
    });
  });

  // =========================================================================
  // 7. Diagnosis Execution & API Request
  // =========================================================================
  async function runDiagnosisWithFile(file) {
    showLoading("Running Dual-Stage Deep Learning & Grad-CAM Analysis...");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("include_visualizations", "true");
    formData.append("top_k", "5");
    formData.append("auto_crop", chkAutoCrop ? chkAutoCrop.checked.toString() : "true");
    if (cropSelector && cropSelector.value) {
      formData.append("target_species", cropSelector.value);
    }

    try {
      const resp = await fetch("/api/v1/diagnose", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: "Server error occurred" }));
        throw new Error(errData.detail || `Server returned code ${resp.status}`);
      }

      const data = await resp.json();
      renderDiagnosisResults(data);
    } catch (err) {
      showError(err.message || "Failed to complete leaf diagnosis.");
    }
  }

  async function runDiagnosisWithUrl(url) {
    showLoading("Fetching remote leaf image and running neural inference...");
    const formData = new FormData();
    formData.append("image_url", url);
    formData.append("include_visualizations", "true");
    formData.append("top_k", "5");
    formData.append("auto_crop", chkAutoCrop ? chkAutoCrop.checked.toString() : "true");
    if (cropSelector && cropSelector.value) {
      formData.append("target_species", cropSelector.value);
    }

    try {
      const resp = await fetch("/api/v1/diagnose", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: "Server error occurred" }));
        throw new Error(errData.detail || `Server returned code ${resp.status}`);
      }

      const data = await resp.json();
      renderDiagnosisResults(data);
    } catch (err) {
      showError(err.message || "Failed to complete leaf diagnosis from URL.");
    }
  }

  // =========================================================================
  // 8. Render Results Dashboard
  // =========================================================================
  function renderDiagnosisResults(data) {
    hideLoading();
    errorSection.classList.add("d-none");
    resultsSection.classList.remove("d-none");
    btnPrintReport.classList.remove("d-none");

    const s1 = data.stage1_binary;
    const s2 = data.stage2_fine;
    const s3 = data.stage3_severity;
    const s4 = data.stage4_treatment;
    currentVisualizations = data.visualizations || {};

    // Auto-crop badge & Isolated leaf tab
    if (data.is_cropped && currentVisualizations.cropped_leaf) {
      if (resAutocropBadge) resAutocropBadge.classList.remove("d-none");
      if (vtabBtnCrop) vtabBtnCrop.classList.remove("d-none");
    } else {
      if (resAutocropBadge) resAutocropBadge.classList.add("d-none");
      if (vtabBtnCrop) vtabBtnCrop.classList.add("d-none");
    }

    // Crop Filter Badge
    if (data.target_species && resCropFilterBadge && resCropFilterName) {
      resCropFilterName.innerText = `Crop Locked: ${data.target_species}`;
      resCropFilterBadge.classList.remove("d-none");
    } else if (resCropFilterBadge) {
      resCropFilterBadge.classList.add("d-none");
    }

    // 1. Health Status Banner
    const isHealthy = data.is_healthy;
    resOverallStatusBadge.innerText = isHealthy ? "Healthy Plant" : "Diseased Leaf";
    resOverallStatusBadge.className = `badge ${isHealthy ? "badge-healthy" : "badge-diseased"}`;

    resSpeciesBadge.innerText = s2.species;
    resDiseaseName.innerText = s2.condition_name;
    resPathogenName.innerText = s4.pathogen_type || "None / Physiological";

    // Confidence
    const confPct = Math.round(s2.confidence * 1000) / 10;
    resConfidenceVal.innerText = `${confPct}%`;
    resConfidenceBar.style.width = `${confPct}%`;

    // Severity
    resSeverityRating.innerText = s3.severity;
    const sevClass = s3.severity.toLowerCase();
    if (sevClass === "healthy") {
      resSeverityRating.style.color = "var(--mint-light)";
    } else if (sevClass === "mild") {
      resSeverityRating.style.color = "var(--mint-light)";
    } else if (sevClass === "moderate") {
      resSeverityRating.style.color = "var(--amber-warning)";
    } else {
      resSeverityRating.style.color = "var(--crimson-danger)";
    }
    resAffectedPct.innerText = `${s3.affected_area_pct.toFixed(1)}% Surface Affected`;

    // Urgent Action Banner
    resImmediateActionText.innerText = s4.immediate_action || "Continue routine preventative care.";
    if (isHealthy) {
      resUrgentBanner.style.borderColor = "rgba(16, 185, 129, 0.3)";
      resUrgentBanner.style.background = "rgba(16, 185, 129, 0.1)";
      const bell = resUrgentBanner.querySelector(".urgent-icon-col");
      if (bell) bell.style.color = "var(--mint-light)";
    } else {
      resUrgentBanner.style.borderColor = "rgba(245, 158, 11, 0.3)";
      resUrgentBanner.style.background = "rgba(245, 158, 11, 0.1)";
      const bell = resUrgentBanner.querySelector(".urgent-icon-col");
      if (bell) bell.style.color = "var(--amber-warning)";
    }

    // Disease Description
    resDescription.innerText = s4.description || "No specific pathogen notes recorded.";

    // 2. Explainability Visualizer
    updateVisualizerView("composite_panel");
    viewerTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-view") === "composite_panel");
    });

    // Severity Progress Bar
    const affPctClamped = Math.min(100, Math.max(0, s3.affected_area_pct));
    meterProgressBar.style.width = `${affPctClamped}%`;
    meterAffectedPctBadge.innerText = `${s3.affected_area_pct.toFixed(1)}%`;

    // Top-5 Candidates
    resTopkList.innerHTML = "";
    if (s2.top_predictions && s2.top_predictions.length > 0) {
      s2.top_predictions.forEach((cand) => {
        const item = document.createElement("div");
        item.className = "topk-item";
        const labelClean = cand.label.replace("___", " &bull; ").replaceAll("_", " ");
        const candConf = (cand.confidence * 100).toFixed(1);
        item.innerHTML = `
          <span class="topk-label">${labelClean}</span>
          <span class="topk-prob">${candConf}%</span>
        `;
        resTopkList.appendChild(item);
      });
    }

    // 3. Populate Agronomic Treatment Lists
    populateList(resCulturalList, s4.cultural_controls, "fa-scissors");
    populateList(resChemicalList, s4.chemical_controls, "fa-spray-can");
    populateList(resBiologicalList, s4.biological_controls, "fa-bug");
    populateList(resPreventiveList, s4.preventive_measures, "fa-shield-halved");

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function populateList(container, items, iconClass) {
    container.innerHTML = "";
    if (!items || items.length === 0) {
      container.innerHTML = `<li class="treatment-item"><i class="fa-solid fa-check text-emerald"></i> No specific actions required.</li>`;
      return;
    }
    items.forEach((txt) => {
      const li = document.createElement("li");
      li.className = "treatment-item";
      li.innerHTML = `
        <i class="fa-solid ${iconClass} treatment-bullet-icon"></i>
        <span>${txt}</span>
      `;
      container.appendChild(li);
    });
  }

  // Visualizer Tab Switching
  viewerTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      viewerTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const viewKey = btn.getAttribute("data-view");
      updateVisualizerView(viewKey);
    });
  });

  function updateVisualizerView(viewKey) {
    if (!currentVisualizations) return;
    const imgData = currentVisualizations[viewKey];
    if (imgData) {
      visMainImage.src = imgData;
    }
  }

  // Treatment Tab Switching
  treatmentTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      treatmentTabBtns.forEach((b) => b.classList.remove("active"));
      treatmentPanels.forEach((p) => p.classList.add("d-none"));

      btn.classList.add("active");
      const targetPanelId = btn.getAttribute("data-treatment-tab");
      const targetPanel = document.getElementById(targetPanelId);
      if (targetPanel) targetPanel.classList.remove("d-none");
    });
  });

  // =========================================================================
  // 9. Supported Classes Modal
  // =========================================================================
  btnOpenClasses.addEventListener("click", async () => {
    classesModal.classList.remove("d-none");
    if (!cachedClassesData) {
      classesGridContainer.innerHTML = `<div style="padding: 20px; color: var(--text-muted);"><i class="fa-solid fa-spinner fa-spin"></i> Loading 38 classes...</div>`;
      try {
        const resp = await fetch("/api/v1/classes");
        if (resp.ok) {
          cachedClassesData = await resp.json();
          renderClassesGrid(cachedClassesData.classes);
        }
      } catch (err) {
        classesGridContainer.innerHTML = `<div style="color: var(--crimson-danger);">Failed to load class directory.</div>`;
      }
    } else {
      renderClassesGrid(cachedClassesData.classes);
    }
  });

  btnCloseClasses.addEventListener("click", () => {
    classesModal.classList.add("d-none");
  });

  classesModal.addEventListener("click", (e) => {
    if (e.target === classesModal) {
      classesModal.classList.add("d-none");
    }
  });

  classesSearchInput.addEventListener("input", (e) => {
    if (!cachedClassesData) return;
    const q = e.target.value.toLowerCase().trim();
    const filtered = cachedClassesData.classes.filter((c) => {
      return c.species.toLowerCase().includes(q) || c.condition.toLowerCase().includes(q);
    });
    renderClassesGrid(filtered);
  });

  function renderClassesGrid(classes) {
    classesGridContainer.innerHTML = "";
    if (classes.length === 0) {
      classesGridContainer.innerHTML = `<div style="padding: 20px; color: var(--text-muted);">No matching crop or pathogen found.</div>`;
      return;
    }
    classes.forEach((c) => {
      const card = document.createElement("div");
      card.className = "class-card";
      card.innerHTML = `
        <span class="class-card-species">${c.species}</span>
        <span class="class-card-condition">${c.condition}</span>
        <span class="class-card-status">${c.is_healthy ? "<span class='text-healthy'>● Healthy</span>" : "<span class='text-amber'>● Diseased</span>"}</span>
      `;
      classesGridContainer.appendChild(card);
    });
  }

  // =========================================================================
  // 10. Print / Export Action
  // =========================================================================
  const triggerPrint = () => {
    window.print();
  };
  btnPrintReport.addEventListener("click", triggerPrint);
  btnPrintAction.addEventListener("click", triggerPrint);

  // =========================================================================
  // 11. Loading & Error Helpers
  // =========================================================================
  function showLoading(msg) {
    loadingProgressText.innerText = msg;
    loadingSection.classList.remove("d-none");
    errorSection.classList.add("d-none");
    resultsSection.classList.add("d-none");
  }

  function hideLoading() {
    loadingSection.classList.add("d-none");
  }

  function showError(msg) {
    hideLoading();
    errorMessage.innerText = msg;
    errorSection.classList.remove("d-none");
    errorSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  btnDismissError.addEventListener("click", () => {
    errorSection.classList.add("d-none");
  });
});
