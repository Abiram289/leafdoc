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

  // Preprocessor & Detector Controls
  const chkAutoCrop = document.getElementById("chk-auto-crop");
  const chkYoloDetector = document.getElementById("chk-yolo-detector");
  const cropSelector = document.getElementById("crop-selector");

  // Visualizer
  const visMainImage = document.getElementById("vis-main-image");
  const vtabBtnCrop = document.getElementById("vtab-btn-crop");
  const vtabBtnYolo = document.getElementById("vtab-btn-yolo");
  const viewerTabs = document.querySelectorAll(".vtab-btn");
  const meterProgressBar = document.getElementById("meter-progress-bar");
  const meterAffectedPctBadge = document.getElementById("meter-affected-pct-badge");
  const resTopkList = document.getElementById("res-topk-list");

  // Multi-Leaf Candidate Selection Tray
  const multiLeafTray = document.getElementById("multi-leaf-tray");
  const candidateLeavesGrid = document.getElementById("candidate-leaves-grid");
  const resDetectorBadge = document.getElementById("res-detector-badge");

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
  let lastDiagnosedSource = null;

  // =========================================================================
  // 1. Initial Health Check
  // =========================================================================
  async function checkSystemHealth() {
    try {
      const resp = await fetch("/api/v1/health");
      if (!resp.ok) throw new Error("Health check failed");
      const data = await resp.json();

      const devName = data.device_name ? data.device_name : data.device.toUpperCase();
      if (systemStatusText) {
        systemStatusText.innerHTML = `Ready &bull; <strong>${devName}</strong> (${data.total_classes} Classes)`;
      }
      if (systemStatusBadge) {
        systemStatusBadge.title = `PyTorch compute device: ${data.device}, Models verified.`;
      }
    } catch (err) {
      console.warn("Backend not yet responding:", err);
      if (systemStatusText) systemStatusText.innerText = "Engine Initializing...";
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

  // Helpers for tab switching
  function switchToFileTab() {
    tabBtns.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.add("d-none"));
    const uploadBtn = document.getElementById("tab-btn-upload");
    const uploadPanel = document.getElementById("upload-panel");
    if (uploadBtn) uploadBtn.classList.add("active");
    if (uploadPanel) uploadPanel.classList.remove("d-none");
    if (mediaStream) stopCamera();
  }

  function switchToCameraTab() {
    tabBtns.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.add("d-none"));
    const cameraBtn = document.getElementById("tab-btn-camera");
    const cameraPanel = document.getElementById("camera-panel");
    if (cameraBtn) cameraBtn.classList.add("active");
    if (cameraPanel) cameraPanel.classList.remove("d-none");
  }

  function switchToUrlTab(url) {
    tabBtns.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.add("d-none"));
    const urlBtn = document.getElementById("tab-btn-url");
    const urlPanel = document.getElementById("url-panel");
    if (urlBtn) urlBtn.classList.add("active");
    if (urlPanel) urlPanel.classList.remove("d-none");
    if (imageUrlInput && url) imageUrlInput.value = url;
    if (mediaStream) stopCamera();
  }

  // Botanical Toast Notification
  function showToastNotification(message, icon = "fa-leaf") {
    let toast = document.getElementById("leafdoc-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "leafdoc-toast";
      toast.className = "leafdoc-toast";
      document.body.appendChild(toast);
    }
    toast.innerHTML = `<i class="fa-solid ${icon} text-emerald"></i> <span>${message}</span>`;
    toast.classList.add("show");
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
      toast.classList.remove("show");
    }, 3200);
  }

  // =========================================================================
  // 3. File Drag & Drop + Browse + Clipboard Paste
  // =========================================================================
  if (dropzone) {
    dropzone.addEventListener("click", (e) => {
      if (e.target !== btnClearFile && (!btnClearFile || !btnClearFile.contains(e.target))) {
        if (fileInput) fileInput.click();
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // PREVENT BROWSER DEFAULT: Prevent opening dragged files in a new tab
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
    document.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  // Visual dragover feedback on dropzone
  let dragDepthCounter = 0;
  window.addEventListener("dragenter", (e) => {
    dragDepthCounter++;
    dropzone.classList.add("dragover");
  });

  window.addEventListener("dragleave", (e) => {
    dragDepthCounter--;
    if (dragDepthCounter <= 0) {
      dragDepthCounter = 0;
      dropzone.classList.remove("dragover");
    }
  });

  // Handle drop ANYWHERE on page or directly on dropzone
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragDepthCounter = 0;
    dropzone.classList.remove("dragover");

    const dt = e.dataTransfer;
    if (!dt) return;

    // 1. Dropped files from local system
    if (dt.files && dt.files.length > 0) {
      const file = dt.files[0];
      if (file.type && file.type.startsWith("image/")) {
        switchToFileTab();
        handleFileSelected(file);
        showToastNotification(`Loaded: ${file.name}`);
        return;
      } else {
        showError("Please drop an image file (JPEG, PNG, or WebP).");
        return;
      }
    }

    // 2. Dropped items (alternative dataTransfer item API)
    if (dt.items && dt.items.length > 0) {
      for (let i = 0; i < dt.items.length; i++) {
        if (dt.items[i].kind === "file") {
          const file = dt.items[i].getAsFile();
          if (file && file.type && file.type.startsWith("image/")) {
            switchToFileTab();
            handleFileSelected(file);
            showToastNotification(`Loaded: ${file.name}`);
            return;
          }
        }
      }
    }

    // 3. Dropped image URL from web browser tab or Google Images
    const urlData = dt.getData("text/uri-list") || dt.getData("URL") || dt.getData("text/plain");
    if (urlData && (urlData.startsWith("http://") || urlData.startsWith("https://") || urlData.startsWith("data:image/"))) {
      switchToUrlTab(urlData.trim());
      showToastNotification("Remote image URL loaded! Ready to diagnose.");
    }
  });

  // =========================================================================
  // Clipboard Paste (Ctrl + V)
  // =========================================================================
  window.addEventListener("paste", (e) => {
    const activeEl = document.activeElement;
    const activeTag = activeEl ? activeEl.tagName.toLowerCase() : "";
    const isTextInput = (activeTag === "input" && activeEl.type === "text") || activeTag === "textarea";

    const clipboardData = e.clipboardData || window.clipboardData;
    if (!clipboardData) return;

    // 1. Check for image items (Snipping Tool, PrintScreen, Copy Image)
    const items = clipboardData.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        if (items[i].type && items[i].type.startsWith("image/")) {
          e.preventDefault();
          const blob = items[i].getAsFile();
          if (blob) {
            const ext = blob.type.split("/")[1] || "png";
            const file = new File([blob], `pasted_leaf_${Date.now()}.${ext}`, { type: blob.type });
            switchToFileTab();
            handleFileSelected(file);
            showToastNotification("Leaf image pasted from clipboard! Ready to diagnose.", "fa-paste");
            return;
          }
        }
      }
    }

    // 2. Check for image files in clipboardData.files
    if (clipboardData.files && clipboardData.files.length > 0) {
      const file = clipboardData.files[0];
      if (file.type && file.type.startsWith("image/")) {
        e.preventDefault();
        switchToFileTab();
        handleFileSelected(file);
        showToastNotification(`Pasted: ${file.name}`, "fa-paste");
        return;
      }
    }

    // 3. If user copied an image URL and presses Ctrl+V anywhere outside a text input
    if (!isTextInput) {
      const pastedText = clipboardData.getData("text").trim();
      if (pastedText && (pastedText.startsWith("http://") || pastedText.startsWith("https://"))) {
        const hasImgExt = /\.(jpg|jpeg|png|webp|bmp|gif)(\?.*)?$/i.test(pastedText);
        if (hasImgExt) {
          e.preventDefault();
          switchToUrlTab(pastedText);
          showToastNotification("Pasted image URL ready to diagnose!", "fa-paste");
        }
      }
    }
  });

  function handleFileSelected(file, autoScroll = true) {
    if (!file.type || !file.type.startsWith("image/")) {
      showError("Please select a valid image file (JPEG, PNG, WebP).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (previewImage) previewImage.src = ev.target.result;
      if (dropzonePreview) dropzonePreview.classList.remove("d-none");
      if (btnDiagnoseFile) btnDiagnoseFile.classList.remove("d-none");
      // Scroll smoothly to button only if autoScroll is enabled and results are not active
      if (autoScroll && resultsSection && resultsSection.classList.contains("d-none")) {
        btnDiagnoseFile.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    };
    reader.readAsDataURL(file);
  }

  if (btnClearFile) {
    btnClearFile.addEventListener("click", (e) => {
      e.stopPropagation();
      selectedFile = null;
      if (fileInput) fileInput.value = "";
      if (previewImage) previewImage.src = "";
      if (dropzonePreview) dropzonePreview.classList.add("d-none");
      if (btnDiagnoseFile) btnDiagnoseFile.classList.add("d-none");
    });
  }

  if (btnDiagnoseFile) {
    btnDiagnoseFile.addEventListener("click", () => {
      if (selectedFile) {
        runDiagnosisWithFile(selectedFile);
      }
    });
  }

  // =========================================================================
  // 4. Live Webcam Scanner
  // =========================================================================
  if (btnStartCamera) {
    btnStartCamera.addEventListener("click", async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (webcamVideo) webcamVideo.srcObject = mediaStream;
        btnStartCamera.classList.add("d-none");
        if (btnSnapPhoto) btnSnapPhoto.classList.remove("d-none");
      } catch (err) {
        showError("Unable to access camera. Please allow camera permissions or upload an image file.");
        console.error(err);
      }
    });
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
      if (webcamVideo) webcamVideo.srcObject = null;
      if (btnStartCamera) btnStartCamera.classList.remove("d-none");
      if (btnSnapPhoto) btnSnapPhoto.classList.add("d-none");
    }
  }

  if (btnSnapPhoto) {
    btnSnapPhoto.addEventListener("click", () => {
      if (!webcamVideo || !webcamVideo.videoWidth || !webcamCanvas) return;
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
  }

  // =========================================================================
  // 5. Remote Image URL
  // =========================================================================
  if (btnDiagnoseUrl) {
    btnDiagnoseUrl.addEventListener("click", () => {
      if (!imageUrlInput) return;
      const url = imageUrlInput.value.trim();
      if (!url) {
        showError("Please enter a valid image URL.");
        return;
      }
      runDiagnosisWithUrl(url);
    });
  }

  if (imageUrlInput) {
    imageUrlInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (btnDiagnoseUrl) btnDiagnoseUrl.click();
      }
    });
  }

  // =========================================================================
  // 6. Instant 1-Click Samples
  // =========================================================================
  samplePills.forEach((pill) => {
    pill.addEventListener("click", async () => {
      const src = pill.getAttribute("data-src");
      if (!src) return;

      switchToFileTab();
      const title = pill.getAttribute("data-title") || "sample leaf";
      showLoading(`Loading sample leaf: ${title}...`);
      try {
        const response = await fetch(src);
        if (!response.ok) throw new Error("Could not fetch sample leaf file");
        const blob = await response.blob();
        const filename = src.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });

        // Update preview in dropzone without auto-scrolling
        handleFileSelected(file, false);
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
  async function runDiagnosisWithFile(file, selectedBoxIdx = null) {
    lastDiagnosedSource = { type: "file", data: file };
    const actionText = selectedBoxIdx !== null
      ? `Re-diagnosing Candidate Leaf #${selectedBoxIdx + 1} with Dual-Stage AI...`
      : "Running Dual-Stage Deep Learning & Grad-CAM Analysis...";
    showLoading(actionText);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("include_visualizations", "true");
    formData.append("top_k", "5");
    formData.append("auto_crop", chkAutoCrop ? chkAutoCrop.checked.toString() : "true");
    formData.append("use_neural_detector", chkYoloDetector ? chkYoloDetector.checked.toString() : "true");
    if (selectedBoxIdx !== null) {
      formData.append("selected_box_idx", selectedBoxIdx.toString());
    }
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

  async function runDiagnosisWithUrl(url, selectedBoxIdx = null) {
    lastDiagnosedSource = { type: "url", data: url };
    const actionText = selectedBoxIdx !== null
      ? `Re-diagnosing Candidate Leaf #${selectedBoxIdx + 1} from remote image...`
      : "Fetching remote leaf image and running neural inference...";
    showLoading(actionText);

    const formData = new FormData();
    formData.append("image_url", url);
    formData.append("include_visualizations", "true");
    formData.append("top_k", "5");
    formData.append("auto_crop", chkAutoCrop ? chkAutoCrop.checked.toString() : "true");
    formData.append("use_neural_detector", chkYoloDetector ? chkYoloDetector.checked.toString() : "true");
    if (selectedBoxIdx !== null) {
      formData.append("selected_box_idx", selectedBoxIdx.toString());
    }
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

    // YOLO Detector Badge & Detected Leaves Tab
    if (data.detector_used === "yolov8" && resDetectorBadge) {
      resDetectorBadge.classList.remove("d-none");
    } else if (resDetectorBadge) {
      resDetectorBadge.classList.add("d-none");
    }

    if (currentVisualizations.detection_overlay && vtabBtnYolo) {
      vtabBtnYolo.classList.remove("d-none");
    } else if (vtabBtnYolo) {
      vtabBtnYolo.classList.add("d-none");
    }

    // Multi-Leaf Candidate Selection Tray
    if (data.detected_leaves && data.detected_leaves.length > 1 && multiLeafTray && candidateLeavesGrid) {
      multiLeafTray.classList.remove("d-none");
      candidateLeavesGrid.innerHTML = "";
      data.detected_leaves.forEach((leaf) => {
        const card = document.createElement("div");
        card.className = `candidate-leaf-card ${leaf.is_primary ? "active" : ""}`;
        card.title = `Diagnose Leaf Blade #${leaf.index + 1} (${leaf.width}×${leaf.height}px)`;
        card.innerHTML = `
          <div class="candidate-leaf-top">
            <span class="candidate-leaf-rank"><i class="fa-solid fa-leaf text-purple"></i> Blade #${leaf.index + 1}</span>
            ${leaf.is_primary ? '<span class="candidate-primary-pill">Active</span>' : ''}
          </div>
          <div class="candidate-leaf-label">Leaf Blade #${leaf.index + 1}</div>
          <div class="candidate-leaf-meta">
            <span>Box Detection: ${Math.round(leaf.confidence * 100)}%</span>
            <span>${leaf.width}×${leaf.height}px</span>
          </div>
          <button class="candidate-btn-diagnose" type="button">
            ${leaf.is_primary ? '<i class="fa-solid fa-check"></i> Diagnosed' : '<i class="fa-solid fa-magnifying-glass"></i> Diagnose Leaf'}
          </button>
        `;

        card.addEventListener("click", () => {
          if (!leaf.is_primary && lastDiagnosedSource) {
            if (lastDiagnosedSource.type === "file") {
              runDiagnosisWithFile(lastDiagnosedSource.data, leaf.index);
            } else if (lastDiagnosedSource.type === "url") {
              runDiagnosisWithUrl(lastDiagnosedSource.data, leaf.index);
            }
          }
        });

        candidateLeavesGrid.appendChild(card);
      });
    } else if (multiLeafTray) {
      multiLeafTray.classList.add("d-none");
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
    if (data.target_species) {
      const lockNote = document.createElement("div");
      lockNote.className = "topk-lock-note";
      lockNote.style.cssText = "font-size: 12px; color: var(--teal-accent); margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: rgba(20, 184, 166, 0.1); border-radius: var(--radius-sm); border: 1px solid rgba(20, 184, 166, 0.2);";
      lockNote.innerHTML = `
        <span><i class="fa-solid fa-filter"></i> Filter locked to <strong>${data.target_species}</strong></span>
        <button type="button" id="btn-reset-crop-filter" style="background: none; border: none; color: var(--mint-light); cursor: pointer; text-decoration: underline; font-size: 11px; font-weight: 600;">Clear Lock</button>
      `;
      resTopkList.appendChild(lockNote);

      const btnReset = lockNote.querySelector("#btn-reset-crop-filter");
      if (btnReset) {
        btnReset.addEventListener("click", () => {
          if (cropSelector) cropSelector.value = "";
          showToastNotification("Crop filter cleared to Auto-Detect (All 38 Classes)");
          if (lastDiagnosedSource) {
            if (lastDiagnosedSource.type === "file") {
              runDiagnosisWithFile(lastDiagnosedSource.data);
            } else if (lastDiagnosedSource.type === "url") {
              runDiagnosisWithUrl(lastDiagnosedSource.data);
            }
          }
        });
      }
    }

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
  if (btnOpenClasses) {
    btnOpenClasses.addEventListener("click", async () => {
      if (classesModal) classesModal.classList.remove("d-none");
      if (!cachedClassesData) {
        if (classesGridContainer) {
          classesGridContainer.innerHTML = `<div style="padding: 20px; color: var(--text-muted);"><i class="fa-solid fa-spinner fa-spin"></i> Loading 38 classes...</div>`;
        }
        try {
          const resp = await fetch("/api/v1/classes");
          if (resp.ok) {
            cachedClassesData = await resp.json();
            renderClassesGrid(cachedClassesData.classes);
          }
        } catch (err) {
          if (classesGridContainer) {
            classesGridContainer.innerHTML = `<div style="color: var(--crimson-danger);">Failed to load class directory.</div>`;
          }
        }
      } else {
        renderClassesGrid(cachedClassesData.classes);
      }
    });
  }

  if (btnCloseClasses) {
    btnCloseClasses.addEventListener("click", () => {
      if (classesModal) classesModal.classList.add("d-none");
    });
  }

  if (classesModal) {
    classesModal.addEventListener("click", (e) => {
      if (e.target === classesModal) {
        classesModal.classList.add("d-none");
      }
    });
  }

  // Close modal on Escape key
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && classesModal && !classesModal.classList.contains("d-none")) {
      classesModal.classList.add("d-none");
    }
  });

  if (classesSearchInput) {
    classesSearchInput.addEventListener("input", (e) => {
      if (!cachedClassesData) return;
      const q = e.target.value.toLowerCase().trim();
      const filtered = cachedClassesData.classes.filter((c) => {
        return c.species.toLowerCase().includes(q) || c.condition.toLowerCase().includes(q);
      });
      renderClassesGrid(filtered);
    });
  }

  function renderClassesGrid(classes) {
    if (!classesGridContainer) return;
    classesGridContainer.innerHTML = "";
    if (!classes || classes.length === 0) {
      classesGridContainer.innerHTML = `<div style="padding: 20px; color: var(--text-muted);"><i class="fa-solid fa-circle-info"></i> No matching crop or pathogen found.</div>`;
      return;
    }
    classes.forEach((c) => {
      const card = document.createElement("div");
      card.className = "class-card";
      card.style.cursor = "pointer";
      card.title = `Click to filter and diagnose ${c.species}`;
      card.innerHTML = `
        <span class="class-card-species">${c.species}</span>
        <span class="class-card-condition">${c.condition}</span>
        <span class="class-card-status">${c.is_healthy ? "<span class='text-healthy'>● Healthy</span>" : "<span class='text-amber'>● Diseased</span>"}</span>
      `;
      card.addEventListener("click", () => {
        if (cropSelector) {
          cropSelector.value = c.species;
          showToastNotification(`Crop lock set to: ${c.species}`);
        }
        if (classesModal) classesModal.classList.add("d-none");
      });
      classesGridContainer.appendChild(card);
    });
  }

  // =========================================================================
  // 10. Print / Export Action
  // =========================================================================
  const triggerPrint = () => {
    window.print();
  };
  if (btnPrintReport) btnPrintReport.addEventListener("click", triggerPrint);
  if (btnPrintAction) btnPrintAction.addEventListener("click", triggerPrint);

  // =========================================================================
  // 11. Loading & Error Helpers
  // =========================================================================
  function showLoading(msg) {
    if (loadingProgressText) loadingProgressText.innerText = msg;
    if (loadingSection) loadingSection.classList.remove("d-none");
    if (errorSection) errorSection.classList.add("d-none");
    if (resultsSection) resultsSection.classList.add("d-none");
  }

  function hideLoading() {
    if (loadingSection) loadingSection.classList.add("d-none");
  }

  function showError(msg) {
    hideLoading();
    if (errorMessage) errorMessage.innerText = msg;
    if (errorSection) {
      errorSection.classList.remove("d-none");
      errorSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  if (btnDismissError) {
    btnDismissError.addEventListener("click", () => {
      if (errorSection) errorSection.classList.add("d-none");
    });
  }
});
