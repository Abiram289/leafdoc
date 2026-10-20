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

  // Urgent and Immediate Action Banners
  const resUrgentActionText = document.getElementById("res-urgent-action-text");
  const resRegionalNameBadge = document.getElementById("res-regional-name-badge");
  const resRegionalNameText = document.getElementById("res-regional-name-text");

  // Treatment Elements (Unified Single Tab)
  const resImmediateBanner = document.getElementById("res-immediate-banner");
  const resImmediateSeverityLabel = document.getElementById("res-immediate-severity-label");
  const resChemicalList = document.getElementById("res-chemical-list");
  const resFertilizerList = document.getElementById("res-fertilizer-list");
  const resPreventionList = document.getElementById("res-prevention-list");
  const resCulturalList = document.getElementById("res-cultural-list");
  const resBiologicalList = document.getElementById("res-biological-list");
  const resPreventiveList = document.getElementById("res-preventive-list");
  const treatmentFilterPills = document.querySelectorAll(".tfilter-pill");

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
  function isImageFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith("image/")) return true;
    const name = (file.name || "").toLowerCase();
    return /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(name);
  }

  function processDroppedData(dt) {
    if (!dt) return false;

    // 1. Dropped files from local system (Desktop, File Explorer)
    if (dt.files && dt.files.length > 0) {
      for (let i = 0; i < dt.files.length; i++) {
        const file = dt.files[i];
        if (isImageFile(file)) {
          switchToFileTab();
          handleFileSelected(file);
          showToastNotification(`Loaded: ${file.name}`);
          return true;
        }
      }
      showError("Please drop an image file (JPEG, PNG, or WebP).");
      return true;
    }

    // 2. Dropped items (alternative dataTransfer item API)
    if (dt.items && dt.items.length > 0) {
      for (let i = 0; i < dt.items.length; i++) {
        const item = dt.items[i];
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (file && isImageFile(file)) {
            switchToFileTab();
            handleFileSelected(file);
            showToastNotification(`Loaded: ${file.name}`);
            return true;
          }
        }
      }
    }

    // 3. Dropped image URL from web browser tab or Google Images
    const urlData = dt.getData("text/uri-list") || dt.getData("URL") || dt.getData("text/plain");
    if (urlData && (urlData.startsWith("http://") || urlData.startsWith("https://") || urlData.startsWith("data:image/"))) {
      switchToUrlTab(urlData.trim());
      showToastNotification("Remote image URL loaded! Ready to diagnose.");
      return true;
    }

    // 4. Dropped HTML with <img> tag from external webpage
    const htmlData = dt.getData("text/html");
    if (htmlData) {
      const match = htmlData.match(/<img[^>]+src=["']([^"']+)["']/i);
      if (match && match[1]) {
        const imgSrc = match[1].trim();
        if (imgSrc.startsWith("http://") || imgSrc.startsWith("https://") || imgSrc.startsWith("data:image/")) {
          switchToUrlTab(imgSrc);
          showToastNotification("Web image loaded! Ready to diagnose.");
          return true;
        }
      }
    }

    return false;
  }

  // Prevent default browser behavior (opening file in new tab)
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
    }, false);
    document.addEventListener(eventName, (e) => {
      e.preventDefault();
    }, false);
  });

  // Visual dragover feedback on dropzone
  let dragDepthCounter = 0;
  window.addEventListener("dragenter", (e) => {
    dragDepthCounter++;
    if (dropzone) dropzone.classList.add("dragover");
  });

  window.addEventListener("dragleave", (e) => {
    dragDepthCounter--;
    if (dragDepthCounter <= 0) {
      dragDepthCounter = 0;
      if (dropzone) dropzone.classList.remove("dragover");
    }
  });

  // Dedicated handlers directly on dropzone
  if (dropzone) {
    dropzone.addEventListener("click", (e) => {
      if (e.target !== btnClearFile && (!btnClearFile || !btnClearFile.contains(e.target))) {
        if (fileInput) fileInput.click();
      }
    });

    dropzone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = "copy";
      }
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", (e) => {
      if (e.target === dropzone || !dropzone.contains(e.relatedTarget)) {
        dropzone.classList.remove("dragover");
      }
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragDepthCounter = 0;
      dropzone.classList.remove("dragover");
      processDroppedData(e.dataTransfer);
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Handle drop anywhere on window as well
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepthCounter = 0;
    if (dropzone) dropzone.classList.remove("dragover");
    processDroppedData(e.dataTransfer);
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
      if (isImageFile(file)) {
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
    if (!isImageFile(file)) {
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
    const tamilDiseaseName = s4.tamil_name || "";
    if (tamilDiseaseName) {
      resDiseaseName.innerHTML = `${s2.condition_name} <span class="treatment-tamil-chip" style="margin-left: 8px; font-size: 13px; font-weight: 600; vertical-align: middle;"><i class="fa-solid fa-seedling"></i> ${tamilDiseaseName}</span>`;
    } else {
      resDiseaseName.innerText = s2.condition_name;
    }
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

    // Urgent Action Banner (Top Level)
    if (resImmediateActionText) {
      resImmediateActionText.innerText = s4.immediate_action || "Continue routine preventative care.";
    }
    if (resUrgentActionText) {
      resUrgentActionText.innerText = s4.immediate_action || "Continue routine preventative care.";
    }
    if (resUrgentBanner) {
      if (isHealthy) {
        resUrgentBanner.style.borderColor = "rgba(16, 185, 129, 0.35)";
        resUrgentBanner.style.background = "rgba(16, 185, 129, 0.1)";
        const bell = resUrgentBanner.querySelector(".urgent-icon-col");
        if (bell) bell.style.color = "var(--mint-light)";
      } else {
        resUrgentBanner.style.borderColor = "rgba(245, 158, 11, 0.35)";
        resUrgentBanner.style.background = "rgba(245, 158, 11, 0.1)";
        const bell = resUrgentBanner.querySelector(".urgent-icon-col");
        if (bell) bell.style.color = "var(--amber-warning)";
      }
    }

    // Disease Description & Regional Desi Name Badge
    if (resDescription) {
      const fullDesc = s4.description || "No specific pathogen notes recorded.";
      const regionalMatch = fullDesc.match(/\((?:Commonly known in Indian farming as|Popularly called in India as|Known locally as)\s*([^)]+)\)/i);
      
      if (regionalMatch && resRegionalNameBadge && resRegionalNameText) {
        resRegionalNameText.innerText = regionalMatch[1].trim();
        resRegionalNameBadge.classList.remove("d-none");
        resDescription.innerText = fullDesc.replace(regionalMatch[0], "").trim();
      } else {
        if (resRegionalNameBadge) resRegionalNameBadge.classList.add("d-none");
        resDescription.innerText = fullDesc;
      }
    }

    // 2. Explainability Visualizer
    updateVisualizerView("composite_panel");
    viewerTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-view") === "composite_panel");
    });

    // Severity Progress Bar
    const affPctClamped = Math.min(100, Math.max(0, s3.affected_area_pct));
    if (meterProgressBar) meterProgressBar.style.width = `${affPctClamped}%`;
    if (meterAffectedPctBadge) meterAffectedPctBadge.innerText = `${s3.affected_area_pct.toFixed(1)}%`;

    // Top-5 Candidates
    if (resTopkList) {
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
    }

    // 3. Populate Unified Indian Agronomic Treatment Guide
    if (resImmediateActionText && s4.immediate_action) {
      resImmediateActionText.innerText = s4.immediate_action;
    }
    if (resImmediateSeverityLabel) {
      if (isHealthy) {
        resImmediateSeverityLabel.innerHTML = `<i class="fa-solid fa-circle-check text-emerald"></i> ALL CLEAR &bull; REGULAR FIELD CARE (வழக்கமான பயிர் பராமரிப்பு)`;
        if (resImmediateBanner) {
          resImmediateBanner.style.background = "linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.1) 100%)";
          resImmediateBanner.style.borderColor = "rgba(16, 185, 129, 0.4)";
          resImmediateBanner.style.borderLeftColor = "#10b981";
        }
      } else {
        const sev = s4.severity_level || (s3 ? s3.severity : "") || "ACTION";
        resImmediateSeverityLabel.innerHTML = `<i class="fa-solid fa-bolt text-amber"></i> SEVERITY: <strong>${sev.toUpperCase()}</strong> &bull; TODAY'S PRIORITY ACTION (உடனடி முதல் நடவடிக்கை)`;
        if (resImmediateBanner) {
          resImmediateBanner.style.background = "linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(180, 83, 9, 0.12) 100%)";
          resImmediateBanner.style.borderColor = "rgba(245, 158, 11, 0.45)";
          resImmediateBanner.style.borderLeftColor = "#f59e0b";
        }
      }
    }

    // Populate Section 1: Cures & Sprays
    let cureCount = 0;
    if (isHealthy) {
      if (resChemicalList) {
        resChemicalList.innerHTML = `
          <li class="treatment-item healthy-item cure-item">
            <i class="fa-solid fa-circle-check text-emerald treatment-bullet-icon"></i>
            <div class="treatment-item-wrap">
              <div class="treatment-item-title-row">
                <span class="treatment-item-title text-healthy">No Chemical Sprays Required</span>
              </div>
              <p class="treatment-item-desc">The leaf displays healthy vegetative tissue. Avoid unnecessary fungicide application to protect beneficial microbial balance and save input costs.</p>
            </div>
          </li>
        `;
      }
      cureCount = 0;
    } else {
      populateList(resChemicalList, s4.chemical_controls, "fa-spray-can", "cure-item");
      cureCount = (s4.chemical_controls && s4.chemical_controls.length) ? s4.chemical_controls.length : 0;
    }

    // Populate Section 2: Fertilizers & Tonics
    const fertItems = (s4.fertilizers_and_nutrition && s4.fertilizers_and_nutrition.length > 0)
      ? s4.fertilizers_and_nutrition
      : [
          "Foliar NPK 19:19:19 (Water-Soluble): Mix 75 grams in a 15-liter pump (5 g/L). Spray every 10–12 days to stimulate fresh green foliage recovery.",
          "Calcium Nitrate + Boron: Spray 15g Calcium Nitrate + 5g Boron per 15L pump to strengthen leaf cell walls and stop blossom-end rot.",
          "Chelated Micronutrients (Zinc, Iron, Boron): Spray 20g per 15L pump to correct leaf yellowing chlorosis.",
          "Soil Health & Root Booster: Apply well-rotted Gobar ki khad (FYM) or Vermicompost with Trichoderma viride around root base."
        ];
    populateList(resFertilizerList, fertItems, "fa-seedling", "fert-item");
    const fertCount = fertItems.length;

    // Populate Section 3: Preventions & Desi Upay
    const combinedPreventions = [];
    if (s4.biological_controls && s4.biological_controls.length > 0) {
      s4.biological_controls.forEach((item) => {
        if (!item.toLowerCase().includes("no specific") && !combinedPreventions.includes(item)) {
          combinedPreventions.push(item);
        }
      });
    }
    if (s4.preventive_measures && s4.preventive_measures.length > 0) {
      s4.preventive_measures.forEach((item) => {
        if (!item.toLowerCase().includes("no specific") && !combinedPreventions.includes(item)) {
          combinedPreventions.push(item);
        }
      });
    }
    if (s4.cultural_controls && s4.cultural_controls.length > 0) {
      s4.cultural_controls.forEach((item) => {
        if (!item.toLowerCase().includes("no specific") && !combinedPreventions.includes(item)) {
          combinedPreventions.push(item);
        }
      });
    }
    populateList(resPreventionList, combinedPreventions, "fa-shield-halved", "prev-item");
    const prevCount = combinedPreventions.length;
    const totalSolutionsCount = cureCount + fertCount + prevCount;

    // Backward compatibility lists
    if (resCulturalList) populateList(resCulturalList, s4.cultural_controls, "fa-scissors");
    if (resBiologicalList) populateList(resBiologicalList, s4.biological_controls, "fa-bug");
    if (resPreventiveList && resPreventiveList !== resPreventionList) {
      populateList(resPreventiveList, s4.preventive_measures, "fa-shield-halved");
    }

    // Update pill count badges
    const badgeCure = document.getElementById("count-cure");
    const badgeFert = document.getElementById("count-fertilizer");
    const badgePrev = document.getElementById("count-prevention");
    const badgeAll = document.getElementById("count-all");
    if (badgeCure) badgeCure.innerText = cureCount > 0 ? cureCount : "";
    if (badgeFert) badgeFert.innerText = fertCount > 0 ? fertCount : "";
    if (badgePrev) badgePrev.innerText = prevCount > 0 ? prevCount : "";
    if (badgeAll) badgeAll.innerText = totalSolutionsCount > 0 ? totalSolutionsCount : "";

    // Set default tab: "cure" for diseased, "fertilizer" for healthy
    const defaultFilter = (isHealthy || cureCount === 0) ? "fertilizer" : "cure";
    if (treatmentFilterPills && treatmentFilterPills.length > 0) {
      treatmentFilterPills.forEach(p => {
        p.classList.toggle("active", p.getAttribute("data-filter") === defaultFilter);
      });
      const secCure = document.getElementById("section-cure");
      const secFert = document.getElementById("section-fertilizer");
      const secPrev = document.getElementById("section-prevention");
      if (secCure) secCure.classList.toggle("d-none", defaultFilter !== "cure" && defaultFilter !== "all");
      if (secFert) secFert.classList.toggle("d-none", defaultFilter !== "fertilizer" && defaultFilter !== "all");
      if (secPrev) secPrev.classList.toggle("d-none", defaultFilter !== "prevention" && defaultFilter !== "all");
    }

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function formatTreatmentItem(txt, defaultIcon) {
    if (!txt) return "";
    let trimmed = txt.trim();
    if (/^(none required|no specific|not required)/i.test(trimmed)) {
      return `
        <div class="treatment-item-wrap healthy-tip">
          <div class="treatment-item-title-row">
            <span class="treatment-item-title text-healthy"><i class="fa-solid fa-circle-check"></i> No Intervention Needed</span>
          </div>
          <p class="treatment-item-desc">${trimmed}</p>
        </div>
      `;
    }

    // Find colon separating Title from Details, IGNORED inside brackets [ ] or parentheses ( )
    let colonIdx = -1;
    let bracketDepth = 0;
    for (let i = 0; i < trimmed.length; i++) {
      const ch = trimmed[i];
      if (ch === '[' || ch === '(') {
        bracketDepth++;
      } else if (ch === ']' || ch === ')') {
        if (bracketDepth > 0) bracketDepth--;
      } else if (ch === ':' && bracketDepth === 0) {
        if (i < 130) {
          colonIdx = i;
          break;
        }
      }
    }

    let title = "";
    let body = trimmed;
    if (colonIdx > 0) {
      title = trimmed.substring(0, colonIdx).trim();
      body = trimmed.substring(colonIdx + 1).trim();
    }

    // Extract [Active: ...] or [Chemical: ...]
    let activeTag = "";
    const activeRegex = /\[(?:Active|Chemical|Active Ingredient):\s*([^\]]+)\]/i;
    const activeMatch = title.match(activeRegex) || body.match(activeRegex);
    if (activeMatch) {
      activeTag = `<span class="treatment-active-chip"><i class="fa-solid fa-flask"></i> ${activeMatch[1].trim()}</span>`;
      title = title.replace(activeRegex, "").trim();
      body = body.replace(activeRegex, "").trim();
    }

    // Extract [Tamil: ...] or [Common: ...]
    let tamilTag = "";
    const tamilRegex = /\[(?:Tamil|Local|Common|தமிழ்):\s*([^\]]+)\]/i;
    const tamilMatch = title.match(tamilRegex) || body.match(tamilRegex);
    if (tamilMatch) {
      tamilTag = `<span class="treatment-tamil-chip"><i class="fa-solid fa-seedling"></i> ${tamilMatch[1].trim()}</span>`;
      title = title.replace(tamilRegex, "").trim();
      body = body.replace(tamilRegex, "").trim();
    }

    // Defensive cleanup of stray bracket characters or leading colons
    title = title.replace(/^[\]):,\s-]+/, "").replace(/[\[:(,\s-]+$/, "").trim();
    body = body.replace(/^[\]):,\s-]+/, "").trim();

    // Check for knapsack pump dose regex (e.g. 35-40 grams per 15-liter pump, 8-10 ml per 15L pump, etc.)
    const doseRegex = /(?:take|mix|apply|spray)?\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?\s*(?:grams?|g|ml|liters?|L|kg|tablespoons?)(?:\s*(?:to\s*[0-9]+\s*(?:ml|L|g))?)?\s*(?:per|\/|in\s*a?)\s*(?:15\s*[-–]?\s*(?:liter|l|L)|knapsack|pump|10L|barrel)[^.,;)]*)/i;
    let doseChip = "";
    const doseMatch = (body + " " + title).match(doseRegex);
    if (doseMatch) {
      doseChip = `<span class="treatment-dose-chip"><i class="fa-solid fa-prescription-bottle"></i> ${doseMatch[1].trim()}</span>`;
    }

    // Also detect Tamil dose phrase if present (e.g. 15 லிட்டர் தெளிப்பான் பம்பிற்கு 35 கிராம்)
    const tamilDoseRegex = /(15\s*லிட்டர்\s*(?:தெளிப்பான்\s*)?பம்பிற்கு\s*[0-9]+(?:\s*[-–]\s*[0-9]+)?\s*(?:கிராம்|மி\.லி))/i;
    const tamilDoseMatch = (body + " " + title).match(tamilDoseRegex);
    if (tamilDoseMatch && !doseChip) {
      doseChip = `<span class="treatment-dose-chip"><i class="fa-solid fa-prescription-bottle"></i> ${tamilDoseMatch[1].trim()}</span>`;
    }

    const displayTitle = title || body;
    const displayBody = title ? body : "";

    return `
      <div class="treatment-item-wrap">
        <div class="treatment-item-title-row">
          <span class="treatment-item-title">${displayTitle}</span>
          ${tamilTag}
          ${activeTag}
          ${doseChip}
        </div>
        ${displayBody ? `<p class="treatment-item-desc">${displayBody}</p>` : ""}
      </div>
    `;
  }

  function populateList(container, items, iconClass, categoryClass = "") {
    if (!container) return;
    container.innerHTML = "";
    if (!items || items.length === 0) {
      container.innerHTML = `
        <li class="treatment-item empty ${categoryClass}">
          <i class="fa-solid fa-check text-emerald treatment-bullet-icon"></i>
          <div class="treatment-item-wrap"><span class="treatment-item-desc">No specific actions required.</span></div>
        </li>
      `;
      return;
    }
    items.forEach((txt) => {
      const li = document.createElement("li");
      li.className = `treatment-item ${categoryClass}`.trim();
      li.innerHTML = `
        <i class="fa-solid ${iconClass} treatment-bullet-icon"></i>
        ${formatTreatmentItem(txt, iconClass)}
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

  // Single Tab & Filter Pill Switching
  treatmentFilterPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      treatmentFilterPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      const filter = pill.getAttribute("data-filter");
      const secCure = document.getElementById("section-cure");
      const secFert = document.getElementById("section-fertilizer");
      const secPrev = document.getElementById("section-prevention");

      if (filter === "all") {
        if (secCure) secCure.classList.remove("d-none");
        if (secFert) secFert.classList.remove("d-none");
        if (secPrev) secPrev.classList.remove("d-none");
      } else if (filter === "cure") {
        if (secCure) secCure.classList.remove("d-none");
        if (secFert) secFert.classList.add("d-none");
        if (secPrev) secPrev.classList.add("d-none");
      } else if (filter === "fertilizer") {
        if (secCure) secCure.classList.add("d-none");
        if (secFert) secFert.classList.remove("d-none");
        if (secPrev) secPrev.classList.add("d-none");
      } else if (filter === "prevention") {
        if (secCure) secCure.classList.add("d-none");
        if (secFert) secFert.classList.add("d-none");
        if (secPrev) secPrev.classList.remove("d-none");
      }
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
