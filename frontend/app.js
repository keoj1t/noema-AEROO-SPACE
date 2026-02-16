// ===== DEBUG CONFIGURATION =====
const DEBUG_MODE = false;

function logDebug(message, data = null) {
    if (DEBUG_MODE) {
        const timestamp = new Date().toISOString().split('T')[1];
        console.log(`[${timestamp}] [DEBUG] ${message}`, data ? data : '');
    }
}

function logError(message, error = null) {
    const timestamp = new Date().toISOString().split('T')[1];
    console.error(`[${timestamp}] [ERROR] ${message}`, error ? error : '');
}

// ===== FILE UPLOAD =====
async function upload() {
  const fileInput = document.getElementById("fileInput");
  const resultEl = document.getElementById("result");
  if (!fileInput) return;
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    logError('No file selected');
    if (resultEl) resultEl.innerText = 'Please select a file first.';
    return;
  }
  
  logDebug('File upload initiated', { name: file.name, size: file.size });
  
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: form
    });

    logDebug('Upload response received', { status: res.status });
    const data = await res.json();
    logDebug('Upload response data', data);
    if (resultEl) resultEl.innerText = JSON.stringify(data, null, 2);
  } catch (error) {
    logError('Upload failed', error);
    if (resultEl) resultEl.innerText = `Error: ${error.message}`;
  }
}
