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

// ===== CHAT WIDGET =====
function escapeHtml(str) {
  if (str == null) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

const chat = document.getElementById("chatMessages");
const input = document.getElementById("chatInput");

if (chat && input) {
  logDebug('Chat widget initialized');

  input.addEventListener("keydown", async e => {
    if (e.key === "Enter") {
      const msg = input.value.trim();
      if (!msg) return;
      logDebug('User message sent', { message: msg });
      input.value = "";

      const userDiv = document.createElement("div");
      userDiv.innerHTML = "<strong>You:</strong> " + escapeHtml(msg);
      chat.appendChild(userDiv);

      try {
        const res = await fetch("/agent", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({message: msg})
        });

        logDebug('Agent response received', { status: res.status });
        const data = await res.json();
        logDebug('Agent response data', data);

        const botDiv = document.createElement("div");
        botDiv.innerHTML = "<strong>NOEMA:</strong> " + escapeHtml(data.answer != null ? data.answer : '');
        chat.appendChild(botDiv);
      } catch (error) {
        logError('Agent request failed', error);
        const errDiv = document.createElement("div");
        errDiv.innerHTML = "<strong>NOEMA:</strong> Error: " + escapeHtml(error.message || 'Unknown error');
        chat.appendChild(errDiv);
      }
    }
  });
}
