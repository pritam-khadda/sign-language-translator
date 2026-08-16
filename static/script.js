let lastPrediction = "";
let history = JSON.parse(localStorage.getItem("sign_history") || "[]");

// ===============================
// THEME
// ===============================

function toggleTheme() {
    const html = document.documentElement;

    const current = html.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";

    html.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
}


// ===============================
// TOAST
// ===============================

function showToast(message) {
    const container = document.getElementById("toast-container");

    if (!container) return;

    const toast = document.createElement("div");

    toast.className = "toast";
    toast.innerText = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 2500);
}


// ===============================
// UPDATE UI
// ===============================

async function updateUI() {

    try {

        const response = await fetch("/get_data", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("Backend error");
        }

        const data = await response.json();

        // Prediction
        const prediction =
            document.getElementById("live-prediction");

        if (prediction) {

            const newPrediction =
                data.prediction || "?";

            if (newPrediction !== lastPrediction) {

                prediction.innerText = newPrediction;

                lastPrediction = newPrediction;
            }
        }


        // Word
        const word =
            document.getElementById("current-word");

        if (word) {
            word.innerText = data.word || "---";
        }


        // Confidence
        const confidence =
            Number(data.confidence || 0);

        const confValue =
            document.getElementById("conf-val");

        const confBar =
            document.getElementById("conf-bar");

        if (confValue) {
            confValue.innerText =
                confidence.toFixed(1) + "%";
        }

        if (confBar) {
            confBar.style.width =
                Math.min(100, confidence) + "%";
        }


        // Voice status
        const voiceStatus =
            document.getElementById("voice-status");

        if (voiceStatus && data.stats) {

            voiceStatus.innerText =
                "Status: " +
                (data.stats.voice
                    ? "Enabled"
                    : "Disabled");
        }


        // Connection
        const connection =
            document.getElementById("conn-status");

        if (connection) {

            connection.innerHTML =
                "● AI SYSTEM ONLINE";

            connection.style.color =
                "#10B981";
        }


        // History
        if (
            data.word &&
            data.word !== "---" &&
            data.word.length > 0
        ) {

            const exists =
                history.some(
                    item => item.text === data.word
                );

            if (!exists) {
                addToHistory(data.word);
            }
        }

    } catch (error) {

        console.error("get_data error:", error);

        const connection =
            document.getElementById("conn-status");

        if (connection) {

            connection.innerHTML =
                "● BACKEND RECONNECTING";

            connection.style.color =
                "#EF4444";
        }
    }
}


// ===============================
// HISTORY
// ===============================

function addToHistory(text) {

    if (!text || text === "---") {
        return;
    }

    const entry = {
        text: text,
        time: new Date().toLocaleTimeString(),
        id: Date.now()
    };

    history.unshift(entry);

    if (history.length > 20) {
        history.pop();
    }

    localStorage.setItem(
        "sign_history",
        JSON.stringify(history)
    );

    renderHistory();
}


function renderHistory() {

    const container =
        document.getElementById("history-list");

    if (!container) return;

    if (history.length === 0) {

        container.innerHTML =
            `<p style="opacity:0.6;">No history yet.</p>`;

        return;
    }

    container.innerHTML =
        history.map(item => `

            <div class="history-item">

                <div>
                    <strong>${item.text}</strong>

                    <br>

                    <small style="opacity:0.5">
                        ${item.time}
                    </small>
                </div>

                <div>

                    <button
                        onclick="copySpecific('${item.text}')"
                        class="btn btn-secondary"
                        type="button">
                        <i class="fas fa-copy"></i>
                    </button>

                    <button
                        onclick="deleteHistory(${item.id})"
                        class="btn btn-secondary"
                        type="button">
                        <i class="fas fa-trash"></i>
                    </button>

                </div>

            </div>

        `).join("");
}


function deleteHistory(id) {

    history =
        history.filter(
            item => item.id !== id
        );

    localStorage.setItem(
        "sign_history",
        JSON.stringify(history)
    );

    renderHistory();

    showToast("History item deleted");
}


function clearHistory() {

    history = [];

    localStorage.removeItem(
        "sign_history"
    );

    renderHistory();

    showToast("History cleared");
}


// ===============================
// COPY
// ===============================

async function copyTranslation() {

    const element =
        document.getElementById("current-word");

    if (!element) return;

    const text =
        element.innerText.trim();

    if (
        !text ||
        text === "---" ||
        text === "Waiting..."
    ) {

        showToast("Nothing to copy");

        return;
    }

    try {

        await navigator.clipboard.writeText(text);

        showToast("Translation copied!");

    } catch (error) {

        console.error(error);

        showToast("Copy failed");
    }
}


async function copySpecific(text) {

    try {

        await navigator.clipboard.writeText(text);

        showToast("Copied!");

    } catch (error) {

        console.error(error);

        showToast("Copy failed");
    }
}


// ===============================
// CLEAR WORD
// ===============================

async function clearCurrentWord() {

    try {

        const response =
            await fetch("/clear", {
                cache: "no-store"
            });

        if (!response.ok) {
            throw new Error("Clear failed");
        }

        document.getElementById(
            "current-word"
        ).innerText = "---";

        document.getElementById(
            "live-prediction"
        ).innerText = "?";

        document.getElementById(
            "conf-val"
        ).innerText = "0%";

        document.getElementById(
            "conf-bar"
        ).style.width = "0%";

        lastPrediction = "";

        showToast("Translation cleared");

    } catch (error) {

        console.error(error);

        showToast("Clear failed");
    }
}


// ===============================
// VOICE
// ===============================

async function toggleVoice() {

    try {

        const response =
            await fetch("/toggle_voice", {
                cache: "no-store"
            });

        if (!response.ok) {
            throw new Error("Voice request failed");
        }

        const data =
            await response.json();

        const status =
            document.getElementById(
                "voice-status"
            );

        if (status) {

            status.innerText =
                "Status: " +
                (data.voice
                    ? "Enabled"
                    : "Disabled");
        }

        showToast(
            data.voice
                ? "Voice Enabled 🔊"
                : "Voice Disabled 🔇"
        );

    } catch (error) {

        console.error(error);

        showToast("Voice toggle failed");
    }
}


// ===============================
// DOWNLOAD HISTORY
// ===============================

function downloadHistory() {

    if (history.length === 0) {

        showToast("No history to download");

        return;
    }

    const content =
        history
            .map(
                h => `[${h.time}] ${h.text}`
            )
            .join("\n");

    const blob =
        new Blob(
            [content],
            { type: "text/plain" }
        );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;
    link.download = "SignSpeak_History.txt";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);

    showToast("History downloaded");
}


// ===============================
// FULLSCREEN
// ===============================

function openFullscreen() {

    const video =
        document.getElementById("videoStream");

    if (!video) {

        showToast("Camera not found");

        return;
    }

    if (video.requestFullscreen) {

        video.requestFullscreen();

    } else {

        showToast(
            "Fullscreen not supported"
        );
    }
}


// ===============================
// KEYBOARD SHORTCUTS
// ===============================

document.addEventListener(
    "keydown",
    function(event) {

        if (
            event.target.tagName === "INPUT" ||
            event.target.tagName === "TEXTAREA"
        ) {
            return;
        }

        const key =
            event.key.toLowerCase();

        if (key === "c") {
            clearCurrentWord();
        }

        if (key === "v") {
            toggleVoice();
        }

        if (key === "f") {
            openFullscreen();
        }
    }
);


// ===============================
// INITIALIZATION
// ===============================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        const savedTheme =
            localStorage.getItem("theme");

        document.documentElement.setAttribute(
            "data-theme",
            savedTheme === "light"
                ? "light"
                : "dark"
        );

        renderHistory();

        updateUI();

        setInterval(
            updateUI,
            500
        );

        console.log(
            "✅ Sign Speak AI JS loaded"
        );
    }
);
