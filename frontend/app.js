/**
 * AI Générative Multi-Agents - Frontend JavaScript
 */

// === Configuration ===
const API_BASE = window.location.origin;

// === DOM Elements ===
const elements = {
    chatContainer: document.getElementById('chat-container'),
    promptInput: document.getElementById('prompt-input'),
    charCount: document.getElementById('char-count'),
    optimizeToggle: document.getElementById('optimize-toggle'),
    dataGuardToggle: document.getElementById('data-guard-toggle'),
    generateBtn: document.getElementById('generate-btn'),

    // File Upload
    fileDropZone: document.getElementById('file-drop-zone'),
    fileInput: document.getElementById('file-input'),
    fileList: document.getElementById('file-list'),

    // Stats / Footer
    quotaStatus: document.getElementById('quota-status'),
    availableModels: document.getElementById('available-models'),

    // Lang
    langSelect: document.getElementById('lang-select')
};

// === State ===
let uploadedFiles = [];
let chatHistory = []; // Stores {role, content}
let currentLang = 'fr'; // Default language

// === Provider Icons ===
const providerIcons = {
    groq: '⚡',
    openrouter: '🔀',
    gemini: '💎',
    huggingface: '🤗',
    cohere: '🧠',
    cloudflare: '☁️'
};

// === Event Listeners ===
document.addEventListener('DOMContentLoaded', () => {
    // Initialiser la langue depuis le localStorage si dispo
    const savedLang = localStorage.getItem('ai_website_lang');
    if (savedLang && ['fr', 'en'].includes(savedLang)) {
        currentLang = savedLang;
    }

    initApp();
    updateTexts(); // Appliquer les traductions initiale
});

function initApp() {
    // Character counter
    elements.promptInput.addEventListener('input', () => {
        updateCharCount();
        autoResizeTextarea();
    });

    // Generate button
    elements.generateBtn.addEventListener('click', handleGenerate);

    // Enter key to submit (Shift+Enter for new line)
    elements.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
        }
    });

    // File Upload Events
    if (elements.fileDropZone) {
        // Trigger file input click on button click
        const attachBtn = elements.fileDropZone.querySelector('.attach-btn');
        if (attachBtn) {
            attachBtn.addEventListener('click', () => elements.fileInput.click());
        }

        elements.fileInput.addEventListener('change', handleFileSelect);
    }

    // Load initial quota status
    loadQuotaStatus();

    // Language Switch
    if (elements.langSelect) {
        elements.langSelect.value = currentLang;
        elements.langSelect.addEventListener('change', (e) => {
            currentLang = e.target.value;
            localStorage.setItem('ai_website_lang', currentLang);
            updateTexts();
        });
    }
}

function updateTexts() {
    const t = translations[currentLang];

    // Page Title
    document.title = t.title || "AI Multi-Agents";

    // Header
    const tagline = document.querySelector('.tagline');
    if (tagline) tagline.textContent = t.tagline;

    // Placeholder
    elements.promptInput.placeholder = t.placeholder;

    // Toggles
    const optimizeLabel = elements.optimizeToggle.parentElement.querySelector('.toggle-label');
    if (optimizeLabel) optimizeLabel.textContent = t.optimizeLabel;

    const dataGuardLabel = elements.dataGuardToggle.parentElement.querySelector('.toggle-label');
    if (dataGuardLabel) dataGuardLabel.textContent = t.dataGuardLabel;

    // Welcome Message
    const welcomeH2 = document.querySelector('.welcome-message h2');
    if (welcomeH2) welcomeH2.textContent = t.welcomeTitle;

    const welcomeP = document.querySelector('.welcome-message p');
    if (welcomeP) welcomeP.textContent = t.welcomeDesc;

    // Features
    const features = document.querySelectorAll('.feature-item span:last-child');
    if (features.length >= 3) {
        features[0].textContent = t.featureSpeed;
        features[1].textContent = t.featureGuard;
        features[2].textContent = t.featureMulti;
    }

    // Footer
    const footerText = document.querySelector('.footer-text');
    if (footerText && elements.availableModels) {
        // Preserve the count if it exists
        const count = elements.availableModels.textContent;
        footerText.innerHTML = t.footer.replace('{count}', `<span id="available-models">${count}</span>`);
        // Re-bind the element reference as we replaced innerHTML
        elements.availableModels = document.getElementById('available-models');
    }
}

function updateCharCount() {
    const count = elements.promptInput.value.length;
    if (elements.charCount) {
        elements.charCount.textContent = count.toLocaleString();

        if (count > 9000) elements.charCount.style.color = '#ef4444';
        else if (count > 7500) elements.charCount.style.color = '#f59e0b';
        else elements.charCount.style.color = '';
    }
}

function autoResizeTextarea() {
    const el = elements.promptInput;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

// === API Calls ===
async function handleGenerate() {
    const prompt = elements.promptInput.value.trim();

    // Prevent empty or files-only submissions if that's not desired (usually good to check)
    if (!prompt && uploadedFiles.length === 0) {
        return;
    }

    // Disable input while generating
    setLoading(true);

    try {
        // 1. Prepare Content
        let fullPrompt = prompt;
        let images = [];
        let displayPrompt = prompt;

        if (uploadedFiles.length > 0) {
            const { textFiles, imageFiles } = await readFileContents();

            // Append text files to prompt
            if (textFiles.length > 0) {
                const filesText = textFiles.map(f =>
                    `\n\n--- Fichier: ${f.name} ---\n${f.content}`
                ).join('');
                fullPrompt = prompt + filesText; // Hidden context
                displayPrompt += `\n\n*[${textFiles.length} fichiers texte joints]*`;
            }

            images = imageFiles;
            if (images.length > 0) {
                displayPrompt += `\n\n*[${images.length} images jointes]*`; // TODO: Display thumbnails in chat
            }
        }

        // 2. Add User Message to Chat
        addMessageToChat('user', displayPrompt);
        elements.promptInput.value = '';
        autoResizeTextarea();
        uploadedFiles = []; // Clear files after send
        updateFileList();

        // 3. Add Thinking Placeholder
        const thinkingId = addThinkingMessage();

        // 4. API Call
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: fullPrompt,
                optimize_prompt: elements.optimizeToggle.checked,
                enable_data_guard: elements.dataGuardToggle.checked,
                images: images,
                history: chatHistory,
                language: currentLang
            })
        });

        // Remove thinking bubble
        removeMessage(thinkingId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Erreur HTTP ${response.status}`);
        }

        const data = await response.json();

        // 5. Handle Response
        if (!data.success) {
            addMessageToChat('assistant', `⚠️ **Erreur:** ${data.error || 'La génération a échoué.'}`);
        } else {
            // Update History only on success
            chatHistory.push({ role: 'user', content: fullPrompt }); // Use full prompt with file content for context
            chatHistory.push({ role: 'assistant', content: data.content });

            // Display Assistant Message
            addMessageToChat('assistant', data.content, data);
        }

    } catch (error) {
        console.error('Generation error:', error);
        addMessageToChat('assistant', `❌ **Erreur système:** ${error.message}`);
    } finally {
        setLoading(false);
        loadQuotaStatus();
    }
}

async function loadQuotaStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models/available`);
        if (response.ok) {
            const available = await response.json();
            if (elements.availableModels) elements.availableModels.textContent = available.length;
            if (elements.quotaStatus) {
                const t = translations[currentLang];
                elements.quotaStatus.textContent = t.quotaAvailable.replace('{count}', available.length);
            }
        }
    } catch (e) { console.error(e); }
}

// === Chat Rendering ===

function addMessageToChat(role, content, metadata = null) {
    const msgId = 'msg-' + Date.now();
    const chatContainer = elements.chatContainer;

    // Create Message Element
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.id = msgId;

    let innerHTML = '';

    if (role === 'assistant' && metadata) {
        // Assistant Message with Metadata
        const provider = metadata.model.provider;
        const modelName = metadata.model.model_name;
        const icon = providerIcons[provider] || '🤖';

        innerHTML += `
        <div class="message-bubble">
            <div class="message-header">
                <div class="model-badge-mini">
                    <span class="provider-icon-mini">${icon}</span>
                    <span class="model-name">${modelName}</span>
                </div>
                <div class="message-meta">
                    <span class="meta-tag">${metadata.model.task_type}</span>
                </div>
            </div>
            <div class="message-content">${renderMarkdown(content)}</div>
            
            ${metadata.optimization && metadata.optimization.was_optimized ? `
            <div class="opt-details-mini">
                <div class="opt-toggle-mini" onclick="toggleOptMini('${msgId}')">
                    <span>✨ Prompt optimisé</span>
                    <span class="arrow">▼</span>
                </div>
                <div class="opt-content hidden" id="opt-${msgId}">
                    <p><strong>Original:</strong> ${metadata.optimization.original}</p>
                    <p><strong>Optimisé:</strong> ${metadata.optimization.optimized}</p>
                </div>
            </div>` : ''}
            
            ${metadata.data_guard && metadata.data_guard.was_cleaned ? `
            <div class="warning-mini">
                🛡️ Données nettoyées: ${metadata.data_guard.detected_types.join(', ')}
            </div>` : ''}
        </div>`;
    } else {
        // User Message or Simple Text
        innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${renderMarkdown(content)}</div>
        </div>`;
    }

    msgDiv.innerHTML = innerHTML;

    // Hide welcome message if it's the first message
    const welcome = chatContainer.querySelector('.welcome-message');
    if (welcome) welcome.style.display = 'none';

    chatContainer.appendChild(msgDiv);
    scrollToBottom();

    return msgId;
}

function addThinkingMessage() {
    const msgId = 'thinking-' + Date.now();
    const chatContainer = elements.chatContainer;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant thinking';
    msgDiv.id = msgId;

    msgDiv.innerHTML = `
    <div class="message-bubble">
        <div class="message-content">
            <span class="dot-typing"></span>
            <em>Réflexion en cours...</em>
        </div>
    </div>`;

    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgId;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function toggleOptMini(msgId) {
    const content = document.getElementById(`opt-${msgId}`);
    if (content) content.classList.toggle('hidden');
}
window.toggleOptMini = toggleOptMini; // Make global

// === Helpers like Markdown, File Upload (Existing) ===
// (Including simplified versions of previous helpers)

function setLoading(isLoading) {
    elements.generateBtn.disabled = isLoading;
    if (isLoading) elements.generateBtn.classList.add('loading');
    else elements.generateBtn.classList.remove('loading');

    elements.promptInput.disabled = isLoading;
}

function renderMarkdown(text) {
    if (!text) return '';
    if (typeof marked === 'undefined') return text.replace(/\n/g, '<br>');

    try {
        marked.setOptions({ breaks: true, gfm: true });
        const html = marked.parse(text);

        // Timeout to highlight code
        if (typeof hljs !== 'undefined') {
            setTimeout(() => {
                document.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
            }, 10);
        }
        return html;
    } catch (e) { return text; }
}

// === File Upload Logic ===
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
    e.target.value = '';
}

function addFiles(files) {
    const textExtensions = ['.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.css', '.xml', '.yaml', '.yml', '.log', '.sql', '.java'];
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];
    const allowedExtensions = [...textExtensions, ...imageExtensions];

    files.forEach(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(ext)) return;

        // Max 500KB text, 2MB image
        const isImage = imageExtensions.includes(ext);
        const maxSize = isImage ? 2 * 1024 * 1024 : 500 * 1024;

        if (file.size <= maxSize) {
            file.isImage = isImage;
            uploadedFiles.push(file);
        }
    });

    updateFileList();
}

function updateFileList() {
    elements.fileList.innerHTML = uploadedFiles.map((file, index) => `
        <div class="file-item">
            <span>${file.isImage ? '🖼️' : '📄'} ${file.name}</span>
            <button onclick="removeFile(${index})">×</button>
        </div>
    `).join('');
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileList();
}
window.removeFile = removeFile;

async function readFileContents() {
    const textFiles = [];
    const imageFiles = [];

    for (const file of uploadedFiles) {
        try {
            if (file.isImage) {
                const base64 = await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(file);
                });
                imageFiles.push({ name: file.name, type: file.type, base64: base64 });
            } else {
                const text = await file.text();
                textFiles.push({ name: file.name, content: text });
            }
        } catch (e) {
            console.error(e);
        }
    }
    return { textFiles, imageFiles };
}
