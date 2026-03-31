// Global state
let currentSession = null;
let outputEventSource = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const configSection = document.getElementById('configSection');
const outputSection = document.getElementById('outputSection');
const analysisForm = document.getElementById('analysisForm');
const outputText = document.getElementById('outputText');
const downloadResults = document.getElementById('downloadResults');
const newAnalysis = document.getElementById('newAnalysis');
const clientSelect = document.getElementById('client');
const ollamaSettings = document.getElementById('ollamaSettings');
const openaiSettings = document.getElementById('openaiSettings');
const modelscopeSettings = document.getElementById('modelscopeSettings');
const statusIndicator = document.getElementById('statusIndicator');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
analysisForm.addEventListener('submit', handleAnalysis);
clientSelect.addEventListener('change', toggleClientSettings);
document.getElementById('temperature').addEventListener('input', (e) => {
    document.getElementById('temperatureValue').textContent = e.target.value;
});
downloadResults.addEventListener('click', downloadAnalysisResults);
newAnalysis.addEventListener('click', resetUI);

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (isValidVideoFile(file)) {
        handleFile(file);
    } else {
        alert('Please upload a valid video file (MP4, AVI, MOV, or MKV)');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isValidVideoFile(file)) {
        handleFile(file);
    }
}

function isValidVideoFile(file) {
    const validTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    return validTypes.some(type => file.name.toLowerCase().endsWith(type));
}

async function handleFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (response.ok) {
            currentSession = data.session_id;
            showConfigSection();
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        alert(`Error uploading file: ${error.message}`);
    }
}

// Analysis Handlers
async function handleAnalysis(e) {
    e.preventDefault();
    if (!currentSession) return;
    
    const formData = new FormData(analysisForm);
    showOutputSection();
    
    if (outputEventSource) {
        outputEventSource.close();
    }
    
    outputText.innerHTML = '';
    statusIndicator.style.display = 'block';
    
    try {
        const response = await fetch(`/analyze/${currentSession}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to start analysis');
        }
    } catch (error) {
        appendConsoleLine(`Error starting analysis: ${error.message}`, 'error');
        finishAnalysis(false);
        return;
    }

    outputEventSource = new EventSource(`/analyze/${currentSession}/stream`);
    
    outputEventSource.onmessage = (event) => {
        const line = event.data;
        appendConsoleLine(line);
        
        if (line.includes('Analysis completed successfully')) {
            finishAnalysis(true);
        } else if (line.includes('Analysis failed')) {
            finishAnalysis(false);
        }
    };
    
    outputEventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        outputEventSource.close();
        appendConsoleLine('Connection to server lost. Please check if the backend is running.', 'error');
        finishAnalysis(false);
    };
}

function appendConsoleLine(text, type = 'info') {
    const div = document.createElement('div');
    div.className = 'console-line';
    if (type === 'error') div.style.color = '#ef4444';
    div.textContent = `> ${text}`;
    outputText.appendChild(div);
    div.parentElement.scrollTop = div.parentElement.scrollHeight;
}

function finishAnalysis(success) {
    if (outputEventSource) outputEventSource.close();
    statusIndicator.style.display = 'none';
    document.querySelector('.output-actions').style.display = 'flex';
    downloadResults.style.display = success ? 'inline-block' : 'none';
}

// UI Updates
function showConfigSection() {
    dropZone.style.display = 'none';
    configSection.style.display = 'block';
    // Set default model based on initial client
    toggleClientSettings();
}

function showOutputSection() {
    configSection.style.display = 'none';
    outputSection.style.display = 'block';
    document.querySelector('.output-actions').style.display = 'none';
}

function toggleClientSettings() {
    const client = clientSelect.value;
    const modelInput = document.getElementById('model');
    
    ollamaSettings.style.display = 'none';
    openaiSettings.style.display = 'none';
    modelscopeSettings.style.display = 'none';
    
    if (client === 'ollama') {
        ollamaSettings.style.display = 'block';
        modelInput.value = 'llama3.2-vision';
    } else if (client === 'openai_api') {
        openaiSettings.style.display = 'block';
        modelInput.value = 'gpt-4-vision-preview';
    } else if (client === 'modelscope') {
        modelscopeSettings.style.display = 'block';
        modelInput.value = window.DEFAULT_MODEL || 'Qwen/Qwen2-VL-7B-Instruct';
    }
}

// Results Handling
async function downloadAnalysisResults() {
    if (!currentSession) return;
    window.location.href = `/results/${currentSession}`;
}

function resetUI() {
    if (currentSession) {
        fetch(`/cleanup/${currentSession}`, { method: 'POST' })
            .catch(error => console.error('Cleanup error:', error));
    }
    
    currentSession = null;
    if (outputEventSource) {
        outputEventSource.close();
    }
    
    analysisForm.reset();
    dropZone.style.display = 'block';
    configSection.style.display = 'none';
    outputSection.style.display = 'none';
    outputText.innerHTML = '';
    fileInput.value = '';
    toggleClientSettings();
}

// Initialize UI
toggleClientSettings();
