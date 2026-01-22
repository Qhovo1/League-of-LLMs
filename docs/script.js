// Global variables
let currentData = [];
let allData = {
    math: [
        {"Rank": 1, "Model": "gemini-2.5-pro-exp-03-25", "From": "Google", "Score": 85.64626},
        {"Rank": 2, "Model": "deepseek-r1", "From": "DeepSeek", "Score": 69.86395},
        {"Rank": 3, "Model": "gpt-4.1", "From": "OpenAI", "Score": 64.62585},
        {"Rank": 4, "Model": "o3-mini", "From": "OpenAI", "Score": 59.52381},
        {"Rank": 5, "Model": "o1", "From": "OpenAI", "Score": 57.0068},
        {"Rank": 6, "Model": "deepseek-v3", "From": "DeepSeek", "Score": 47.55102},
        {"Rank": 7, "Model": "claude-3-7-sonnet-20250219", "From": "Anthropic", "Score": 42.51701},
        {"Rank": 8, "Model": "qwen2.5-max", "From": "Alibaba", "Score": 30.34014}
    ],
    programming: [
        {"Rank": 1, "Model": "gemini-2.5-pro-exp-03-25", "From": "Google", "Score": 89.6},
        {"Rank": 2, "Model": "o3-mini", "From": "OpenAI", "Score": 79.546938776},
        {"Rank": 3, "Model": "deepseek-r1", "From": "DeepSeek", "Score": 77.9061224498},
        {"Rank": 4, "Model": "gpt-4.1", "From": "OpenAI", "Score": 76.608163265},
        {"Rank": 5, "Model": "o1", "From": "OpenAI", "Score": 76.334693877},
        {"Rank": 6, "Model": "deepseek-v3", "From": "DeepSeek", "Score": 67.248979592},
        {"Rank": 7, "Model": "claude-3-7-sonnet-20250219", "From": "Anthropic", "Score": 60.216326531},
        {"Rank": 8, "Model": "qwen2.5-max", "From": "Alibaba", "Score": 55.5510204087}
    ]
};

// Company logo mapping
const companyLogos = {
    "Google": "logo/gemini.png",
    "OpenAI": "logo/gpt.png",
    "DeepSeek": "logo/Deepseek.png",
    "Anthropic": "logo/Claude.png",
    "Alibaba": "logo/QWen.png"
};

let currentTask = 'math';
let currentSort = {
    column: 'rank',
    direction: 'asc'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    // Process the embedded data
    allData.math = processData(allData.math);
    allData.programming = processData(allData.programming);
    // Display the initial data (Math by default)
    updateTable();
});

// Set up event listeners
function initializeEventListeners() {
    // Task selector change event
    const taskSelector = document.getElementById('taskSelector');
    taskSelector.addEventListener('change', function() {
        currentTask = this.value;
        updateTable();
    });

    // Search box input event
    const searchBox = document.getElementById('searchBox');
    searchBox.addEventListener('input', function() {
        filterTable(this.value);
    });

    // Table header click events for sorting
    const tableHeaders = document.querySelectorAll('th.sortable');
    tableHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const column = this.getAttribute('data-column');
            sortTable(column);
        });
    });
}

// Process raw data into standardized format
function processData(rawData) {
    return rawData.map((row, index) => {
        // Handle different possible column names
        return {
            rank: row.Rank || row.rank || (index + 1),
            model: row.Model || row.model || '',
            from: row.From || row.from || '',
            score: parseFloat(row.Score || row.score || 0)
        };
    }).filter(row => row.model); // Filter out empty rows
}

// Update the table with current task data
function updateTable() {
    currentData = [...allData[currentTask]];

    // Apply current sort
    applySortToData();

    // Clear search boximage.png
    document.getElementById('searchBox').value = '';

    // Render the table
    renderTable(currentData);
}

// Filter table based on search query
function filterTable(query) {
    if (!query.trim()) {
        renderTable(currentData);
        return;
    }

    const searchTerm = query.toLowerCase();
    const filtered = currentData.filter(row =>
        row.model.toLowerCase().includes(searchTerm)
    );

    renderTable(filtered);
}

// Sort table by column
function sortTable(column) {
    // Toggle sort direction if clicking the same column
    if (currentSort.column === column) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = column;
        currentSort.direction = 'asc';
    }

    applySortToData();

    // Get current search query
    const searchQuery = document.getElementById('searchBox').value;
    if (searchQuery.trim()) {
        filterTable(searchQuery);
    } else {
        renderTable(currentData);
    }

    updateSortIndicators();
}

// Apply sorting to current data
function applySortToData() {
    const { column, direction } = currentSort;
    const multiplier = direction === 'asc' ? 1 : -1;

    currentData.sort((a, b) => {
        let aVal = a[column];
        let bVal = b[column];

        // Handle numeric values
        if (column === 'rank' || column === 'score') {
            aVal = parseFloat(aVal) || 0;
            bVal = parseFloat(bVal) || 0;
        } else {
            // Handle string values (case-insensitive)
            aVal = String(aVal).toLowerCase();
            bVal = String(bVal).toLowerCase();
        }

        if (aVal < bVal) return -1 * multiplier;
        if (aVal > bVal) return 1 * multiplier;
        return 0;
    });
}

// Update sort indicators in table headers
function updateSortIndicators() {
    // Remove all existing sort classes
    document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
    });

    // Add class to currently sorted column
    const sortedHeader = document.querySelector(`th[data-column="${currentSort.column}"]`);
    if (sortedHeader) {
        sortedHeader.classList.add(`sorted-${currentSort.direction}`);
    }
}

// Render table with data
function renderTable(data) {
    const tbody = document.getElementById('leaderboardBody');

    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="no-results">No results found</td></tr>';
        return;
    }

    tbody.innerHTML = data.map(row => {
        const logoPath = companyLogos[row.from] || '';
        const modelColumn = logoPath
            ? `<div class="company-cell"><img src="${logoPath}" alt="${escapeHtml(row.from)}" class="company-logo">${escapeHtml(row.model)}</div>`
            : escapeHtml(row.model);

        return `
        <tr>
            <td>${row.rank}</td>
            <td>${modelColumn}</td>
            <td>${escapeHtml(row.from)}</td>
            <td>${row.score.toFixed(1)}</td>
        </tr>
    `;
    }).join('');

    updateSortIndicators();
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
