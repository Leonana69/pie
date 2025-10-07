let profilingData = null;
let expandedNodes = new Set();
let minTimeFilter = 0;
let currentDirectory = '';
let selectedFilePath = null;

// Initialize
window.addEventListener('load', () => {
    setupFileLoader();

    // Ensure expand all checkbox is unchecked by default
    document.getElementById('expandAll').checked = false;

    // Auto-load if data is already available
    fetch('/data.json')
        .then(r => {
            if (r.ok) return r.json();
            throw new Error('No initial data');
        })
        .then(data => {
            profilingData = data;
            showContent();
            renderAll();
        })
        .catch(() => {
            // No initial data, show file loader
            console.log('No initial data, waiting for file selection');
        })
        .finally(() => {
            // Load directory browser regardless
            loadDirectory(null);
        });
});

function setupFileLoader() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) loadFromFile(file);
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.json')) {
            loadFromFile(file);
        } else {
            alert('Please drop a valid JSON file');
        }
    });
}

function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            profilingData = data;
            showContent();
            renderAll();
        } catch (err) {
            alert('Error parsing JSON: ' + err.message);
        }
    };
    reader.readAsText(file);
}

function loadDirectory(path) {
    // Show loading state
    const browser = document.getElementById('directoryBrowser');
    browser.classList.add('loading');

    const pathDisplay = document.getElementById('currentPath');
    const dirList = document.getElementById('dirList');
    const fileList = document.getElementById('fileList');

    // Clear previous content
    dirList.innerHTML = '';
    fileList.innerHTML = '';

    currentDirectory = path;

    const url = (path === null || path === '') ? '/list_dirs' : `/list_dirs?path=${encodeURIComponent(path)}`;
    const fileUrl = (path === null || path === '') ? '/list_files' : `/list_files?path=${encodeURIComponent(path)}`;

    // Load both directories and files in parallel
    Promise.all([
        fetch(url).then(r => r.json()),
        fetch(fileUrl).then(r => r.json())
    ])
    .then(([dirs, files]) => {
        console.log('Loaded directories:', dirs);
        console.log('Loaded files:', files);

        // Update path display
        if (path) {
            pathDisplay.textContent = path;
        } else {
            pathDisplay.textContent = '(current directory)';
        }

        // Render directories
        if (dirs.length === 0) {
            dirList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No subdirectories</li>';
        } else {
            dirs.forEach(dir => {
                const li = document.createElement('li');
                li.className = 'dir-item';
                li.innerHTML = `
                    <span class="dir-icon">📁</span>
                    <div class="file-info">
                        <div class="dir-name">${dir.name}</div>
                    </div>
                `;
                li.onclick = () => loadDirectory(dir.path);
                dirList.appendChild(li);
            });
        }

        // Render files
        if (files.length === 0) {
            fileList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No JSON files in this directory</li>';
        } else {
            files.forEach(file => {
                const li = document.createElement('li');
                li.className = 'file-item';
                const date = new Date(file.mtime * 1000).toLocaleString();
                const size = (file.size / 1024).toFixed(1);

                li.innerHTML = `
                    <span class="file-icon">📄</span>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${size} KB • ${date}</div>
                    </div>
                `;
                li.onclick = () => selectFile(file.path, li);
                fileList.appendChild(li);
            });
        }
    })
    .catch(err => {
        console.error('Error loading directory:', err);
        pathDisplay.textContent = 'Error loading directory';
        dirList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading directories</li>';
        fileList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading files</li>';
    })
    .finally(() => {
        // Hide loading state
        browser.classList.remove('loading');
    });
}

function selectFile(path, element) {
    // Remove previous selection
    document.querySelectorAll('.file-item').forEach(item => {
        item.classList.remove('selected');
    });

    // Mark as selected
    element.classList.add('selected');
    selectedFilePath = path;
    document.getElementById('loadButton').disabled = false;
}

function loadSelectedFile() {
    if (!selectedFilePath) {
        alert('Please select a file');
        return;
    }

    fetch(`/load_file?path=${encodeURIComponent(selectedFilePath)}`)
        .then(r => r.json())
        .then(data => {
            profilingData = data;
            showContent();
            renderAll();
        })
        .catch(err => alert('Error loading file: ' + err.message));
}

function showContent() {
    const fileLoader = document.getElementById('fileLoader');
    fileLoader.classList.add('compact');
    fileLoader.querySelector('h3').textContent = '📂 Load Another File';
    document.getElementById('contentWrapper').classList.add('visible');
}

function switchTab(tab) {
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tab + '-tab').classList.add('active');

    // Re-render flame graph when switching to it (to get correct width)
    if (tab === 'flame') {
        setTimeout(() => renderFlameGraph(), 10);
    }
}

function renderAll() {
    renderMetadata();
    renderStats();
    renderTree();
    renderTable();
    renderFlameGraph();
}

function renderMetadata() {
    const meta = document.getElementById('metadata');
    meta.innerHTML = `
        <strong>Timestamp:</strong> ${new Date(profilingData.timestamp).toLocaleString()}
    `;
}

function renderStats() {
    const stats = document.getElementById('stats');
    const tree = profilingData.profiling_tree;

    let totalTime = 0;
    let totalOps = 0;
    let maxTime = 0;
    let maxOp = '';

    function traverse(nodes) {
        nodes.forEach(node => {
            totalOps++;
            totalTime += node.avg_latency_ms;
            if (node.avg_latency_ms > maxTime) {
                maxTime = node.avg_latency_ms;
                maxOp = node.name;
            }
            if (node.children) traverse(node.children);
        });
    }

    traverse(tree);

    stats.innerHTML = `
        <div class="stat-card">
            <div class="label">Total Time</div>
            <div class="value">${tree[0].avg_latency_ms.toFixed(2)}<span class="unit">ms</span></div>
        </div>
        <div class="stat-card">
            <div class="label">Operations</div>
            <div class="value">${totalOps}</div>
        </div>
        <div class="stat-card">
            <div class="label">Slowest Operation</div>
            <div class="value">${maxOp}</div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">${maxTime.toFixed(2)} ms</div>
        </div>
    `;
}

function renderTree() {
    const container = document.getElementById('treeView');
    container.innerHTML = '<ul class="tree-node">' +
        profilingData.profiling_tree.map(node => renderNode(node)).join('') +
        '</ul>';
}

function renderNode(node) {
    if (node.avg_latency_ms < minTimeFilter) return '';

    const hasChildren = node.children && node.children.length > 0;
    const nodeId = node.full_path;
    const isExpanded = expandedNodes.has(nodeId);

    // Use percent_of_total if available, otherwise fall back to 100 for root
    const percent = node.percent_of_total !== undefined ? node.percent_of_total.toFixed(1) : '100.0';

    return `
        <li class="tree-node">
            <div class="node-content" onclick="toggleNode('${nodeId}')" style="cursor: ${hasChildren ? 'pointer' : 'default'}">
                <span class="toggle ${hasChildren ? '' : 'empty'}">
                    ${hasChildren ? (isExpanded ? '▼' : '▶') : ''}
                </span>
                <span class="node-name">${node.name}</span>
                <span class="node-time">${node.avg_latency_ms.toFixed(2)} ms</span>
                <span class="node-percent">${percent}%</span>
                <div class="node-bar">
                    <div class="node-bar-fill" style="width: ${Math.min(parseFloat(percent), 100)}%"></div>
                </div>
                <span class="node-samples">${node.samples} sample${node.samples > 1 ? 's' : ''}</span>
            </div>
            ${hasChildren ? `
                <ul class="children ${isExpanded ? '' : 'hidden'}" id="children-${nodeId}">
                    ${node.children.map(child => renderNode(child)).join('')}
                </ul>
            ` : ''}
        </li>
    `;
}

function toggleNode(nodeId) {
    if (expandedNodes.has(nodeId)) {
        expandedNodes.delete(nodeId);
    } else {
        expandedNodes.add(nodeId);
    }
    renderTree();
}

function toggleExpandAll() {
    const expandAll = document.getElementById('expandAll').checked;

    function collectAllNodes(nodes) {
        let ids = [];
        nodes.forEach(node => {
            ids.push(node.full_path);
            if (node.children) {
                ids = ids.concat(collectAllNodes(node.children));
            }
        });
        return ids;
    }

    if (expandAll) {
        expandedNodes = new Set(collectAllNodes(profilingData.profiling_tree));
    } else {
        expandedNodes.clear();
    }
    renderTree();
}

function applyFilters() {
    minTimeFilter = parseFloat(document.getElementById('minTime').value) || 0;
    renderTree();
    renderTable();
}

function renderTable() {
    const container = document.getElementById('tableView');

    let rows = [];
    function traverse(nodes, depth = 0) {
        nodes.forEach(node => {
            if (node.avg_latency_ms >= minTimeFilter) {
                rows.push({...node, depth});
                if (node.children) traverse(node.children, depth + 1);
            }
        });
    }
    traverse(profilingData.profiling_tree);

    // Sort by time descending
    rows.sort((a, b) => b.avg_latency_ms - a.avg_latency_ms);

    container.innerHTML = `
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Avg Time (ms)</th>
                    <th>Min (ms)</th>
                    <th>Max (ms)</th>
                    <th>Std Dev (ms)</th>
                    <th>Samples</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(row => `
                    <tr>
                        <td style="padding-left: ${row.depth * 20 + 12}px">${row.name}</td>
                        <td><strong>${row.avg_latency_ms.toFixed(3)}</strong></td>
                        <td>${row.min_ms.toFixed(3)}</td>
                        <td>${row.max_ms.toFixed(3)}</td>
                        <td>${row.std_dev_ms.toFixed(3)}</td>
                        <td>${row.samples}</td>
                        <td>${row.percent_of_total ? row.percent_of_total.toFixed(1) + '%' : '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderFlameGraph() {
    const container = document.getElementById('flameView');

    if (!profilingData || !profilingData.profiling_tree) {
        console.error('No profiling data available for flame graph');
        container.innerHTML = '<div style="padding: 20px; color: #999;">No profiling data loaded</div>';
        return;
    }

    // Get container width, use a minimum if container is hidden
    const containerWidth = container.clientWidth || container.parentElement.clientWidth || 1200;
    const width = Math.max(containerWidth - 60, 600);
    const barHeight = 20;
    const barGap = 1;

    console.log('Rendering flame graph, container width:', containerWidth, 'svg width:', width);
    console.log('Profiling tree:', profilingData.profiling_tree);

    // Calculate total height needed
    function getMaxDepth(node, depth = 0) {
        if (!node.children || node.children.length === 0) return depth;
        return Math.max(...node.children.map(c => getMaxDepth(c, depth + 1)));
    }

    const maxDepth = Math.max(...profilingData.profiling_tree.map(n => getMaxDepth(n))) + 1;
    const height = maxDepth * (barHeight + barGap) + 40;

    console.log('Max depth:', maxDepth, 'Height:', height);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.style.background = '#fff';

    // Calculate total time for root width calculation
    const totalTime = profilingData.profiling_tree.reduce((sum, n) => sum + n.avg_latency_ms, 0);

    function renderFlameNode(node, x, nodeWidth, depth) {
        if (nodeWidth < 0.5) return;

        const y = depth * (barHeight + barGap);

        // Material Design color palette
        const materialColors = [
            '#EF5350', // Red 400
            '#EC407A', // Pink 400
            '#AB47BC', // Purple 400
            '#7E57C2', // Deep Purple 400
            '#5C6BC0', // Indigo 400
            '#42A5F5', // Blue 400
            '#29B6F6', // Light Blue 400
            '#26C6DA', // Cyan 400
            '#26A69A', // Teal 400
            '#66BB6A', // Green 400
            '#9CCC65', // Light Green 400
            '#D4E157', // Lime 400
            '#FFEE58', // Yellow 400
            '#FFCA28', // Amber 400
            '#FFA726', // Orange 400
            '#FF7043', // Deep Orange 400
        ];

        // Create rect
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', nodeWidth);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('class', 'flame-rect');

        // Pick color based on depth, cycling through material palette
        const color = materialColors[depth % materialColors.length];
        rect.setAttribute('fill', color);

        // Add tooltip
        const percent = node.percent_of_total ? node.percent_of_total.toFixed(1) : '?';
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${node.name}\n${node.avg_latency_ms.toFixed(2)} ms (${percent}%)\n${node.samples} sample${node.samples > 1 ? 's' : ''}`;
        rect.appendChild(title);

        svg.appendChild(rect);

        // Add text if wide enough
        if (nodeWidth > 40) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x + 4);
            text.setAttribute('y', y + barHeight / 2 + 4);
            text.setAttribute('class', 'flame-text');
            text.setAttribute('fill', '#fff');
            text.setAttribute('font-size', '11px');
            text.setAttribute('font-family', 'sans-serif');

            // Truncate text to fit
            const maxChars = Math.floor(nodeWidth / 7);
            const displayText = node.name.length > maxChars
                ? node.name.substring(0, maxChars - 1) + '…'
                : node.name;
            text.textContent = displayText;
            svg.appendChild(text);
        }

        // Render children
        if (node.children && node.children.length > 0) {
            let childX = x;
            const nodeTime = node.avg_latency_ms;

            node.children.forEach(child => {
                // Calculate child width proportional to parent width
                const childProportion = child.avg_latency_ms / nodeTime;
                const childWidth = nodeWidth * childProportion;

                renderFlameNode(child, childX, childWidth, depth + 1);
                childX += childWidth;
            });
        }
    }

    // Render all root nodes
    let currentX = 0;
    profilingData.profiling_tree.forEach(node => {
        const nodeWidth = (node.avg_latency_ms / totalTime) * width;
        console.log(`Rendering root node: ${node.name}, width: ${nodeWidth}, x: ${currentX}`);
        renderFlameNode(node, currentX, nodeWidth, 0);
        currentX += nodeWidth;
    });

    console.log('Total SVG children:', svg.childNodes.length);

    container.innerHTML = '';
    container.appendChild(svg);

    console.log('Flame graph rendered successfully');
}
