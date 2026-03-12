/* Nexus Knowledge Graph — D3 force-directed visualization */

const TYPE_BASE = {
    person:  { r: 234, g: 179, b: 56 },
    org:     { r: 200, g: 65,  b: 55 },
    country: { r: 210, g: 195, b: 160 },
    concept: { r: 220, g: 140, b: 55 },
    treaty:  { r: 150, g: 175, b: 75 },
    unknown: { r: 150, g: 150, b: 160 },
};

function rgb(type) {
    const c = TYPE_BASE[type] || TYPE_BASE.unknown;
    return `${c.r}, ${c.g}, ${c.b}`;
}

let simulation, svg, g, linkGroup, nodeGroup, allNodes, allLinks;
let currentData = null;
let activeTypes = new Set(['person', 'org', 'country', 'concept', 'treaty', 'unknown']);

const LABEL_PERCENTILE = 0.35;

function initGraph(focusEntityId) {
    const container = document.querySelector('.graph-canvas-wrapper');
    const width = container.clientWidth;
    const height = Math.max(container.clientHeight, 600);

    svg = d3.select('#graph-svg')
        .attr('width', width)
        .attr('height', height);

    const defs = svg.append('defs');

    // Two glow filters per type: idle (subtle) and hover (strong)
    Object.entries(TYPE_BASE).forEach(([type, c]) => {
        // Idle glow
        const idle = defs.append('filter')
            .attr('id', `glow-${type}`)
            .attr('x', '-60%').attr('y', '-60%')
            .attr('width', '220%').attr('height', '220%');
        idle.append('feGaussianBlur')
            .attr('stdDeviation', '4')
            .attr('result', 'blur');
        idle.append('feFlood')
            .attr('flood-color', `rgba(${c.r}, ${c.g}, ${c.b}, 0.35)`)
            .attr('result', 'color');
        idle.append('feComposite')
            .attr('in', 'color').attr('in2', 'blur')
            .attr('operator', 'in').attr('result', 'glow');
        const m1 = idle.append('feMerge');
        m1.append('feMergeNode').attr('in', 'glow');
        m1.append('feMergeNode').attr('in', 'SourceGraphic');

        // Hover glow — bigger, brighter
        const hover = defs.append('filter')
            .attr('id', `glow-hover-${type}`)
            .attr('x', '-80%').attr('y', '-80%')
            .attr('width', '260%').attr('height', '260%');
        hover.append('feGaussianBlur')
            .attr('stdDeviation', '8')
            .attr('result', 'blur');
        hover.append('feFlood')
            .attr('flood-color', `rgba(${c.r}, ${c.g}, ${c.b}, 0.7)`)
            .attr('result', 'color');
        hover.append('feComposite')
            .attr('in', 'color').attr('in2', 'blur')
            .attr('operator', 'in').attr('result', 'glow');
        const m2 = hover.append('feMerge');
        m2.append('feMergeNode').attr('in', 'glow');
        m2.append('feMergeNode').attr('in', 'SourceGraphic');
    });

    const zoom = d3.zoom()
        .scaleExtent([0.2, 4])
        .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    g = svg.append('g');
    linkGroup = g.append('g').attr('class', 'links');
    nodeGroup = g.append('g').attr('class', 'nodes');

    fetchAndRender(focusEntityId);
    setupControls(focusEntityId);
}

function fetchAndRender(focusEntityId) {
    const minCo = document.getElementById('min-co-slider')?.value || 2;
    fetch(`/api/graph-data?min_events=3&min_co=${minCo}`)
        .then(r => r.json())
        .then(data => {
            currentData = data;
            render(data, focusEntityId);
        });
}

function render(data, focusEntityId) {
    const nodes = data.nodes.filter(n => activeTypes.has(n.type));
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = data.links.filter(l => nodeIds.has(l.source?.id ?? l.source) && nodeIds.has(l.target?.id ?? l.target));

    allNodes = nodes;
    allLinks = links;

    const nodeCountEl = document.getElementById('node-count');
    const linkCountEl = document.getElementById('link-count');
    if (nodeCountEl) nodeCountEl.textContent = nodes.length;
    if (linkCountEl) linkCountEl.textContent = links.length;

    const container = document.querySelector('.graph-canvas-wrapper');
    const width = container.clientWidth;
    const height = Math.max(container.clientHeight, 600);

    const maxEvents = d3.max(nodes, d => d.event_count) || 1;
    const rScale = d3.scaleSqrt().domain([1, maxEvents]).range([5, 26]);

    const sortedCounts = nodes.map(n => n.event_count).sort((a, b) => a - b);
    const labelThreshold = sortedCounts[Math.floor(sortedCounts.length * (1 - LABEL_PERCENTILE))] || 1;

    const maxWeight = d3.max(links, d => d.weight) || 1;

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(120).strength(0.2))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide().radius(d => rScale(d.event_count) + 10));

    // Links
    linkGroup.selectAll('line').remove();
    const link = linkGroup.selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', d => {
            const src = nodes.find(n => n.id === (d.source?.id ?? d.source));
            const t = src ? src.type : 'unknown';
            return `rgba(${rgb(t)}, 0.5)`;
        })
        .attr('stroke-opacity', 0.1)
        .attr('stroke-width', d => Math.max(0.5, d.weight / maxWeight * 2));

    // Nodes
    nodeGroup.selectAll('g').remove();
    const node = nodeGroup.selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'graph-node')
        .style('cursor', 'pointer')
        .call(d3.drag()
            .on('start', dragStart)
            .on('drag', dragging)
            .on('end', dragEnd));

    // Circle — idle: visible stroke + glow, subtle fill
    node.append('circle')
        .attr('r', d => rScale(d.event_count))
        .attr('fill', d => `rgba(${rgb(d.type)}, 0.08)`)
        .attr('stroke', d => `rgba(${rgb(d.type)}, 0.55)`)
        .attr('stroke-width', 1.5)
        .attr('filter', d => `url(#glow-${d.type in TYPE_BASE ? d.type : 'unknown'})`);

    // Thumbnails
    node.filter(d => d.thumbnail_url)
        .each(function(d) {
            const r = rScale(d.event_count);
            const clipId = `clip-${d.id}`;
            d3.select(this).append('clipPath')
                .attr('id', clipId)
                .append('circle')
                .attr('r', r - 1);
            d3.select(this).append('image')
                .attr('href', d.thumbnail_url)
                .attr('x', -r).attr('y', -r)
                .attr('width', r * 2).attr('height', r * 2)
                .attr('clip-path', `url(#${clipId})`)
                .attr('preserveAspectRatio', 'xMidYMid slice')
                .attr('opacity', 0.8);
        });

    // Labels
    node.append('text')
        .text(d => {
            const name = d.name.toUpperCase();
            return name.length > 14 ? name.slice(0, 13) + '\u2026' : name;
        })
        .attr('dy', d => rScale(d.event_count) + 13)
        .attr('text-anchor', 'middle')
        .attr('fill', d => `rgba(${rgb(d.type)}, 0.7)`)
        .attr('font-family', "'Share Tech Mono', monospace")
        .attr('font-size', '0.55rem')
        .attr('letter-spacing', '0.06em')
        .attr('opacity', d => d.event_count >= labelThreshold ? 0.7 : 0)
        .attr('pointer-events', 'none');

    // Hover — brighten stroke, fill with SAME type color, switch to stronger glow filter
    node.on('mouseenter', function(event, d) {
        const t = d.type in TYPE_BASE ? d.type : 'unknown';
        d3.select(this).select('circle')
            .attr('filter', `url(#glow-hover-${t})`)
            .transition().duration(150)
            .attr('fill', `rgba(${rgb(t)}, 0.25)`)
            .attr('stroke', `rgba(${rgb(t)}, 0.95)`)
            .attr('stroke-width', 2);
        d3.select(this).select('text')
            .transition().duration(150)
            .attr('opacity', 1)
            .attr('fill', `rgba(${rgb(t)}, 1)`);
        link.transition().duration(150)
            .attr('stroke-opacity', l =>
                (l.source.id === d.id || l.target.id === d.id) ? 0.35 : 0.03
            );
    });

    node.on('mouseleave', function(event, d) {
        const t = d.type in TYPE_BASE ? d.type : 'unknown';
        d3.select(this).select('circle')
            .attr('filter', `url(#glow-${t})`)
            .transition().duration(300)
            .attr('fill', `rgba(${rgb(t)}, 0.08)`)
            .attr('stroke', `rgba(${rgb(t)}, 0.55)`)
            .attr('stroke-width', 1.5);
        d3.select(this).select('text')
            .transition().duration(300)
            .attr('opacity', d.event_count >= labelThreshold ? 0.7 : 0)
            .attr('fill', `rgba(${rgb(t)}, 0.7)`);
        link.transition().duration(300)
            .attr('stroke-opacity', 0.1);
    });

    node.on('click', (event, d) => {
        event.stopPropagation();
        openPanel(d.id);
    });

    svg.on('click', () => {
        document.getElementById('entity-panel').classList.add('hidden');
    });

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    if (focusEntityId) {
        const focusNode = nodes.find(n => n.id === focusEntityId);
        if (focusNode) {
            setTimeout(() => {
                openPanel(focusEntityId);
                const transform = d3.zoomIdentity
                    .translate(width / 2, height / 2)
                    .scale(1.5)
                    .translate(-focusNode.x, -focusNode.y);
                svg.transition().duration(750).call(
                    d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => g.attr('transform', e.transform)).transform,
                    transform
                );
            }, 1500);
        }
    }
}

function openPanel(entityId) {
    const panel = document.getElementById('entity-panel');
    const content = document.getElementById('panel-content');
    content.innerHTML = '<div style="padding:1rem;color:var(--nx-text-dim);font-family:Share Tech Mono,monospace;font-size:0.8rem;">LOADING\u2026</div>';
    panel.classList.remove('hidden');

    fetch(`/api/entity-panel/${entityId}`)
        .then(r => r.text())
        .then(html => { content.innerHTML = html; });
}

function dragStart(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragging(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnd(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

function setupControls(focusEntityId) {
    document.querySelectorAll('.type-toggle').forEach(cb => {
        cb.addEventListener('change', () => {
            activeTypes = new Set(
                Array.from(document.querySelectorAll('.type-toggle:checked')).map(c => c.value)
            );
            if (currentData) render(currentData, null);
        });
    });

    const slider = document.getElementById('min-co-slider');
    const valueDisplay = document.getElementById('min-co-value');
    if (slider) {
        slider.addEventListener('input', () => {
            valueDisplay.textContent = slider.value;
        });
        slider.addEventListener('change', () => {
            fetchAndRender(null);
        });
    }
}
