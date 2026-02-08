import React, { useRef, useEffect, useState, useMemo } from 'react'
import * as d3 from 'd3'

// Color scale for stat-based line coloring: red (bad) -> amber (mid) -> green (good)
const STAT_RANGES = {
  wOBA:          { min: 0.220, mid: 0.320, max: 0.420, invert: false },
  BA:            { min: 0.180, mid: 0.260, max: 0.340, invert: false },
  SLG:           { min: 0.280, mid: 0.420, max: 0.560, invert: false },
  K_pct:         { min: 0.10,  mid: 0.22,  max: 0.35,  invert: true },
  BB_pct:        { min: 0.04,  mid: 0.09,  max: 0.16,  invert: false },
  whiff_rate_vs: { min: 0.15,  mid: 0.25,  max: 0.38,  invert: true },
}

function getStatColor(stat, value) {
  const range = STAT_RANGES[stat] || STAT_RANGES.wOBA
  const { min, max, invert } = range
  let t = (value - min) / (max - min)
  t = Math.max(0, Math.min(1, t))
  if (invert) t = 1 - t
  if (t < 0.5) {
    return d3.interpolateRgb('#c1292e', '#d4a053')(t * 2)
  }
  return d3.interpolateRgb('#d4a053', '#2a7d5f')(( t - 0.5) * 2)
}

function getStatThickness(stat, value) {
  const range = STAT_RANGES[stat] || STAT_RANGES.wOBA
  const { min, max, invert } = range
  let t = (value - min) / (max - min)
  t = Math.max(0, Math.min(1, t))
  if (invert) t = 1 - t
  return 1.5 + t * 6
}

function formatStat(stat, value) {
  if (stat === 'K_pct' || stat === 'BB_pct' || stat === 'whiff_rate_vs') {
    return (value * 100).toFixed(1) + '%'
  }
  return value.toFixed(3)
}


export default function GraphCanvas({
  clusters,
  visibleClusters,
  activeBatters,
  hitterData,
  pitcherSeasons,
  selectedStat,
  minPA,
  showPitcherDots,
  onLineClick,
}) {
  const svgRef = useRef(null)
  const containerRef = useRef(null)
  const zoomGroupRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState(null)
  const zoomRef = useRef(null)

  // Resize observer
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setDimensions({ width, height })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Compute PCA coordinate bounds from pitcher data
  const pcaBounds = useMemo(() => {
    const pts = pitcherSeasons.filter(p => p.pca_x != null && p.pca_y != null)
    if (pts.length === 0) {
      // Fallback: generate from cluster centroid positions
      const cids = Object.keys(clusters)
      const xs = cids.map(id => clusters[id]?.pca_x || 0)
      const ys = cids.map(id => clusters[id]?.pca_y || 0)
      return {
        xMin: Math.min(...xs) - 2, xMax: Math.max(...xs) + 2,
        yMin: Math.min(...ys) - 2, yMax: Math.max(...ys) + 2,
      }
    }
    const xs = pts.map(p => p.pca_x)
    const ys = pts.map(p => p.pca_y)
    const pad = 0.5
    return {
      xMin: Math.min(...xs) - pad, xMax: Math.max(...xs) + pad,
      yMin: Math.min(...ys) - pad, yMax: Math.max(...ys) + pad,
    }
  }, [pitcherSeasons, clusters])

  // Scales: PCA coords -> pixel coords
  const scales = useMemo(() => {
    const { width, height } = dimensions
    const margin = 60
    const xScale = d3.scaleLinear()
      .domain([pcaBounds.xMin, pcaBounds.xMax])
      .range([margin, width - margin])
    const yScale = d3.scaleLinear()
      .domain([pcaBounds.yMin, pcaBounds.yMax])
      .range([height - margin, margin]) // flip y
    return { xScale, yScale }
  }, [dimensions, pcaBounds])

  // Cluster centroid pixel positions
  const clusterNodes = useMemo(() => {
    return Object.keys(clusters)
      .filter(id => visibleClusters.has(id))
      .map(id => {
        const c = clusters[id]
        return {
          id,
          px: scales.xScale(c.pca_x || 0),
          py: scales.yScale(c.pca_y || 0),
          color: c.color,
          name: c.short_name,
          fullName: c.full_name,
          pitcherCount: c.pitcher_count || 0,
          centroid: c.centroid || {},
        }
      })
  }, [clusters, visibleClusters, scales])

  // Pitcher dot pixel positions
  const pitcherDots = useMemo(() => {
    if (!showPitcherDots) return []
    return pitcherSeasons
      .filter(p => p.pca_x != null && visibleClusters.has(String(p.cluster)))
      .map(p => ({
        key: `${p.pitcher}-${p.game_year}`,
        px: scales.xScale(p.pca_x),
        py: scales.yScale(p.pca_y),
        color: clusters[String(p.cluster)]?.color || '#999',
        name: p.player_name || 'Unknown',
        year: p.game_year,
        cluster: p.cluster,
        whiff: p.whiff_rate,
        velo: p.avg_velo_FF,
      }))
  }, [pitcherSeasons, scales, showPitcherDots, visibleClusters, clusters])

  // Batter lines: from a batter position to each cluster centroid
  const { lines, batterNodes } = useMemo(() => {
    const { width, height } = dimensions
    const linesArr = []
    const batNodes = []

    // Place batters spaced evenly at the bottom-center area of the canvas
    activeBatters.forEach((batter, bi) => {
      const total = activeBatters.length
      const cx = width / 2
      const cy = height / 2
      const spread = Math.min(260, total * 50)
      const bx = cx + (bi - (total - 1) / 2) * (spread / Math.max(total, 1))
      const by = cy

      batNodes.push({ id: batter.batter, name: batter.batter_name, px: bx, py: by })

      for (const cn of clusterNodes) {
        const row = hitterData.find(
          r => r.batter === batter.batter && String(r.cluster) === cn.id
        )
        const pa = row?.PA || 0
        const statVal = row?.[selectedStat] || 0

        linesArr.push({
          key: `${batter.batter}-${cn.id}`,
          batter,
          clusterId: cn.id,
          x1: bx, y1: by,
          x2: cn.px, y2: cn.py,
          statVal, pa,
          color: pa >= minPA ? getStatColor(selectedStat, statVal) : '#ccc',
          thickness: pa >= minPA ? getStatThickness(selectedStat, statVal) : 1,
          dashed: pa < minPA,
          clusterName: cn.name,
          batterName: batter.batter_name,
        })
      }
    })

    return { lines: linesArr, batterNodes: batNodes }
  }, [activeBatters, clusterNodes, hitterData, selectedStat, minPA, dimensions])

  // --- D3 Rendering ---
  useEffect(() => {
    const svg = d3.select(svgRef.current)
    const { width, height } = dimensions
    svg.attr('width', width).attr('height', height)

    // Setup zoom
    let zoomG = svg.select('g.zoom-group')
    if (zoomG.empty()) {
      zoomG = svg.append('g').attr('class', 'zoom-group')
      zoomGroupRef.current = zoomG

      const zoom = d3.zoom()
        .scaleExtent([0.3, 8])
        .on('zoom', (event) => {
          zoomG.attr('transform', event.transform)
        })
      svg.call(zoom)
      zoomRef.current = zoom

      // Defs
      const defs = svg.append('defs')
      const shadow = defs.append('filter').attr('id', 'textBg')
      shadow.append('feFlood').attr('flood-color', '#F8F6F1').attr('flood-opacity', 0.85)
      shadow.append('feComposite').attr('in', 'SourceGraphic')
    }

    const g = zoomG

    // === LAYER 1: Pitcher dots ===
    const dotsLayer = g.selectAll('g.dots-layer').data([0])
    const dotsG = dotsLayer.enter().append('g').attr('class', 'dots-layer').merge(dotsLayer)

    const dots = dotsG.selectAll('circle.dot').data(
      showPitcherDots ? pitcherDots : [],
      d => d.key
    )

    dots.exit().transition().duration(300).attr('r', 0).remove()

    const dotsEnter = dots.enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => d.px)
      .attr('cy', d => d.py)
      .attr('r', 0)
      .attr('fill', d => d.color)
      .attr('opacity', 0.4)
      .attr('stroke', 'none')
      .style('cursor', 'pointer')
      .on('mouseenter', (event, d) => {
        d3.select(event.target).attr('r', 6).attr('opacity', 0.9)
        setTooltip({
          x: event.clientX, y: event.clientY,
          content: `${d.name} (${d.year})\n${d.velo ? d.velo.toFixed(1) + ' mph' : ''} | Whiff: ${d.whiff ? (d.whiff * 100).toFixed(1) + '%' : '?'}`,
        })
      })
      .on('mouseleave', (event) => {
        d3.select(event.target).attr('r', 3.5).attr('opacity', 0.4)
        setTooltip(null)
      })

    dotsEnter.transition().duration(600).attr('r', 3.5)

    dots.transition().duration(400)
      .attr('cx', d => d.px).attr('cy', d => d.py).attr('fill', d => d.color)

    // === LAYER 2: Lines (bezier curves) ===
    const linesLayer = g.selectAll('g.lines-layer').data([0])
    const linesG = linesLayer.enter().append('g').attr('class', 'lines-layer').merge(linesLayer)

    const pathData = lines.map(d => {
      // Quadratic bezier: control point offset perpendicular to the line
      const mx = (d.x1 + d.x2) / 2
      const my = (d.y1 + d.y2) / 2
      const dx = d.x2 - d.x1
      const dy = d.y2 - d.y1
      const len = Math.sqrt(dx * dx + dy * dy) || 1
      // Offset perpendicular (alternating direction based on cluster index)
      const idx = parseInt(d.clusterId) || 0
      const sign = idx % 2 === 0 ? 1 : -1
      const offset = Math.min(40, len * 0.08) * sign
      const cx = mx + (-dy / len) * offset
      const cy = my + (dx / len) * offset
      return { ...d, pathStr: `M ${d.x1},${d.y1} Q ${cx},${cy} ${d.x2},${d.y2}` }
    })

    const pathSel = linesG.selectAll('path.stat-line').data(pathData, d => d.key)

    pathSel.exit().transition().duration(300).attr('opacity', 0).remove()

    const pathEnter = pathSel.enter()
      .append('path')
      .attr('class', 'stat-line')
      .attr('d', d => `M ${d.x1},${d.y1} Q ${d.x1},${d.y1} ${d.x1},${d.y1}`)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', d => d.thickness)
      .attr('stroke-dasharray', d => d.dashed ? '6,4' : 'none')
      .attr('opacity', 0)
      .style('cursor', 'pointer')
      .on('mouseenter', (event, d) => {
        d3.select(event.target).attr('stroke-width', d.thickness + 2)
        setTooltip({
          x: event.clientX, y: event.clientY,
          content: `${d.batterName} vs ${d.clusterName}\n${selectedStat}: ${formatStat(selectedStat, d.statVal)} (${d.pa} PA)`,
        })
      })
      .on('mouseleave', (event, d) => {
        d3.select(event.target).attr('stroke-width', d.thickness)
        setTooltip(null)
      })
      .on('click', (event, d) => onLineClick(d.batter, d.clusterId))

    pathEnter.each(function(d, i) {
      d3.select(this)
        .transition().delay(i * 80).duration(700).ease(d3.easeCubicOut)
        .attr('d', d.pathStr)
        .attr('opacity', d.dashed ? 0.25 : 0.7)
    })

    pathSel.transition().duration(500)
      .attr('d', d => d.pathStr)
      .attr('stroke', d => d.color)
      .attr('stroke-width', d => d.thickness)
      .attr('opacity', d => d.dashed ? 0.25 : 0.7)

    // === LAYER 3: Cluster labels ===
    const labelsLayer = g.selectAll('g.labels-layer').data([0])
    const labelsG = labelsLayer.enter().append('g').attr('class', 'labels-layer').merge(labelsLayer)

    const labelSel = labelsG.selectAll('g.cluster-label').data(clusterNodes, d => d.id)

    labelSel.exit().transition().duration(300).attr('opacity', 0).remove()

    const labelEnter = labelSel.enter()
      .append('g')
      .attr('class', 'cluster-label')
      .attr('transform', d => `translate(${d.px},${d.py})`)
      .attr('opacity', 0)
      .style('cursor', 'pointer')
      .on('mouseenter', (event, d) => {
        const c = d.centroid
        setTooltip({
          x: event.clientX, y: event.clientY,
          content: `${d.fullName}\n${d.pitcherCount} pitcher-seasons\nVelo: ${c.avg_velo_FF?.toFixed(1) || '?'} mph | Whiff: ${c.whiff_rate ? (c.whiff_rate * 100).toFixed(1) + '%' : '?'}\nArm Angle: ${c.arm_angle?.toFixed(1) || '?'}deg`,
        })
      })
      .on('mouseleave', () => setTooltip(null))

    // Background pill
    labelEnter.append('rect')
      .attr('rx', 12).attr('ry', 12)
      .attr('fill', d => d.color)
      .attr('opacity', 0.12)
      .attr('x', -50).attr('y', -12)
      .attr('width', 100).attr('height', 24)

    // Small colored dot at centroid
    labelEnter.append('circle')
      .attr('r', 5)
      .attr('fill', d => d.color)
      .attr('opacity', 0.8)

    // Text
    labelEnter.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', -10)
      .attr('fill', d => d.color)
      .attr('font-size', 11)
      .attr('font-weight', 700)
      .attr('letter-spacing', '0.05em')
      .style('text-transform', 'uppercase')
      .style('paint-order', 'stroke')
      .style('stroke', '#F8F6F1')
      .style('stroke-width', '3px')
      .text(d => d.name)

    labelEnter.transition().duration(500).attr('opacity', 1)

    // Auto-size the background pill to text
    labelsG.selectAll('g.cluster-label').each(function() {
      const g = d3.select(this)
      const text = g.select('text')
      const bbox = text.node()?.getBBox()
      if (bbox) {
        g.select('rect')
          .attr('x', bbox.x - 8)
          .attr('y', bbox.y - 4)
          .attr('width', bbox.width + 16)
          .attr('height', bbox.height + 8)
      }
    })

    labelSel.transition().duration(500)
      .attr('transform', d => `translate(${d.px},${d.py})`)
      .attr('opacity', 1)

    // === LAYER 4: Batter nodes ===
    const batLayer = g.selectAll('g.batter-layer').data([0])
    const batG = batLayer.enter().append('g').attr('class', 'batter-layer').merge(batLayer)

    const batSel = batG.selectAll('g.batter-node').data(batterNodes, d => d.id)

    batSel.exit().transition().duration(300).attr('opacity', 0).remove()

    const batEnter = batSel.enter()
      .append('g')
      .attr('class', 'batter-node')
      .attr('transform', d => `translate(${d.px},${d.py})`)
      .attr('opacity', 0)

    // Diamond shape for batter
    const ds = 12
    batEnter.append('polygon')
      .attr('points', `0,${-ds} ${ds},0 0,${ds} ${-ds},0`)
      .attr('fill', '#2D2D2D')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)

    batEnter.append('text')
      .attr('y', -18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#2D2D2D')
      .attr('font-size', 13)
      .attr('font-weight', 700)
      .style('paint-order', 'stroke')
      .style('stroke', '#F8F6F1')
      .style('stroke-width', '3px')
      .text(d => d.name)

    batEnter.transition().duration(400).attr('opacity', 1)

    batSel.transition().duration(400)
      .attr('transform', d => `translate(${d.px},${d.py})`)

  }, [clusterNodes, batterNodes, lines, pitcherDots, showPitcherDots, dimensions, selectedStat])

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <svg ref={svgRef} style={{ width: '100%', height: '100%', background: '#F8F6F1' }} />

      {tooltip && (
        <div style={{
          position: 'fixed',
          left: tooltip.x + 14,
          top: tooltip.y - 10,
          background: 'rgba(255,255,252,0.96)',
          border: '1px solid #d4d0c8',
          borderRadius: 8,
          padding: '8px 12px',
          fontSize: 12,
          color: '#2D2D2D',
          pointerEvents: 'none',
          zIndex: 100,
          maxWidth: 300,
          whiteSpace: 'pre-line',
          boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          lineHeight: 1.5,
        }}>
          {tooltip.content}
        </div>
      )}
    </div>
  )
}
