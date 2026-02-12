import React, { useState } from 'react'

export default function ControlBar({ clusters, visibleClusters, onToggleCluster }) {
  const [expanded, setExpanded] = useState(false)
  const clusterIds = Object.keys(clusters)
  const rhpIds = clusterIds.filter(id => clusters[id]?.hand === 'RHP').sort()
  const lhpIds = clusterIds.filter(id => clusters[id]?.hand === 'LHP').sort()

  const showAll = () => clusterIds.forEach(id => { if (!visibleClusters.has(id)) onToggleCluster(id) })
  const hideAll = () => clusterIds.forEach(id => { if (visibleClusters.has(id)) onToggleCluster(id) })
  const rhpOnly = () => clusterIds.forEach(id => {
    const want = clusters[id]?.hand === 'RHP'
    if (want !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const lhpOnly = () => clusterIds.forEach(id => {
    const want = clusters[id]?.hand === 'LHP'
    if (want !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const spOnly = () => clusterIds.forEach(id => {
    const isSP = clusters[id]?.is_sp > 0.5
    if (isSP !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const rpOnly = () => clusterIds.forEach(id => {
    const isRP = clusters[id]?.is_sp <= 0.5
    if (isRP !== visibleClusters.has(id)) onToggleCluster(id)
  })

  const visibleCount = clusterIds.filter(id => visibleClusters.has(id)).length

  return (
    <div style={{
      background: 'rgba(16,20,32,0.95)', borderTop: '1px solid #252a3a',
      backdropFilter: 'blur(8px)',
    }}>
      {/* Collapsed bar — always visible */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '6px 16px',
        cursor: 'pointer',
      }} onClick={() => setExpanded(!expanded)}>
        <span style={{ fontSize: 10, color: '#555', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Archetypes
        </span>
        <span style={{ fontSize: 10, color: '#444' }}>
          {visibleCount}/{clusterIds.length}
        </span>

        <div style={{ display: 'flex', gap: 4, marginLeft: 4 }}>
          <button onClick={e => { e.stopPropagation(); showAll() }} style={btnStyle}>All</button>
          <button onClick={e => { e.stopPropagation(); hideAll() }} style={btnStyle}>None</button>
          <button onClick={e => { e.stopPropagation(); rhpOnly() }} style={btnStyle}>RHP</button>
          <button onClick={e => { e.stopPropagation(); lhpOnly() }} style={btnStyle}>LHP</button>
          <button onClick={e => { e.stopPropagation(); spOnly() }} style={btnStyle}>SP</button>
          <button onClick={e => { e.stopPropagation(); rpOnly() }} style={btnStyle}>RP</button>
        </div>

        <span style={{ marginLeft: 'auto', fontSize: 12, color: '#555' }}>
          {expanded ? '▼' : '▲'}
        </span>
      </div>

      {/* Expanded panel */}
      {expanded && (
        <div style={{
          display: 'flex', gap: 24, padding: '0 16px 12px',
          overflowX: 'auto',
        }}>
          {/* RHP column */}
          <div style={{ minWidth: 200 }}>
            <div style={sectionHeader}>
              Right-Handed ({rhpIds.length})
            </div>
            {rhpIds.map(id => (
              <ClusterRow
                key={id} id={id} cluster={clusters[id]}
                visible={visibleClusters.has(id)}
                onToggle={onToggleCluster}
              />
            ))}
          </div>

          {/* LHP column */}
          <div style={{ minWidth: 200 }}>
            <div style={sectionHeader}>
              Left-Handed ({lhpIds.length})
            </div>
            {lhpIds.map(id => (
              <ClusterRow
                key={id} id={id} cluster={clusters[id]}
                visible={visibleClusters.has(id)}
                onToggle={onToggleCluster}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ClusterRow({ id, cluster, visible, onToggle }) {
  const name = cluster?.short_name || id
  const count = cluster?.pitcher_count || 0
  return (
    <div
      onClick={() => onToggle(id)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px',
        cursor: 'pointer', borderRadius: 6,
        opacity: visible ? 1 : 0.35, transition: 'opacity 0.2s',
        background: visible ? 'rgba(255,255,255,0.03)' : 'transparent',
      }}
    >
      <div style={{
        width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
        background: cluster?.color || '#888',
        boxShadow: visible ? `0 0 6px ${cluster?.color}44` : 'none',
      }} />
      <span style={{
        flex: 1, fontSize: 13, fontWeight: 500,
        color: visible ? cluster?.color || '#ccc' : '#555',
      }}>
        {name}
      </span>
      <span style={{ fontSize: 11, color: '#555', fontWeight: 500, minWidth: 30, textAlign: 'right' }}>
        {count}
      </span>
    </div>
  )
}

const sectionHeader = {
  fontSize: 10, color: '#666', fontWeight: 700, textTransform: 'uppercase',
  letterSpacing: '0.08em', padding: '8px 8px 4px', borderBottom: '1px solid #252a3a',
  marginBottom: 4,
}

const btnStyle = {
  background: 'rgba(255,255,255,0.06)', color: '#888', border: '1px solid #333',
  borderRadius: 6, padding: '2px 8px', fontSize: 10, cursor: 'pointer',
}
