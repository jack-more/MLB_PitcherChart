import React from 'react'

export default function ControlBar({ clusters, visibleClusters, onToggleCluster }) {
  const clusterIds = Object.keys(clusters)
  const rhpIds = clusterIds.filter(id => clusters[id]?.hand === 'RHP')
  const lhpIds = clusterIds.filter(id => clusters[id]?.hand === 'LHP')

  const showAll = () => clusterIds.forEach(id => { if (!visibleClusters.has(id)) onToggleCluster(id) })
  const rhpOnly = () => clusterIds.forEach(id => {
    const want = clusters[id]?.hand === 'RHP'
    if (want !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const lhpOnly = () => clusterIds.forEach(id => {
    const want = clusters[id]?.hand === 'LHP'
    if (want !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const spOnly = () => clusterIds.forEach(id => {
    const isSP = clusters[id]?.centroid?.is_sp > 0.5
    if (isSP !== visibleClusters.has(id)) onToggleCluster(id)
  })
  const rpOnly = () => clusterIds.forEach(id => {
    const isRP = clusters[id]?.centroid?.is_sp <= 0.5
    if (isRP !== visibleClusters.has(id)) onToggleCluster(id)
  })

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '8px 16px',
      background: 'rgba(16,20,32,0.9)', borderTop: '1px solid #252a3a', flexWrap: 'wrap',
      backdropFilter: 'blur(8px)',
    }}>
      <span style={{ fontSize: 10, color: '#555', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Archetypes
      </span>

      <button onClick={showAll} style={btnStyle}>All</button>
      <button onClick={rhpOnly} style={btnStyle}>RHP</button>
      <button onClick={lhpOnly} style={btnStyle}>LHP</button>
      <button onClick={spOnly} style={btnStyle}>SP</button>
      <button onClick={rpOnly} style={btnStyle}>RP</button>

      <div style={{ width: 1, height: 16, background: '#333', margin: '0 2px' }} />

      {/* RHP clusters */}
      {rhpIds.length > 0 && (
        <>
          <span style={{ fontSize: 9, color: '#e07840', fontWeight: 600 }}>RHP</span>
          {rhpIds.map(id => <ClusterToggle key={id} id={id} cluster={clusters[id]} visible={visibleClusters.has(id)} onToggle={onToggleCluster} />)}
          <div style={{ width: 1, height: 16, background: '#333', margin: '0 2px' }} />
        </>
      )}

      {/* LHP clusters */}
      {lhpIds.length > 0 && (
        <>
          <span style={{ fontSize: 9, color: '#457b9d', fontWeight: 600 }}>LHP</span>
          {lhpIds.map(id => <ClusterToggle key={id} id={id} cluster={clusters[id]} visible={visibleClusters.has(id)} onToggle={onToggleCluster} />)}
        </>
      )}
    </div>
  )
}

function ClusterToggle({ id, cluster, visible, onToggle }) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer',
      fontSize: 11, opacity: visible ? 1 : 0.3, transition: 'opacity 0.2s',
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: '50%', background: cluster?.color || '#888',
        border: visible ? `2px solid ${cluster?.color}` : '2px solid #444',
        boxShadow: visible ? `0 0 4px ${cluster?.color}44` : 'none',
      }} />
      <input type="checkbox" checked={visible} onChange={() => onToggle(id)} style={{ display: 'none' }} />
      <span style={{ color: visible ? '#ccc' : '#555' }}>{cluster?.short_name}</span>
    </label>
  )
}

const btnStyle = {
  background: 'rgba(255,255,255,0.06)', color: '#888', border: '1px solid #333',
  borderRadius: 6, padding: '3px 10px', fontSize: 11, cursor: 'pointer',
}
