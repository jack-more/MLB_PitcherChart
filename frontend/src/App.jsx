import React, { useState, useEffect, useCallback } from 'react'
import GalaxyScene from './components/GalaxyScene.jsx'
import BatterSearch from './components/BatterSearch.jsx'
import ControlBar from './components/ControlBar.jsx'
import StatPanel from './components/StatPanel.jsx'
import { loadData } from './data/loader.js'

const STAT_OPTIONS = [
  { value: 'wOBA', label: 'wOBA' },
  { value: 'BA', label: 'BA' },
  { value: 'SLG', label: 'SLG' },
  { value: 'K_pct', label: 'K%' },
  { value: 'BB_pct', label: 'BB%' },
  { value: 'whiff_rate_vs', label: 'Whiff%' },
]

const selectStyle = {
  background: 'rgba(255,255,255,0.06)', color: '#e0e0e0', border: '1px solid #333',
  borderRadius: 6, padding: '6px 10px', fontSize: 13, outline: 'none',
}

export default function App() {
  const [clusters, setClusters] = useState({})
  const [batters, setBatters] = useState([])
  const [hitterVsCluster, setHitterVsCluster] = useState([])
  const [pitcherSeasons, setPitcherSeasons] = useState([])
  const [dataLoaded, setDataLoaded] = useState(false)

  const [activeBatters, setActiveBatters] = useState([])
  const [selectedStat, setSelectedStat] = useState('wOBA')
  const [visibleClusters, setVisibleClusters] = useState(new Set())
  const [selectedYear, setSelectedYear] = useState('all')
  const [minPA, setMinPA] = useState(10)
  const [selectedLine, setSelectedLine] = useState(null)
  const [showPitcherDots, setShowPitcherDots] = useState(true)

  useEffect(() => {
    loadData().then(data => {
      setClusters(data.clusters)
      setBatters(data.batters)
      setHitterVsCluster(data.hitterVsCluster)
      setPitcherSeasons(data.pitcherSeasons || [])
      setVisibleClusters(new Set(Object.keys(data.clusters)))
      setDataLoaded(true)
    })
  }, [])

  const addBatter = useCallback((batter) => {
    setActiveBatters(prev => {
      if (prev.find(b => b.batter === batter.batter)) return prev
      if (prev.length >= 5) return prev
      return [...prev, batter]
    })
  }, [])

  const removeBatter = useCallback((batterId) => {
    setActiveBatters(prev => prev.filter(b => b.batter !== batterId))
    setSelectedLine(null)
  }, [])

  const toggleCluster = useCallback((clusterId) => {
    setVisibleClusters(prev => {
      const next = new Set(prev)
      next.has(clusterId) ? next.delete(clusterId) : next.add(clusterId)
      return next
    })
  }, [])

  const onLineClick = useCallback((batterObj, clusterId) => {
    setSelectedLine({ batter: batterObj, cluster: clusterId })
  }, [])

  const closePanel = useCallback(() => setSelectedLine(null), [])

  const filteredHitterData = hitterVsCluster.filter(row => {
    if (selectedYear !== 'all' && row.game_year !== parseInt(selectedYear)) return false
    return true
  })

  const aggregatedHitterData = selectedYear === 'all'
    ? aggregateAcrossYears(filteredHitterData)
    : filteredHitterData

  const years = [...new Set(hitterVsCluster.map(r => r.game_year))].sort()

  if (!dataLoaded) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', color: '#555', fontSize: '1.1rem',
        fontFamily: "'Inter', sans-serif", background: '#0a0e1a',
        flexDirection: 'column', gap: 12,
      }}>
        <div style={{
          width: 40, height: 40, border: '3px solid #333',
          borderTop: '3px solid #2a9d5f', borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <span>Loading Archetype Universe...</span>
      </div>
    )
  }

  return (
    <div style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', background: '#0a0e1a' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 14, padding: '8px 16px',
        background: 'rgba(16,20,32,0.9)', borderBottom: '1px solid #252a3a', zIndex: 10,
        backdropFilter: 'blur(8px)',
      }}>
        <h1 style={{
          fontFamily: "'DM Serif Display', serif", fontSize: 17, fontWeight: 400,
          color: '#e0e0e0', margin: 0, letterSpacing: '-0.02em', whiteSpace: 'nowrap',
        }}>
          Archetype Universe
        </h1>
        <span style={{ fontSize: 10, color: '#555' }}>
          {Object.keys(clusters).length} archetypes · {pitcherSeasons.length.toLocaleString()} pitcher-seasons · 2015-2025
        </span>

        <div style={{ width: 1, height: 20, background: '#333', margin: '0 2px' }} />

        <BatterSearch batters={batters} onSelect={addBatter} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <label style={{ fontSize: 11, color: '#666' }}>Stat:</label>
          <select value={selectedStat} onChange={e => setSelectedStat(e.target.value)} style={selectStyle}>
            {STAT_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <label style={{ fontSize: 11, color: '#666' }}>Year:</label>
          <select value={selectedYear} onChange={e => setSelectedYear(e.target.value)} style={selectStyle}>
            <option value="all">All Years</option>
            {years.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <label style={{ fontSize: 11, color: '#666' }}>Min PA:</label>
          <input
            type="range" min={1} max={100} value={minPA}
            onChange={e => setMinPA(parseInt(e.target.value))}
            style={{ width: 60, accentColor: '#2a9d5f' }}
          />
          <span style={{ fontSize: 11, color: '#888', minWidth: 20 }}>{minPA}</span>
        </div>

        <label style={{ fontSize: 11, color: '#666', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
          <input type="checkbox" checked={showPitcherDots} onChange={e => setShowPitcherDots(e.target.checked)}
            style={{ accentColor: '#2a9d5f' }}
          />
          Dots
        </label>

        {/* Active batter chips */}
        <div style={{ display: 'flex', gap: 6, marginLeft: 'auto', flexWrap: 'wrap' }}>
          {activeBatters.map(b => (
            <div key={b.batter} style={{
              background: 'rgba(255,255,255,0.06)', border: '1px solid #333', borderRadius: 16,
              padding: '3px 10px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 5,
              color: '#e0e0e0',
            }}>
              {b.batter_name}
              <button onClick={() => removeBatter(b.batter)} style={{
                background: 'none', border: 'none', color: '#555', cursor: 'pointer',
                fontSize: 13, lineHeight: 1, padding: 0,
              }}>x</button>
            </div>
          ))}
        </div>
      </div>

      {/* Main 3D canvas */}
      <div style={{ flex: 1, position: 'relative' }}>
        <GalaxyScene
          clusters={clusters}
          visibleClusters={visibleClusters}
          activeBatters={activeBatters}
          hitterData={aggregatedHitterData}
          pitcherSeasons={pitcherSeasons}
          selectedStat={selectedStat}
          minPA={minPA}
          showPitcherDots={showPitcherDots}
          onLineClick={onLineClick}
        />

        {selectedLine && (
          <StatPanel
            batter={selectedLine.batter}
            clusterId={selectedLine.cluster}
            clusterName={clusters[selectedLine.cluster]?.short_name || `Cluster ${selectedLine.cluster}`}
            clusterColor={clusters[selectedLine.cluster]?.color || '#888'}
            hitterData={hitterVsCluster}
            selectedYear={selectedYear}
            onClose={closePanel}
          />
        )}
      </div>

      {/* Bottom control bar */}
      <ControlBar clusters={clusters} visibleClusters={visibleClusters} onToggleCluster={toggleCluster} />
    </div>
  )
}


function aggregateAcrossYears(rows) {
  const grouped = {}
  for (const row of rows) {
    const k = `${row.batter}__${row.cluster}`
    if (!grouped[k]) {
      grouped[k] = {
        batter: row.batter, batter_name: row.batter_name, cluster: row.cluster,
        PA: 0, AB: 0, H: 0, HR: 0, BB: 0, K: 0, HBP: 0,
        singles: 0, doubles: 0, triples: 0,
        woba_sum: 0, woba_denom_sum: 0, pitches_seen: 0,
      }
    }
    const g = grouped[k]
    g.PA += row.PA || 0; g.AB += row.AB || 0; g.H += row.H || 0
    g.HR += row.HR || 0; g.BB += row.BB || 0; g.K += row.K || 0
    g.HBP += row.HBP || 0; g.singles += row.singles || 0
    g.doubles += row.doubles || 0; g.triples += row.triples || 0
    g.woba_sum += (row.wOBA || 0) * (row.PA || 0)
    g.woba_denom_sum += row.PA || 0
    g.pitches_seen += row.pitches_seen || 0
  }
  return Object.values(grouped).map(g => ({
    ...g,
    BA: g.AB > 0 ? g.H / g.AB : 0,
    OBP: g.PA > 0 ? (g.H + g.BB + g.HBP) / g.PA : 0,
    SLG: g.AB > 0 ? (g.singles + 2 * g.doubles + 3 * g.triples + 4 * g.HR) / g.AB : 0,
    K_pct: g.PA > 0 ? g.K / g.PA : 0,
    BB_pct: g.PA > 0 ? g.BB / g.PA : 0,
    wOBA: g.woba_denom_sum > 0 ? g.woba_sum / g.woba_denom_sum : 0,
    whiff_rate_vs: 0,
  }))
}
