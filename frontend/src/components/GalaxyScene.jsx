import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { getStatColor, getStatT, getPAThickness } from '../helpers/statColors.js'

const SCALE = 1.0  // PCA coords scale factor

/** Instanced dots for all pitcher seasons (~6K) */
function PitcherDots({ pitcherSeasons, clusters, visibleClusters, onHover }) {
  const meshRef = useRef()
  const tempObj = useMemo(() => new THREE.Object3D(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])

  const visible = useMemo(() => {
    return pitcherSeasons.filter(p =>
      p.pca_x != null && visibleClusters.has(String(p.cluster))
    )
  }, [pitcherSeasons, visibleClusters])

  const colors = useMemo(() => {
    const arr = new Float32Array(visible.length * 3)
    visible.forEach((p, i) => {
      const c = clusters[String(p.cluster)]
      tempColor.set(c?.color || '#888')
      arr[i * 3] = tempColor.r
      arr[i * 3 + 1] = tempColor.g
      arr[i * 3 + 2] = tempColor.b
    })
    return arr
  }, [visible, clusters])

  useEffect(() => {
    if (!meshRef.current) return
    visible.forEach((p, i) => {
      tempObj.position.set(p.pca_x * SCALE, p.pca_y * SCALE, (p.pca_z || 0) * SCALE)
      tempObj.updateMatrix()
      meshRef.current.setMatrixAt(i, tempObj.matrix)
    })
    meshRef.current.instanceMatrix.needsUpdate = true
    meshRef.current.count = visible.length
  }, [visible, tempObj])

  useEffect(() => {
    if (!meshRef.current) return
    const colorAttr = meshRef.current.geometry.getAttribute('color')
    if (!colorAttr || colorAttr.array.length !== colors.length) {
      meshRef.current.geometry.setAttribute(
        'color',
        new THREE.InstancedBufferAttribute(colors, 3)
      )
    } else {
      colorAttr.array.set(colors)
      colorAttr.needsUpdate = true
    }
  }, [colors])

  if (visible.length === 0) return null

  return (
    <instancedMesh
      ref={meshRef}
      args={[null, null, visible.length]}
      frustumCulled={false}
    >
      <sphereGeometry args={[0.04, 8, 6]} />
      <meshBasicMaterial vertexColors transparent opacity={0.45} />
    </instancedMesh>
  )
}

/** Star pitcher highlights (larger, brighter dots with labels) */
function StarPitchers({ clusters, visibleClusters, pitcherSeasons }) {
  const stars = useMemo(() => {
    const result = []
    for (const [cid, cluster] of Object.entries(clusters)) {
      if (!visibleClusters.has(cid)) continue
      const examples = cluster.example_pitchers || []
      examples.forEach(nameStr => {
        // Find this pitcher in the seasons data
        const namePart = nameStr.replace(/\s*\(\d{4}\)/, '').trim()
        const yearMatch = nameStr.match(/\((\d{4})\)/)
        const year = yearMatch ? parseInt(yearMatch[1]) : null
        const match = pitcherSeasons.find(p => {
          const nameMatch = p.player_name?.toLowerCase().includes(namePart.toLowerCase().split(',')[0]) ||
            p.player_name?.toLowerCase() === namePart.toLowerCase()
          return nameMatch && (!year || p.game_year === year)
        })
        if (match && match.pca_x != null) {
          result.push({
            key: `${match.pitcher}-${match.game_year}`,
            x: match.pca_x * SCALE,
            y: match.pca_y * SCALE,
            z: (match.pca_z || 0) * SCALE,
            name: nameStr,
            color: cluster.color,
          })
        }
      })
    }
    return result
  }, [clusters, visibleClusters, pitcherSeasons])

  return stars.map(s => (
    <group key={s.key} position={[s.x, s.y, s.z]}>
      <mesh>
        <sphereGeometry args={[0.08, 12, 8]} />
        <meshBasicMaterial color={s.color} transparent opacity={0.9} />
      </mesh>
      <pointLight color={s.color} intensity={0.3} distance={1.5} />
      <Html distanceFactor={15} style={{ pointerEvents: 'none' }}>
        <div style={{
          color: '#fff',
          fontSize: 9,
          fontWeight: 600,
          whiteSpace: 'nowrap',
          textShadow: '0 0 6px rgba(0,0,0,0.8)',
          transform: 'translateY(-12px)',
          opacity: 0.85,
        }}>
          {s.name}
        </div>
      </Html>
    </group>
  ))
}

/** Cluster centroid labels floating in 3D */
function ClusterLabels({ clusters, visibleClusters, onClusterHover }) {
  return Object.entries(clusters)
    .filter(([id]) => visibleClusters.has(id))
    .map(([id, c]) => {
      const x = (c.pca_x || 0) * SCALE
      const y = (c.pca_y || 0) * SCALE
      const z = (c.pca_z || 0) * SCALE
      return (
        <group key={id} position={[x, y, z]}>
          {/* Centroid glow sphere */}
          <mesh>
            <sphereGeometry args={[0.12, 16, 12]} />
            <meshBasicMaterial color={c.color} transparent opacity={0.3} />
          </mesh>
          <pointLight color={c.color} intensity={0.5} distance={3} />
          <Html
            distanceFactor={12}
            style={{ pointerEvents: 'auto' }}
            onPointerEnter={() => onClusterHover?.(id)}
            onPointerLeave={() => onClusterHover?.(null)}
          >
            <div style={{
              padding: '4px 10px',
              borderRadius: 12,
              background: `${c.color}22`,
              border: `1px solid ${c.color}55`,
              backdropFilter: 'blur(4px)',
              textAlign: 'center',
              whiteSpace: 'nowrap',
              cursor: 'pointer',
              transform: 'translateY(-20px)',
            }}>
              <div style={{
                color: c.color,
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}>
                {c.short_name}
              </div>
              <div style={{ color: '#888', fontSize: 9 }}>
                {c.pitcher_count} pitchers · {c.hand}
              </div>
            </div>
          </Html>
        </group>
      )
    })
}

/** Axis lines at the origin — faint quadrant markers */
function AxisLines() {
  const lineColor = '#ffffff'
  const lineOpacity = 0.06
  const len = 12
  return (
    <group>
      {/* X axis */}
      <Line
        points={[[-len, 0, 0], [len, 0, 0]]}
        color={lineColor}
        transparent
        opacity={lineOpacity}
        lineWidth={0.5}
        dashed
        dashScale={2}
        dashSize={0.3}
        gapSize={0.3}
      />
      {/* Y axis */}
      <Line
        points={[[0, -len, 0], [0, len, 0]]}
        color={lineColor}
        transparent
        opacity={lineOpacity}
        lineWidth={0.5}
        dashed
        dashScale={2}
        dashSize={0.3}
        gapSize={0.3}
      />
      {/* Side labels */}
      <Html position={[8, 0, 0]} distanceFactor={20} style={{ pointerEvents: 'none' }}>
        <div style={{ color: '#ffffff22', fontSize: 10, fontWeight: 600 }}>RHP</div>
      </Html>
      <Html position={[-8, 0, 0]} distanceFactor={20} style={{ pointerEvents: 'none' }}>
        <div style={{ color: '#ffffff22', fontSize: 10, fontWeight: 600 }}>LHP</div>
      </Html>
    </group>
  )
}

/** Background stars for the space atmosphere */
function BackgroundStars() {
  const positions = useMemo(() => {
    const count = 1500
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 80
      arr[i * 3 + 1] = (Math.random() - 0.5) * 80
      arr[i * 3 + 2] = (Math.random() - 0.5) * 80
    }
    return arr
  }, [])

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={1500}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial color="#ffffff" size={0.08} transparent opacity={0.12} sizeAttenuation />
    </points>
  )
}

/** Energy beam lines from batter to cluster centroids */
function PerformanceLines({
  activeBatters,
  clusters,
  visibleClusters,
  hitterData,
  selectedStat,
  minPA,
  onLineClick,
  onLineHover,
  expandedCluster,
  pitcherSeasons,
}) {
  const lines = useMemo(() => {
    const result = []
    // Place each batter at bottom-center of the scene
    activeBatters.forEach((batter, bi) => {
      const total = activeBatters.length
      const spread = Math.min(3, total * 0.8)
      const bx = (bi - (total - 1) / 2) * (spread / Math.max(total, 1))
      const by = -5
      const bz = 0

      for (const [cid, cluster] of Object.entries(clusters)) {
        if (!visibleClusters.has(cid)) continue
        const row = hitterData.find(
          r => r.batter === batter.batter && String(r.cluster) === cid
        )
        const pa = row?.PA || 0
        const statVal = row?.[selectedStat] || 0
        const meetsPA = pa >= minPA
        const color = meetsPA ? getStatColor(selectedStat, statVal) : '#333333'

        const cx = (cluster.pca_x || 0) * SCALE
        const cy = (cluster.pca_y || 0) * SCALE
        const cz = (cluster.pca_z || 0) * SCALE

        result.push({
          key: `${batter.batter}-${cid}`,
          batter,
          clusterId: cid,
          points: [[bx, by, bz], [cx, cy, cz]],
          color,
          lineWidth: meetsPA ? 1 + getStatT(selectedStat, statVal) * 4 : 0.5,
          opacity: meetsPA ? 0.6 + getStatT(selectedStat, statVal) * 0.3 : 0.1,
          dashed: !meetsPA,
          pa,
          statVal,
        })
      }
    })
    return result
  }, [activeBatters, clusters, visibleClusters, hitterData, selectedStat, minPA])

  // Sub-links for expanded cluster
  const subLinks = useMemo(() => {
    if (!expandedCluster) return []
    const result = []
    const { batterId, clusterId } = expandedCluster
    const cluster = clusters[clusterId]
    if (!cluster) return result

    // Find pitchers in this cluster that the batter has faced
    const clusterPitchers = pitcherSeasons.filter(
      p => String(p.cluster) === clusterId
    )

    // Pick up to 8 pitchers (just show nearest to centroid for now)
    const shown = clusterPitchers.slice(0, 8)
    const cx = (cluster.pca_x || 0) * SCALE
    const cy = (cluster.pca_y || 0) * SCALE
    const cz = (cluster.pca_z || 0) * SCALE

    shown.forEach(p => {
      result.push({
        key: `sub-${batterId}-${p.pitcher}-${p.game_year}`,
        points: [[cx, cy, cz], [p.pca_x * SCALE, p.pca_y * SCALE, (p.pca_z || 0) * SCALE]],
        color: cluster.color,
        lineWidth: 0.8,
        opacity: 0.3,
        name: p.player_name,
      })
    })
    return result
  }, [expandedCluster, clusters, pitcherSeasons])

  return (
    <group>
      {lines.map(l => (
        <group key={l.key}>
          <Line
            points={l.points}
            color={l.color}
            lineWidth={l.lineWidth}
            transparent
            opacity={l.opacity}
            dashed={l.dashed}
            dashScale={l.dashed ? 4 : 1}
            dashSize={l.dashed ? 0.2 : 1}
            gapSize={l.dashed ? 0.2 : 0}
          />
          {/* Clickable / hoverable midpoint */}
          <mesh
            position={[
              (l.points[0][0] + l.points[1][0]) / 2,
              (l.points[0][1] + l.points[1][1]) / 2,
              (l.points[0][2] + l.points[1][2]) / 2,
            ]}
            onClick={(e) => {
              e.stopPropagation()
              onLineClick?.(l.batter, l.clusterId)
            }}
            onPointerEnter={(e) => {
              e.stopPropagation()
              document.body.style.cursor = 'pointer'
              onLineHover?.(l)
            }}
            onPointerLeave={(e) => {
              e.stopPropagation()
              document.body.style.cursor = 'auto'
              onLineHover?.(null)
            }}
          >
            <sphereGeometry args={[0.15, 6, 4]} />
            <meshBasicMaterial transparent opacity={0} />
          </mesh>
        </group>
      ))}
      {/* Sub-links */}
      {subLinks.map(sl => (
        <Line
          key={sl.key}
          points={sl.points}
          color={sl.color}
          lineWidth={sl.lineWidth}
          transparent
          opacity={sl.opacity}
        />
      ))}
    </group>
  )
}

/** Batter diamond markers */
function BatterNodes({ activeBatters }) {
  if (activeBatters.length === 0) return null

  return activeBatters.map((batter, bi) => {
    const total = activeBatters.length
    const spread = Math.min(3, total * 0.8)
    const bx = (bi - (total - 1) / 2) * (spread / Math.max(total, 1))
    return (
      <group key={batter.batter} position={[bx, -5, 0]}>
        <mesh rotation={[0, 0, Math.PI / 4]}>
          <boxGeometry args={[0.2, 0.2, 0.2]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
        <pointLight color="#ffffff" intensity={0.6} distance={2} />
        <Html distanceFactor={10} style={{ pointerEvents: 'none' }}>
          <div style={{
            color: '#fff',
            fontSize: 12,
            fontWeight: 700,
            whiteSpace: 'nowrap',
            textShadow: '0 0 8px rgba(0,0,0,0.9)',
            transform: 'translateY(14px)',
            textAlign: 'center',
          }}>
            {batter.batter_name}
          </div>
        </Html>
      </group>
    )
  })
}

/** Camera auto-framing on load */
function CameraSetup() {
  const { camera } = useThree()
  useEffect(() => {
    camera.position.set(0, -2, 14)
    camera.lookAt(0, 0, 0)
  }, [camera])
  return null
}

export default function GalaxyScene({
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
  const [hoveredLine, setHoveredLine] = useState(null)
  const [expandedCluster, setExpandedCluster] = useState(null)
  const [hoveredCluster, setHoveredCluster] = useState(null)

  const handleLineClick = useCallback((batter, clusterId) => {
    onLineClick?.(batter, clusterId)
    setExpandedCluster(prev => {
      if (prev?.batterId === batter.batter && prev?.clusterId === clusterId) return null
      return { batterId: batter.batter, clusterId }
    })
  }, [onLineClick])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, -2, 14], fov: 55, near: 0.1, far: 200 }}
        style={{ background: 'linear-gradient(180deg, #0a0e1a 0%, #141824 100%)' }}
        gl={{ antialias: true }}
      >
        <CameraSetup />
        <ambientLight intensity={0.15} />
        <BackgroundStars />
        <AxisLines />

        {showPitcherDots && (
          <PitcherDots
            pitcherSeasons={pitcherSeasons}
            clusters={clusters}
            visibleClusters={visibleClusters}
          />
        )}

        <StarPitchers
          clusters={clusters}
          visibleClusters={visibleClusters}
          pitcherSeasons={pitcherSeasons}
        />

        <ClusterLabels
          clusters={clusters}
          visibleClusters={visibleClusters}
          onClusterHover={setHoveredCluster}
        />

        <PerformanceLines
          activeBatters={activeBatters}
          clusters={clusters}
          visibleClusters={visibleClusters}
          hitterData={hitterData}
          selectedStat={selectedStat}
          minPA={minPA}
          onLineClick={handleLineClick}
          onLineHover={setHoveredLine}
          expandedCluster={expandedCluster}
          pitcherSeasons={pitcherSeasons}
        />

        <BatterNodes activeBatters={activeBatters} />

        <OrbitControls
          enableDamping
          dampingFactor={0.12}
          rotateSpeed={0.5}
          zoomSpeed={0.8}
          minDistance={3}
          maxDistance={40}
        />
      </Canvas>

      {/* Hover tooltip (rendered as HTML overlay) */}
      {hoveredLine && (
        <div style={{
          position: 'absolute',
          left: '50%',
          bottom: 60,
          transform: 'translateX(-50%)',
          background: 'rgba(20,24,36,0.92)',
          border: '1px solid #333',
          borderRadius: 10,
          padding: '8px 14px',
          fontSize: 12,
          color: '#e0e0e0',
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        }}>
          <span style={{ fontWeight: 600 }}>{hoveredLine.batter?.batter_name}</span>
          {' vs '}
          <span style={{ color: hoveredLine.color, fontWeight: 600 }}>
            {clusters[hoveredLine.clusterId]?.short_name}
          </span>
          {' — '}
          <span style={{ color: hoveredLine.color }}>
            {selectedStat}: {
              (selectedStat === 'K_pct' || selectedStat === 'BB_pct' || selectedStat === 'whiff_rate_vs')
                ? (hoveredLine.statVal * 100).toFixed(1) + '%'
                : hoveredLine.statVal.toFixed(3)
            }
          </span>
          <span style={{ color: '#666', marginLeft: 8 }}>({hoveredLine.pa} PA)</span>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: 12,
        right: 16,
        background: 'rgba(20,24,36,0.75)',
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 10,
        color: '#888',
        backdropFilter: 'blur(4px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
          <div style={{
            width: 60, height: 6, borderRadius: 3,
            background: 'linear-gradient(90deg, #c1292e, #e07840, #d4a053, #7bc47f, #2a9d5f)',
          }} />
          <span>Struggle → Crush</span>
        </div>
        <div>Line thickness = sample size (PA)</div>
        <div>Bright dots = archetype star pitchers</div>
      </div>
    </div>
  )
}
