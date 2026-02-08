import React, { useMemo } from 'react'

export default function StatPanel({ batter, clusterId, clusterName, clusterColor, hitterData, selectedYear, onClose }) {
  const rows = useMemo(() => {
    return hitterData.filter(
      r => r.batter === batter.batter && String(r.cluster) === String(clusterId)
    ).sort((a, b) => a.game_year - b.game_year)
  }, [hitterData, batter.batter, clusterId])

  const totals = useMemo(() => {
    const t = { PA: 0, AB: 0, H: 0, HR: 0, BB: 0, K: 0, HBP: 0, singles: 0, doubles: 0, triples: 0 }
    for (const r of rows) {
      t.PA += r.PA || 0; t.AB += r.AB || 0; t.H += r.H || 0
      t.HR += r.HR || 0; t.BB += r.BB || 0; t.K += r.K || 0
      t.HBP += r.HBP || 0; t.singles += r.singles || 0
      t.doubles += r.doubles || 0; t.triples += r.triples || 0
    }
    return {
      ...t,
      BA: t.AB > 0 ? t.H / t.AB : 0,
      OBP: t.PA > 0 ? (t.H + t.BB + t.HBP) / t.PA : 0,
      SLG: t.AB > 0 ? (t.singles + 2 * t.doubles + 3 * t.triples + 4 * t.HR) / t.AB : 0,
      K_pct: t.PA > 0 ? t.K / t.PA : 0,
      BB_pct: t.PA > 0 ? t.BB / t.PA : 0,
    }
  }, [rows])

  return (
    <div style={{
      position: 'absolute', top: 0, right: 0, width: 340, height: '100%',
      background: 'rgba(16,20,32,0.95)', borderLeft: '1px solid #252a3a', padding: 20,
      overflowY: 'auto', zIndex: 20,
      animation: 'slideIn 0.3s ease',
      backdropFilter: 'blur(12px)',
      boxShadow: '-4px 0 20px rgba(0,0,0,0.3)',
    }}>
      <style>{`@keyframes slideIn { from { transform: translateX(100%); } to { transform: translateX(0); } }`}</style>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 16, color: '#e0e0e0' }}>{batter.batter_name}</h3>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: clusterColor, fontWeight: 600 }}>
            vs {clusterName}
          </p>
        </div>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', color: '#555', fontSize: 20, cursor: 'pointer',
        }}>x</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 20 }}>
        <StatBox label="PA" value={totals.PA} color={clusterColor} />
        <StatBox label="BA" value={totals.BA.toFixed(3)} color={clusterColor} />
        <StatBox label="OBP" value={totals.OBP.toFixed(3)} color={clusterColor} />
        <StatBox label="SLG" value={totals.SLG.toFixed(3)} color={clusterColor} />
        <StatBox label="HR" value={totals.HR} color={clusterColor} />
        <StatBox label="K%" value={(totals.K_pct * 100).toFixed(1) + '%'} color={clusterColor} />
        <StatBox label="BB%" value={(totals.BB_pct * 100).toFixed(1) + '%'} color={clusterColor} />
        <StatBox label="H" value={totals.H} color={clusterColor} />
        <StatBox label="2B" value={totals.doubles} color={clusterColor} />
      </div>

      {rows.length > 1 && (
        <>
          <h4 style={{ fontSize: 11, color: '#555', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Year-by-Year
          </h4>
          <div style={{ marginBottom: 16 }}>
            {rows.map(r => {
              const woba = r.wOBA || 0
              const barWidth = Math.min(100, Math.max(5, (woba / 0.450) * 100))
              const barColor = woba > 0.370 ? '#2a9d5f' : woba > 0.310 ? '#d4a053' : '#c1292e'
              return (
                <div key={r.game_year} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: '#666', width: 32 }}>{r.game_year}</span>
                  <div style={{ flex: 1, height: 12, background: '#1a1e2e', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{
                      width: `${barWidth}%`, height: '100%', background: barColor,
                      borderRadius: 4, transition: 'width 0.5s ease',
                    }} />
                  </div>
                  <span style={{ fontSize: 11, color: '#aaa', width: 36, textAlign: 'right' }}>{woba.toFixed(3)}</span>
                  <span style={{ fontSize: 10, color: '#555', width: 36, textAlign: 'right' }}>{r.PA} PA</span>
                </div>
              )
            })}
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: '#666', borderBottom: '1px solid #252a3a' }}>
                  {['Year','PA','BA','OBP','SLG','HR','K%'].map(h => (
                    <th key={h} style={{ padding: '4px 5px', textAlign: 'right', fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.game_year} style={{ borderBottom: '1px solid #1a1e2e' }}>
                    <td style={td}>{r.game_year}</td>
                    <td style={td}>{r.PA}</td>
                    <td style={td}>{(r.BA || 0).toFixed(3)}</td>
                    <td style={td}>{(r.OBP || 0).toFixed(3)}</td>
                    <td style={td}>{(r.SLG || 0).toFixed(3)}</td>
                    <td style={td}>{r.HR || 0}</td>
                    <td style={td}>{((r.K_pct || 0) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {rows.length === 0 && (
        <p style={{ color: '#555', fontSize: 13 }}>No data available for this matchup.</p>
      )}
    </div>
  )
}

function StatBox({ label, value, color }) {
  return (
    <div style={{ background: '#1a1e2e', borderRadius: 8, padding: '8px 10px', textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: '#666', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
      <div style={{ fontSize: 16, color: '#e0e0e0', fontWeight: 600 }}>{value}</div>
    </div>
  )
}

const td = { padding: '4px 5px', textAlign: 'right', color: '#aaa' }
