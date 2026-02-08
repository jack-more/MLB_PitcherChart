import React, { useState, useRef, useEffect, useMemo } from 'react'
import Fuse from 'fuse.js'

export default function BatterSearch({ batters, onSelect }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isOpen, setIsOpen] = useState(false)
  const [highlightIdx, setHighlightIdx] = useState(0)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)

  const fuse = useMemo(() => new Fuse(batters, {
    keys: ['batter_name'], threshold: 0.3, distance: 100,
  }), [batters])

  useEffect(() => {
    if (query.length >= 2) {
      setResults(fuse.search(query).slice(0, 8).map(r => r.item))
      setIsOpen(true)
      setHighlightIdx(0)
    } else {
      setResults([])
      setIsOpen(false)
    }
  }, [query, fuse])

  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
          inputRef.current && !inputRef.current.contains(e.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const select = (batter) => {
    onSelect(batter)
    setQuery('')
    setIsOpen(false)
    inputRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlightIdx(p => Math.min(p + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHighlightIdx(p => Math.max(p - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); if (results[highlightIdx]) select(results[highlightIdx]) }
    else if (e.key === 'Escape') setIsOpen(false)
  }

  return (
    <div style={{ position: 'relative', width: 260 }}>
      <input
        ref={inputRef} type="text"
        placeholder="Search batters..."
        value={query} onChange={e => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => results.length > 0 && setIsOpen(true)}
        style={{
          width: '100%', background: 'rgba(255,255,255,0.06)', color: '#e0e0e0',
          border: '1px solid #333', borderRadius: 8, padding: '7px 12px',
          fontSize: 13, outline: 'none',
        }}
      />
      {isOpen && results.length > 0 && (
        <div ref={dropdownRef} style={{
          position: 'absolute', top: '100%', left: 0, width: '100%',
          background: '#1a1e2e', border: '1px solid #333', borderRadius: 8,
          marginTop: 4, zIndex: 50, maxHeight: 280, overflowY: 'auto',
          boxShadow: '0 8px 24px rgba(0,0,0,0.40)',
        }}>
          {results.map((b, i) => (
            <div key={b.batter} onClick={() => select(b)}
              onMouseEnter={() => setHighlightIdx(i)}
              style={{
                padding: '8px 12px', cursor: 'pointer',
                background: i === highlightIdx ? 'rgba(255,255,255,0.08)' : 'transparent',
                borderBottom: i < results.length - 1 ? '1px solid #252a3a' : 'none',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
              <span style={{ fontSize: 13, color: '#e0e0e0' }}>{b.batter_name}</span>
              <span style={{ fontSize: 11, color: '#666' }}>{b.total_PA?.toLocaleString()} PA</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
