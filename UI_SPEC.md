# COSMOS METRICS — UI Specification v7

## Design Philosophy

**Whimsical editorial** meets **data-dense dashboard**. Think high-end data journalism — warm, inviting, almost hand-crafted feeling. The dot clusters should feel like they were scattered by hand onto parchment paper. Typography is confident and editorial: large serif titles, clean sans-serif labels, monospace for data. The overall feel is a beautiful artifact you'd want to frame, not a dark sci-fi terminal.

BUT — this is still the Cosmos Metrics system. The sun/batter concept, the energy beams, the strike zone framing — all still live. They just exist in this warmer, more organic world. The sun glows golden against cream. The beams are watercolor-like washes of green and red. The nebulae are soft pastel clouds.

### Core Aesthetic
- **Warm cream/parchment background** (#F8F6F1 or similar) — NOT white, NOT dark. Warm and inviting like aged paper
- **Organic dot clusters** — Dots feel hand-scattered, slightly irregular spacing, natural cloud formations. NOT grid-aligned, NOT too uniform
- **Editorial typography**: Large confident serif for titles (DM Serif Display / Playfair / similar), clean sans for labels (Inter), monospace for data values (JetBrains Mono / IBM Plex Mono)
- **Muted, earthy palette** — The cluster colors are rich but not neon. Think terracotta, sage, dusty purple, warm grey, muted coral. Colors that feel like they belong on a ceramics shelf
- **Lots of breathing room** — Generous whitespace. Let the dot clusters breathe. Don't crowd the canvas
- **Subtle borders** — Thin, warm grey (#d4d0c8), not harsh. Slight rounded corners on pills/tags only (4-6px), not on main containers
- **The warmth of paper, the density of data**

---

## 1. The Scene (Main Canvas)

### Background
- Warm cream/parchment (#F8F6F1) — the entire page, not just the canvas
- Can have a very subtle paper texture or grain (CSS noise overlay at 2-3% opacity) for tactile quality
- No dark mode — this is a daytime, warm, inviting aesthetic

### The Archetype Map
- The main canvas takes up the majority of the viewport (between the top bar and bottom dock)
- Clusters are positioned using their PCA coordinates, but the layout should feel organic — like a hand-drawn map of constellations
- NO strike zone frame in this version — the clusters float freely on the warm background, arranged by their natural PCA positions (LHP left, RHP right, high whiff up top)
- Instead of a rigid frame, consider very faint, thin connecting lines between adjacent clusters — like a constellation map — or just let them float

### Cluster Layout
- LHP clusters naturally fall to the left side of the canvas
- RHP clusters naturally fall to the right side
- The vertical axis loosely corresponds to whiff rate / pitch style
- Each cluster is its own organic cloud of dots with clear separation between clouds
- Labels float inside or beside each cluster

---

## 2. Archetype Clusters (Dot Clouds)

19 clusters (10 RHP + 9 LHP), each a **soft organic cloud** of dots.

### Pitcher Dots
- ~6,000 dots total, each = one pitcher-season
- Small circles (3-5px radius), positioned at real PCA coordinates
- Color matches cluster palette — muted, earthy tones
- **Opacity gradient within each cluster**: Dots near the centroid are darker/more opaque (70-90%), dots at the edges are lighter/more transparent (20-40%). This creates a natural density gradient — a soft cloud that's darker in the middle and feathers out at the edges
- Dots should feel slightly irregular — like they were stippled by hand. Tiny random size variation (±1px) adds to the organic feel
- On hover, individual dots pop to full opacity with a subtle scale-up

### Cluster Labels
- Positioned at or near each cluster's centroid
- Archetype name in **ALL CAPS**, clean sans-serif (Inter 600), 11-13px
- Bordered pill/tag: thin border matching cluster color, slight rounded corners (4px), semi-transparent background
- Like the reference images: "FLAMETHROWER", "CRAFTY LEFTY", "SINK/SLIDE", "HIGH SPIN", "CHANGEUP"
- Pitcher count can appear on hover or as smaller text below

### Star Pitcher Highlights
- 3 per cluster — slightly larger dots (6-8px), fully opaque, with a subtle ring/outline
- Name labels appear on hover (small, clean, positioned to avoid overlap)

### Cluster Feel
- The clusters should overlap slightly at their edges where archetypes are similar — dots from neighboring clusters interweave at the boundaries, creating those beautiful organic transitions visible in your reference images
- This is NOT a rigid grid or force-directed graph — it's a PCA scatter that naturally groups

---

## 3. The Batter — Search & Activate

When a user searches and selects a batter, the dot clusters become the backdrop for showing that batter's matchup data.

### Batter Entry
- When selected, the batter doesn't appear as a sun on the canvas (that's for the 3D version)
- Instead, **energy lines radiate FROM a batter card/node on the side** into the cluster map
- OR: Each cluster's dots subtly re-color/re-size based on the batter's performance against that archetype:
  - Clusters the batter crushes: dots shift toward **green tones**, become slightly larger
  - Clusters the batter struggles against: dots shift toward **red/warm tones**, become slightly smaller
  - This creates a beautiful organic heatmap effect across the whole canvas

### Batter Card (Sidebar or Overlay)
- When a batter is active, a card appears (left side or floating)
- Shows: Name (large serif), career wOBA, total PA, sun tier class
- The card is warm, editorial, with the batter's "sun tier" indicated by a colored accent (gold for supernova, muted brown for brown dwarf)

### Energy Lines (Optional Layer)
- Thin lines from the batter card to each cluster centroid
- Color: green (crush) → amber (average) → red (struggle) — using muted, warm versions of these colors that fit the palette
- Thickness: PA count
- Style: Slightly curved, with a hand-drawn/slightly wobbly quality (subtle noise on the path) to match the organic aesthetic
- These can be toggled on/off — the heatmap-style cluster recoloring might be enough

---

## 4. Color Palette

### Background & Chrome
- Page background: #F8F6F1 (warm cream)
- Card/panel backgrounds: #FFFFFF or #FEFDFB (slightly warmer white)
- Borders: #d4d0c8 (warm grey)
- Text primary: #2D2D2D (near-black, warm)
- Text secondary: #888 (warm grey)
- Text tertiary: #bbb

### Cluster Colors (Muted, Earthy — inspired by your references)
```
FLAMETHROWER:      #c1292e → muted coral/terracotta
CRAFTY LEFTY:      #6b7f5e → sage/olive green
SINK/SLIDE:        #6b7b8d → slate/warm grey-blue
HIGH SPIN:         #c4a855 → muted gold/ochre
CHANGEUP:          #8b6b9e → dusty purple/mauve
CONTROL ARTIST:    #d4762c → warm burnt orange
CUTTER CARVER:     #7b6b5e → warm taupe/brown
SPLITTER:          #4a7c6f → muted teal
GAS & SNAP:        #c44e52 → warm red
```

These are starting points — adjust so each cluster is distinct but they all feel cohesive, like they belong in the same watercolor palette. Nothing should feel neon or digital.

### Energy Line Colors (Warm Versions)
- Crush: muted sage green (#6b8f5e)
- Above avg: light warm green (#8fad7f)
- Average: warm amber (#c4a055)
- Below avg: warm orange (#d4864c)
- Struggle: muted terracotta (#b84a4a)

### Sun Tier Accent Colors
- Supernova: warm gold (#c4a040)
- Red Giant: burnt orange (#c4763c)
- Main Sequence: warm yellow (#c4a855)
- Dwarf Star: warm grey (#8b8575)
- Brown Dwarf: cool taupe (#6b6560)

---

## 5. Top Bar

### Layout
```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ARCHETYPE ATLAS    Search batter...    2024 Season ▾    wOBA ▾         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Title**: "ARCHETYPE ATLAS" or "COSMOS METRICS" — large, confident serif (DM Serif Display), dark warm (#2D2D2D), left-aligned
- **Search**: Clean input with warm grey border, placeholder text in light grey. Underline-style border (like the second reference) or full border (like the first)
- **Season selector**: "2024 Season ▾" dropdown, clean serif or sans
- **Stat selector**: "wOBA ▾" dropdown
- Minimal, clean, lots of breathing room. NOT dense — editorial and airy
- Thin warm grey bottom border separating from the canvas

---

## 6. Bottom Dock

### Layout
A low-profile dock (120-180px) below the canvas. Clean, information-dense but not cluttered.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ● FLAMETHROWER  ● CRAFTY LEFTY  ● SINK/SLIDE  ● HIGH SPIN  ● CHANGEUP │
│                                                                          │
│  Min PA: ●━━━━━━━━━━━━░░ 100           SAVED: Soto, Judge, Trout       │
│                                                                          │
│  ↻ RESET VIEW                                                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Row 1: Cluster Legend / Toggles
- Horizontal list of all 19 archetypes
- Each: colored dot + name in caps
- Click to toggle visibility
- Can be grouped: RHP clusters, then separator, then LHP clusters
- Like the reference images — clean, horizontal, well-spaced

### Row 2: Controls + Saved Searches
- **Min PA slider**: Clean range slider with monospace value label
- **Saved Searches**: Previously searched batters, shown as text chips or a compact list. Click to reload. Persisted in localStorage
- **Reset View button**: Clean, bordered, "RESET VIEW" or "↻ RESET ZOOM"

### Row 3 (expandable): Matchup Insights
- When a batter is active, this row can expand to show:
  - **Best archetype** (highest wOBA) with green indicator
  - **Worst archetype** (lowest wOBA) with red indicator
  - **vs RHP / vs LHP split** line
  - **Player prop edges** (K prop, H prop, HR prop) — placeholder for future live data
- This keeps the bottom dock lean when no batter is active, then rich with data when one is

### Style
- Background: matches page background (#F8F6F1) or very slightly different (#F0EDE7)
- Thin top border (#d4d0c8)
- Clean sans-serif labels, monospace for numbers
- Generous horizontal spacing between items

---

## 7. Stat Panel (Right Sidebar)

When you click a cluster label or an energy line, a panel slides in from the right.

- Width: ~340px
- Background: white or warm white (#FEFDFB)
- Thin left border (#d4d0c8)
- **Header**: Batter name (serif, large) + "vs" + cluster name (cluster color, bold)
- **Stat grid**: PA, BA, OBP, SLG, HR, K%, BB% — in clean cards with warm backgrounds
- **Year-by-year bars**: Colored bars (green→amber→red based on wOBA), monospace labels
- **Year table**: Clean, minimal borders, warm alternating row colors
- Close: small "×" in top-right

---

## 8. Animations & Interactions

1. **Initial load** — Dots scatter in from random positions to their PCA positions (like they're settling into place), staggered by cluster. Takes ~1.5s total. Satisfying.
2. **Batter search** — Dropdown slides down smoothly. On select, clusters subtly re-color over 500ms
3. **Energy lines** — If shown, draw from batter card to centroids with a smooth ease-out, staggered
4. **Cluster hover** — All dots in that cluster pop to full opacity, label becomes prominent
5. **Dot hover** — Individual dot scales up slightly, tooltip shows pitcher name + year + velo + whiff rate
6. **Cluster toggle** — Dots fade out/in (300ms, ease)
7. **Zoom** — Mouse scroll zooms in/out of the canvas. Dots scale appropriately. Pan with drag.
8. **Reset** — Smooth zoom/pan back to default view
9. **Saved search click** — Subtle flash/pulse as batter data reloads

---

## 9. Responsive Considerations

- The dot canvas should resize fluidly
- On narrower screens, the bottom dock modules can stack vertically
- The stat panel can become a bottom sheet on mobile
- Cluster labels might need to hide at very zoomed-out views to avoid crowding

---

## 10. Data Files

4 JSONs in `frontend/src/data/`:

- **clusters.json** — 19 archetypes (names, colors, hand, PCA positions, pitcher counts, examples, centroid stats)
- **pitcher_seasons.json** — ~6K dots (PCA positions, stats)
- **batters.json** — ~2,700 batters (IDs, names)
- **hitter_vs_cluster.json** — ~108K matchup rows (PA, BA, OBP, SLG, wOBA, K%, BB%, etc.)

**Data quirk**: `batter` field in hitter_vs_cluster = string name (match on `batter_name`)
**Missing**: Batter handedness (can add for positioning), schedule data (for live props)

---

## 11. What Makes This Special

The magic is in the **contrast between the organic, hand-crafted visual feel and the rigorous data underneath**. The dots look like they were scattered by an artist, but they're positioned by K-Means clustering on 22 statistical features across 10 years of Statcast data. The colors feel like a warm palette, but they encode real archetype identities. The energy lines look like watercolor washes, but they represent precise wOBA differentials.

It should feel like opening a beautiful atlas — warm, inviting, something you want to explore and spend time with. Not a dashboard. An **atlas**.
