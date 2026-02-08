/**
 * Data loader for Archetype Atlas.
 * Imports JSON files directly (Vite resolves these at build time).
 */

import clustersData from './clusters.json'
import battersData from './batters.json'
import hitterVsClusterData from './hitter_vs_cluster.json'
import pitcherSeasonsData from './pitcher_seasons.json'

export async function loadData() {
  console.log('Loaded pipeline data:', {
    clusters: Object.keys(clustersData).length,
    batters: battersData.length,
    hitterVsCluster: hitterVsClusterData.length,
    pitcherSeasons: pitcherSeasonsData.length,
  })
  return {
    clusters: clustersData,
    batters: battersData,
    hitterVsCluster: hitterVsClusterData,
    pitcherSeasons: pitcherSeasonsData,
  }
}
