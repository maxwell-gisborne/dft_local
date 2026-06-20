#import "@preview/cetz:0.3.4"
#import "@preview/lilaq:0.4.0" as lq
#import "tables.typ": diagnostic-table

#let line-graph(data) = {
  let series = data.series.map(s => (
    label: s.name,
    x: s.points.map(p => p.x),
    y: s.points.map(p => p.y),
  ))
  lq.diagram(
    width: 11cm,
    height: 6cm,
    xaxis: (label: data.x_label),
    yaxis: (label: data.y_label),
    ..series.map(s => lq.plot(s.x, s.y, label: s.label)),
  )
}

#let unsupported-view(data) = block[
  *Unsupported static view:* #data.title \
  #data.description
]


#let dos-idos-view(data) = [
  #line-graph(data.dos_graph)

  #line-graph(data.idos_graph)
]


#let region-surface-summary(data) = [
  #diagnostic-table(data.summary_table)

  #diagnostic-table(data.band_table)
]
