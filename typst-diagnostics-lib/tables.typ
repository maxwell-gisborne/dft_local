#let diagnostic-table(data) = {
  let columns = data.headers
  table(
    columns: columns.len(),
    inset: 5pt,
    stroke: 0.5pt,
    ..columns.map(h => [*#h*]),
    ..data.rows.flatten().map(cell => [#str(cell)]),
  )
}
