#import "@preview/zero:0.6.1": num, zi

#let render-quantity(q) = {
  let unit = if "unit_symbol" in q and q.unit_symbol != "" {
    q.unit_symbol
  } else if "unit" in q and q.unit != "" {
    q.unit
  } else {
    ""
  }

  if unit == "" or unit == "None" {
    num(str(q.value))
  } else {
    let declared-unit = zi.declare(unit)
    declared-unit[str(q.value)]
  }
}

#let display-value(x) = {
  if type(x) == dictionary and "kind" in x and x.kind == "quantity" {
    render-quantity(x)
  } else if type(x) == bool {
    if x { "true" } else { "false" }
  } else if x == none {
    "none"
  } else if type(x) == int or type(x) == float {
    num(str(x))
  } else {
    str(x)
  }
}

#let diagnostic-table(data) = {
  let columns = data.headers
  table(
    columns: columns.len(),
    inset: 5pt,
    stroke: 0.5pt,
    ..columns.map(h => [*#h*]),
    ..data.rows.flatten().map(cell => [#display-value(cell)]),
  )
}
