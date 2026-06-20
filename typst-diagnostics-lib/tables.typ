#import "@preview/zero:0.6.1": num, zi

#let compact-number(x) = {
  let ax = calc.abs(x)
  if ax != 0 and (ax < 1e-4 or ax >= 1e5) {
    str(x, base: 10)
  } else {
    str(calc.round(x * 1000000) / 1000000)
  }
}

#let render-quantity(q) = {
  let unit = if "unit_symbol" in q and q.unit_symbol != "" {
    q.unit_symbol
  } else if "unit" in q and q.unit != "" {
    q.unit
  } else {
    ""
  }

  let value = compact-number(q.value)
  if unit == "" or unit == "None" {
    num(value)
  } else {
    let declared-unit = zi.declare(unit)
    declared-unit[value]
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
    compact-number(x)
  } else if type(x) == dictionary and "kind" in x {
    str(x.kind)
  } else {
    str(x)
  }
}

#let diagnostic-table(data) = {
  let columns = data.headers
  let wide = columns.len() > 5

  text(size: if wide { 6.2pt } else { 8pt })[
    #table(
      columns: columns.len(),
      inset: if wide { 2pt } else { 4pt },
      stroke: 0.35pt,
      align: horizon,
      ..columns.map(h => text(weight: "bold")[#h]),
      ..data.rows.flatten().map(cell => {
        let rendered = display-value(cell)
        if wide {
          box(width: 100%)[#rendered]
        } else {
          [#rendered]
        }
      }),
    )
  ]
}
