#import "@preview/zero:0.6.1": num, zi

#let diagnostic-num(x) = {
  num(
    x,
    exponent: (sci: 3),
    round: (mode: "figures", precision: 6, pad: false),
  )
}

#let render-quantity(q) = {
  // Only use the clean display symbol. Do not fall back to q.unit,
  // because q.unit is the Python repr Unit(...), not a Typst unit.
  let unit = if "unit_symbol" in q {
    q.unit_symbol
  } else {
    ""
  }

  if unit == "" or unit == none or unit == "None" {
    diagnostic-num(q.value)
  } else {
    let declared-unit = zi.declare(unit)
    declared-unit(
      q.value,
      exponent: (sci: 3),
      round: (mode: "figures", precision: 6, pad: false),
    )
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
    diagnostic-num(x)
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
