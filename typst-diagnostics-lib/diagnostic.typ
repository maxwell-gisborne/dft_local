#let diagnostic-figure(title: none, body, caption: none, caveats: ()) = {
  if title != none [*#title*]
  figure(body, caption: caption)
  if caveats.len() > 0 {
    block[
      *Caveats*
      #for caveat in caveats [- #caveat]
    ]
  }
}

#let diagnostic-note(body) = block[#body]
