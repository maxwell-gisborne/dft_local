// @ts-check

class DftLineGraph extends HTMLElement {
  connectedCallback() {
    if (!this.hasAttribute("data-ready")) {
      this.setAttribute("data-ready", "true");
    }
  }
}

class DftKSpacePlot extends HTMLElement {
  connectedCallback() {
    if (!this.hasAttribute("data-ready")) {
      this.setAttribute("data-ready", "true");
    }
  }
}

if (!customElements.get("dft-line-graph")) {
  customElements.define("dft-line-graph", DftLineGraph);
}

if (!customElements.get("dft-kspace-plot")) {
  customElements.define("dft-kspace-plot", DftKSpacePlot);
}
