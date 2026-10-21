const escapeTooltip = value => String(value).replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[char]);

const tooltipSymbol = (symbol, color) => {
  const fill = symbol === 'emptyCircle' ? 'none' : color;
  const paint = `fill="${fill}" stroke="${color}" stroke-width="1.5"`;
  switch (symbol) {
    case 'rect': return `<rect x="24" y="7" width="8" height="8" ${paint}/>`;
    case 'triangle': return `<path d="M28 5 L34 16 H22 Z" ${paint}/>`;
    case 'diamond': return `<path d="M28 5 L34 11 L28 17 L22 11 Z" ${paint}/>`;
    case 'pin': return `<path d="M28 18 C25 14 23 12 23 9 A5 5 0 1 1 33 9 C33 12 31 14 28 18 Z" ${paint}/>`;
    case 'arrow': return `<path d="M22 8 H28 V5 L35 11 L28 17 V14 H22 Z" ${paint}/>`;
    case 'roundRect': return `<rect x="24" y="7" width="8" height="8" rx="2.5" ${paint}/>`;
    default: return `<circle cx="28" cy="11" r="5" ${paint}/>`;
  }
};

export function formatScanTooltip(params, series) {
  const items = Array.isArray(params) ? params : [params];
  if (!items.length || !items[0]) return '';
  const x = escapeTooltip(items[0].axisValueLabel ?? items[0].axisValue ?? '');
  const rows = items.map(item => {
    const curve = series[item.seriesIndex];
    if (!curve) return '';
    const color = escapeTooltip(curve.lineStyle.color);
    const style = curve.lineStyle.type;
    const dash = Array.isArray(style) ? style.join(',')
      : style === 'dashed' ? '8,5' : style === 'dotted' ? '2,4' : '';
    const sample = `<svg width="56" height="22" viewBox="0 0 56 22" aria-hidden="true" style="flex:none;vertical-align:middle">`
      + `<line x1="2" y1="11" x2="54" y2="11" stroke="${color}" stroke-width="2"`
      + (dash ? ` stroke-dasharray="${dash}"` : '') + '/>'
      + tooltipSymbol(curve.symbol, color) + '</svg>';
    const values = curve.tooltipValues.map(escapeTooltip).join(' · ');
    return `<div style="display:flex;align-items:center;gap:6px;margin-top:4px;min-width:0">${sample}`
      + `<span style="min-width:0;white-space:normal;overflow-wrap:anywhere">${values ? `${values}: ` : ''}`
      + `${escapeTooltip(item.value ?? '—')}</span></div>`;
  });
  return `<div style="width:max-content;max-width:min(640px,calc(100vw - 56px));white-space:normal;overflow-wrap:anywhere">`
    + `<div>${x}</div>${rows.join('')}</div>`;
}
