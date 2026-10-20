(function () {
  'use strict';
  const report = JSON.parse(document.getElementById('scan-data').textContent);
  const batches = report.data;
  const parameters = report.parameters;
  const roles = ['x', 'color', 'shape', 'line'];
  const palette = ['#6c63ff', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#8b5cf6', '#84cc16'];
  const symbols = ['circle', 'rect', 'triangle', 'diamond', 'pin', 'arrow', 'roundRect', 'emptyCircle'];
  const styles = ['solid', 'dashed', 'dotted', [10, 4, 2, 4]];
  const keys = [...new Set(batches.flatMap(batch => Object.values(batch.job.axes || {}).map(axis => axis.key)))];
  const metrics = [...new Set(batches.flatMap(batch => batch.points.flatMap(point =>
    Object.keys(point.metrics || {}).filter(key => typeof point.metrics[key] === 'number'))))].sort();
  const byId = id => document.getElementById(id);
  const label = key => parameters.find(item => item.key === key)?.label || key;
  const value = (point, key) => {
    if (!key) return null;
    if (Object.prototype.hasOwnProperty.call(point.values, key)) return point.values[key];
    if (!point.config) return null;
    if (key === 'inference.parallelism') return point.config.inference.tp_size;
    let current = point.config;
    for (const field of key.split('.')) current = current?.[field];
    if (key.endsWith('_len') && current && typeof current === 'object') {
      current = current.dist === 'fixed' ? current.value : current.mean;
    }
    if (key === 'shared_l3.available_gb' && typeof current === 'number') current /= 1000;
    if (['inference.hardware.memory_bytes_per_gpu', 'inference.storage.runtime_reserve_bytes'].includes(key)
        && typeof current === 'number') current /= 1e9;
    return current ?? null;
  };
  const option = (v, text) => new Option(text, v);
  roles.forEach((role, index) => {
    const input = byId('role-' + role);
    if (role !== 'x') input.add(option('', '无'));
    keys.forEach(key => input.add(option(key, label(key))));
    const first = batches[0]?.job.axes || {};
    input.value = first[role]?.key || (index === 0 ? keys[0] || '' : '');
    input.addEventListener('change', () => {
      for (const other of roles) if (other !== role && byId('role-' + other).value === input.value
        && input.value) byId('role-' + other).value = '';
      redraw();
    });
  });
  metrics.forEach(metric => byId('metric').add(option(metric, metric)));
  byId('metric').value = metrics.includes('throughput') ? 'throughput' : metrics[0] || '';
  byId('metric').addEventListener('change', redraw);
  batches.forEach(batch => {
    const check = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.value = batch.job.id; input.checked = true;
    input.addEventListener('change', redraw);
    check.append(input, document.createTextNode('扫描 ' + batch.job.id.slice(0, 8)));
    byId('batches').append(check);
  });
  const filters = {};
  keys.forEach(key => {
    const select = document.createElement('select');
    select.add(option('', '全部'));
    const unique = [...new Set(batches.flatMap(batch => batch.points.map(p => value(p, key)))
      .filter(v => v != null).map(String))].sort((a, b) => Number.isFinite(+a) && Number.isFinite(+b)
        ? Number(a) - Number(b) : a.localeCompare(b));
    unique.forEach(v => select.add(option(v, v)));
    select.addEventListener('change', redraw);
    filters[key] = select;
    const wrapper = document.createElement('label');
    wrapper.append(document.createTextNode(label(key) + ' '), select);
    byId('filters').append(wrapper);
  });
  const chart = window.echarts && window.echarts.init(byId('chart'));
  if (!chart) byId('chart').textContent = '图表脚本无法加载；下方数据表仍可使用。';
  window.addEventListener('resize', () => chart?.resize());
  let tableRows = [];
  const cell = (row, text) => { const td = document.createElement('td'); td.textContent = text; row.append(td); };
  function redraw() {
    const active = new Set([...byId('batches').querySelectorAll('input:checked')].map(input => input.value));
    const used = Object.fromEntries(roles.map(role => [role, byId('role-' + role).value]));
    const metric = byId('metric').value;
    const seen = new Map();
    batches.filter(batch => active.has(batch.job.id)).forEach(batch => batch.points.forEach(point => {
      if (keys.some(key => filters[key].value && String(value(point, key)) !== filters[key].value)) return;
      const signature = point.signature || batch.job.id + ':' + JSON.stringify(point.values);
      const existing = seen.get(signature);
      if (existing) existing.sources.push(batch.job.id);
      else seen.set(signature, { batch, point, sources: [batch.job.id] });
    }));
    const entries = [...seen.values()];
    const levels = key => [...new Set(entries.map(entry => value(entry.point, key)).filter(v => v != null))];
    const xValues = levels(used.x).sort((a, b) => typeof a === 'number' && typeof b === 'number'
      ? a - b : String(a).localeCompare(String(b)));
    const colors = levels(used.color), shapes = levels(used.shape), lines = levels(used.line);
    const hidden = keys.filter(key => !Object.values(used).includes(key));
    const groups = new Map();
    entries.forEach(({ batch, point, sources }) => {
      const x = value(point, used.x);
      if (x == null) return;
      const color = value(point, used.color), shape = value(point, used.shape), line = value(point, used.line);
      const remainder = hidden.map(key => [key, value(point, key)]);
      const groupKey = JSON.stringify([batch.job.id, color, shape, line, remainder]);
      if (!groups.has(groupKey)) groups.set(groupKey, {
        name: ['扫描 ' + batch.job.id.slice(0, 6),
          ...roles.slice(1).filter(role => used[role]).map(role =>
            label(used[role]) + '=' + String(value(point, used[role]))),
          ...remainder.map(([key, v]) => label(key) + '=' + String(v))].join(' · '),
        ci: Math.max(0, colors.indexOf(color)), si: Math.max(0, shapes.indexOf(shape)),
        li: Math.max(0, lines.indexOf(line)), values: new Map(), sources,
      });
      const y = point.status === 'completed' && typeof point.metrics?.[metric] === 'number'
        ? point.metrics[metric] : null;
      groups.get(groupKey).values.set(String(x), y);
    });
    const series = [...groups.values()].map(group => ({
      name: group.name, type: 'line', connectNulls: false, showSymbol: true,
      symbol: symbols[group.si % symbols.length], symbolSize: 9,
      itemStyle: { color: palette[group.ci % palette.length] },
      lineStyle: { type: styles[group.li % styles.length], color: palette[group.ci % palette.length] },
      data: xValues.map(x => group.values.get(String(x)) ?? null),
    }));
    chart?.setOption({
      animation: false, backgroundColor: 'transparent', textStyle: { color: '#9ca0b0' },
      grid: { left: 64, right: 38, top: 40, bottom: 95 },
      legend: { type: 'scroll', bottom: 0, textStyle: { color: '#9ca0b0', fontSize: 10 } },
      tooltip: { trigger: 'axis', backgroundColor: '#242837', borderColor: '#2e3347',
        textStyle: { color: '#e4e4e7' } },
      toolbox: { iconStyle: { borderColor: '#9ca0b0' }, feature: { saveAsImage: {}, restore: {} } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 50, borderColor: '#2e3347',
        textStyle: { color: '#9ca0b0' }, fillerColor: 'rgba(108,99,255,.18)' }],
      xAxis: { type: 'category', name: label(used.x), data: xValues.map(String),
        nameTextStyle: { color: '#9ca0b0' }, axisLabel: { color: '#9ca0b0' },
        axisLine: { lineStyle: { color: '#2e3347' } } },
      yAxis: { type: 'value', name: metric, scale: true,
        nameTextStyle: { color: '#9ca0b0' }, axisLabel: { color: '#9ca0b0' },
        splitLine: { lineStyle: { color: '#2e3347' } } }, series,
    }, true);
    const body = byId('data-body'); body.replaceChildren();
    tableRows = entries.map(({ batch, point, sources }) => ({
      scan: sources.map(id => id.slice(0, 8)).join(','), x: value(point, used.x),
      color: value(point, used.color), shape: value(point, used.shape),
      line: value(point, used.line), metric: point.metrics?.[metric] ?? null,
      status: point.status, error: point.error || '', config: JSON.stringify(point.config || point.values),
    }));
    tableRows.forEach(data => {
      const row = document.createElement('tr');
      for (const key of ['scan', 'x', 'color', 'shape', 'line', 'metric', 'status', 'error', 'config']) {
        cell(row, data[key] == null ? '—' : String(data[key]));
      }
      body.append(row);
    });
    byId('summary').textContent = entries.length + ' 个独立配置点；去重 '
      + (batches.filter(batch => active.has(batch.job.id)).reduce((n, batch) => n + batch.points.length, 0) - entries.length)
      + ' 个相同输入点。缺失结果不插值或平均。';
  }
  byId('csv').addEventListener('click', () => {
    const fields = ['scan', 'x', 'color', 'shape', 'line', 'metric', 'status', 'error', 'config'];
    const csv = [fields.join(','), ...tableRows.map(row => fields.map(key =>
      '"' + String(row[key] ?? '').replaceAll('"', '""') + '"').join(','))].join('\r\n');
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob(['\ufeff', csv],
      { type: 'text/csv;charset=utf-8' })); link.download = 'scan-data.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  });
  redraw();
})();
