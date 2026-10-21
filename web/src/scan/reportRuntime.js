(function () {
  'use strict';
  const report = JSON.parse(document.getElementById('scan-data').textContent);
  const batches = report.data;
  const parameters = report.parameters;
  const roles = ['x', 'color', 'shape', 'line'];
  const palette = ['#6c63ff', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#8b5cf6', '#84cc16'];
  const symbols = ['circle', 'rect', 'triangle', 'diamond', 'pin', 'arrow', 'roundRect', 'emptyCircle'];
  const styles = ['solid', 'dashed', 'dotted', [10, 4, 2, 4]];
  const keys = report.keys;
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
  const tokenOf = v => JSON.stringify(v ?? null);
  const textOf = v => v == null ? '未设置' : typeof v === 'object' ? JSON.stringify(v) : String(v);
  const limits = { x: Infinity, color: palette.length, shape: symbols.length, line: styles.length };
  const rows = batches.flatMap(batch => batch.points.map(point => ({ batch, point,
    tokens: Object.fromEntries(keys.map(key => [key, tokenOf(value(point, key))])) })));
  const levels = Object.fromEntries(keys.map(key => {
    const unique = new Map(rows.map(row => [row.tokens[key], value(row.point, key) ?? null]));
    return [key, [...unique].sort(([, a], [, b]) => a === null ? 1 : b === null ? -1
      : typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b)))];
  }));
  let backgroundValues = {};
  let choosingBackground = false;
  let selected = {};
  roles.forEach((role, index) => {
    const input = byId('role-' + role);
    if (role !== 'x') input.add(option('', '无'));
    keys.forEach(key => input.add(option(key, label(key))));
    input.value = keys[index] || '';
    input.addEventListener('change', () => { choosingBackground = false; redraw(true); });
  });
  metrics.forEach(metric => byId('metric').add(option(metric, metric)));
  byId('metric').value = metrics.includes('throughput') ? 'throughput' : metrics[0] || '';
  byId('metric').addEventListener('change', () => redraw());
  byId('reset-values').addEventListener('click', () => redraw(true));
  byId('reset-background').addEventListener('click', () => {
    backgroundValues = {};
    choosingBackground = true;
    redraw(true);
  });
  batches.forEach(batch => {
    const check = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.value = batch.job.id; input.checked = true;
    input.addEventListener('change', () => redraw());
    check.append(input, document.createTextNode('扫描 ' + batch.job.id.slice(0, 8)));
    byId('batches').append(check);
  });
  const chart = window.echarts && window.echarts.init(byId('chart'));
  if (!chart) byId('chart').textContent = '图表脚本无法加载；下方数据表仍可使用。';
  window.addEventListener('resize', () => chart?.resize());
  let tableRows = [];
  const cell = (row, text) => { const td = document.createElement('td'); td.textContent = text; row.append(td); };
  function redraw(reset = false) {
    const active = new Set([...byId('batches').querySelectorAll('input:checked')].map(input => input.value));
    const used = Object.fromEntries(roles.map(role => [role, byId('role-' + role).value]));
    const roleKeys = roles.filter(role => used[role]).map(role => used[role]);
    const hidden = keys.filter(key => !roleKeys.includes(key));
    const metric = byId('metric').value;
    const activeRows = rows.filter(row => active.has(row.batch.job.id));
    const goodRows = activeRows.filter(row => row.point.status === 'completed'
      && typeof row.point.metrics?.[metric] === 'number' && value(row.point, used.x) != null);
    const matchesBackground = row => hidden.every(key =>
      !Object.prototype.hasOwnProperty.call(backgroundValues, key) || row.tokens[key] === backgroundValues[key]);
    if (choosingBackground && goodRows.length && !goodRows.some(matchesBackground)) backgroundValues = {};
    const anchor = goodRows.find(matchesBackground) || goodRows[0];
    if (!choosingBackground) {
      backgroundValues = anchor ? Object.fromEntries(hidden.map(key => [key, anchor.tokens[key]])) : {};
    }
    const complete = hidden.every(key => Object.prototype.hasOwnProperty.call(backgroundValues, key));
    const eligible = goodRows.filter(matchesBackground);
    const matchesRoles = row => roleKeys.every(key => selected[key]?.has(row.tokens[key]));
    roles.forEach(role => {
      const input = byId('role-' + role);
      [...input.options].forEach(item => {
        item.disabled = !!item.value && item.value !== input.value && roleKeys.includes(item.value);
      });
    });
    if (reset || !eligible.some(matchesRoles)) {
      selected = {};
      roleKeys.forEach(key => {
        const role = roles.find(item => used[item] === key);
        const available = levels[key].map(([token]) => token).filter(token =>
          eligible.some(row => row.tokens[key] === token));
        const first = anchor?.tokens[key];
        selected[key] = new Set([first, ...available.filter(token => token !== first)]
          .filter(token => token !== undefined).slice(0, limits[role]));
      });
    } else {
      roleKeys.forEach(key => {
        selected[key] = new Set([...selected[key]].filter(token => eligible.some(row =>
          row.tokens[key] === token && roleKeys.every(other =>
            other === key || selected[other].has(row.tokens[other])))));
      });
    }
    const backgroundHost = byId('background-values');
    backgroundHost.replaceChildren();
    byId('background').hidden = hidden.length === 0;
    hidden.forEach(key => {
      const group = document.createElement('fieldset');
      group.className = 'background-group';
      const title = document.createElement('legend');
      title.textContent = label(key);
      const options = document.createElement('div');
      options.className = 'background-options';
      (levels[key] || []).forEach(([token, v]) => {
        const item = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'radio'; input.name = 'background-' + key; input.value = token;
        input.checked = backgroundValues[key] === token;
        input.disabled = !goodRows.some(row => row.tokens[key] === token && matchesRoles(row)
          && hidden.every(other => other === key
            || !Object.prototype.hasOwnProperty.call(backgroundValues, other)
            || row.tokens[other] === backgroundValues[other]));
        input.addEventListener('change', () => {
          backgroundValues[key] = token;
          choosingBackground = !hidden.every(other =>
            Object.prototype.hasOwnProperty.call(backgroundValues, other));
          redraw(true);
        });
        item.append(input, document.createTextNode(textOf(v)));
        options.append(item);
      });
      group.append(title, options);
      backgroundHost.append(group);
    });
    roles.forEach(role => {
      const host = byId('values-' + role);
      host.replaceChildren();
      const key = used[role];
      if (!key) {
        const empty = document.createElement('span');
        empty.className = 'role-empty';
        empty.textContent = '未选择参数';
        host.append(empty);
        return;
      }
      (levels[key] || []).forEach(([token, v]) => {
        const item = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'checkbox'; input.checked = selected[key].has(token);
        const supported = eligible.some(row => row.tokens[key] === token
          && roleKeys.every(other => other === key || selected[other].has(row.tokens[other])));
        const canRemove = selected[key].size > 1 && eligible.some(row => row.tokens[key] !== token
          && roleKeys.every(other => selected[other].has(row.tokens[other])));
        input.disabled = !complete || !supported || (input.checked && !canRemove)
          || (!input.checked && selected[key].size >= limits[role]);
        input.addEventListener('change', () => {
          const next = new Set(selected[key]);
          if (input.checked) next.add(token); else next.delete(token);
          selected[key] = next;
          if (!next.size || !eligible.some(matchesRoles)) {
            if (input.checked) next.delete(token); else next.add(token);
          } else {
            roleKeys.forEach(other => {
              selected[other] = new Set([...selected[other]].filter(current => eligible.some(row =>
                row.tokens[other] === current && roleKeys.every(name =>
                  name === other || selected[name].has(row.tokens[name])))));
            });
          }
          redraw();
        });
        item.append(input, document.createTextNode(textOf(v)));
        host.append(item);
      });
    });
    const combinations = new Set(eligible.filter(matchesRoles).map(row =>
      JSON.stringify(roleKeys.map(key => row.tokens[key]))));
    const total = roleKeys.reduce((count, key) => count * selected[key].size, 1);
    byId('selection-notice').textContent = !eligible.length
      ? '当前批次与指标没有可绘制的数据点；无数据的选项已禁用。'
      : !complete ? '请为每项背景参数选择一个值；全部选齐后显示结果。'
        : `当前背景下 ${combinations.size} 组实际有数据，${total - combinations.size} 组勾选值组合无数据（不补算）。`;
    byId('csv').disabled = !complete;
    const seen = new Map();
    activeRows.forEach(({ batch, point, tokens }) => {
      if (!complete || !anchor || !hidden.every(key => tokens[key] === backgroundValues[key])
          || !roleKeys.every(key => selected[key].has(tokens[key]))) return;
      const signature = point.signature || batch.job.id + ':' + JSON.stringify(point.values);
      const existing = seen.get(signature);
      if (existing) existing.sources.push(batch.job.id);
      else seen.set(signature, { batch, point, sources: [batch.job.id] });
    });
    const entries = [...seen.values()];
    const visibleLevels = key => [...new Map(entries.map(entry => value(entry.point, key))
      .filter(v => v != null).map(v => [tokenOf(v), v])).values()];
    const xValues = visibleLevels(used.x).sort((a, b) => typeof a === 'number' && typeof b === 'number'
      ? a - b : textOf(a).localeCompare(textOf(b)));
    const colors = visibleLevels(used.color), shapes = visibleLevels(used.shape), lines = visibleLevels(used.line);
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
            label(used[role]) + '=' + textOf(value(point, used[role]))),
          ...remainder.map(([key, v]) => label(key) + '=' + textOf(v))].join(' · '),
        tooltipValues: [used.color && color, used.shape && shape, used.line && line]
          .filter(v => v !== '' && v != null).map(textOf),
        ci: Math.max(0, colors.findIndex(v => tokenOf(v) === tokenOf(color))),
        si: Math.max(0, shapes.findIndex(v => tokenOf(v) === tokenOf(shape))),
        li: Math.max(0, lines.findIndex(v => tokenOf(v) === tokenOf(line))), values: new Map(), sources,
      });
      const y = point.status === 'completed' && typeof point.metrics?.[metric] === 'number'
        ? point.metrics[metric] : null;
      groups.get(groupKey).values.set(tokenOf(x), y);
    });
    const series = [...groups.values()].map(group => ({
      name: group.name, tooltipValues: group.tooltipValues,
      type: 'line', smooth: true, connectNulls: false, showSymbol: true,
      symbol: symbols[group.si % symbols.length], symbolSize: 9,
      itemStyle: { color: palette[group.ci % palette.length] },
      lineStyle: { type: styles[group.li % styles.length], color: palette[group.ci % palette.length] },
      data: xValues.map(x => group.values.get(tokenOf(x)) ?? null),
    }));
    chart?.setOption({
      animation: false, backgroundColor: 'transparent', textStyle: { color: '#9ca0b0' },
      grid: { left: 64, right: 38, top: 40, bottom: 95 },
      legend: { type: 'scroll', bottom: 0, textStyle: { color: '#9ca0b0', fontSize: 10 } },
      tooltip: { trigger: 'axis', backgroundColor: '#242837', borderColor: '#2e3347',
        textStyle: { color: '#e4e4e7' },
        extraCssText: 'max-width:calc(100vw - 32px);white-space:normal;box-sizing:border-box;',
        formatter: params => formatScanTooltip(params, series) },
      toolbox: { iconStyle: { borderColor: '#9ca0b0' }, feature: { saveAsImage: {}, restore: {} } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 50, borderColor: '#2e3347',
        textStyle: { color: '#9ca0b0' }, fillerColor: 'rgba(108,99,255,.18)' }],
      xAxis: { type: 'category', name: label(used.x), data: xValues.map(textOf),
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
    byId('summary').textContent = !complete ? '背景参数尚未选齐，图表与点级数据暂不显示。'
      : entries.length + ' 个独立配置点；去重 '
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
