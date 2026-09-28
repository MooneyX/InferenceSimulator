import { CustomChart, type CustomSeriesOption } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  type DataZoomComponentOption,
  type GridComponentOption,
  type TooltipComponentOption,
} from "echarts/components";
import * as echarts from "echarts/core";
import type { ComposeOption } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useMemo, useRef } from "react";

import type { RequestRecord } from "../api/types";
import {
  buildGanttModel,
  GANTT_PHASES,
  type GanttModel,
  type GanttPhase,
} from "../gantt/model";
import "./RequestGantt.css";

type GanttChartOption = ComposeOption<
  | CustomSeriesOption
  | DataZoomComponentOption
  | GridComponentOption
  | TooltipComponentOption
>;

echarts.use([
  CustomChart,
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

export interface RequestGanttProps {
  requests?: readonly RequestRecord[];
}

const PHASE_LABELS: Record<GanttPhase, string> = {
  queue: "排队",
  prefill: "Prefill",
  decode: "Decode",
};

const PHASE_COLORS: Record<GanttPhase, string> = {
  queue: "#64748b",
  prefill: "#3b82f6",
  decode: "#22c55e",
};

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 2,
});

function formatTime(value: number): string {
  return `${numberFormatter.format(value)} ms`;
}

function createTooltipFormatter(model: GanttModel) {
  const rowByRequestId = new Map(model.rows.map((row) => [row.id, row]));

  return (params: unknown): string => {
    if (typeof params !== "object" || params === null || !("value" in params)) {
      return "";
    }

    const value = params.value;
    if (!Array.isArray(value)) {
      return "";
    }

    const phase = GANTT_PHASES[Number(value[3])];
    const row = rowByRequestId.get(Number(value[4]));
    if (!phase || !row) {
      return "";
    }

    return [
      `<strong>请求 #${row.id}</strong>`,
      `阶段：${PHASE_LABELS[phase]}`,
      `TTFT：${formatTime(row.ttft)}`,
      `Decode 时长：${formatTime(row.decodeDuration)}`,
      `到达：${formatTime(row.arrival)}`,
      `Prefill 开始：${formatTime(row.prefill_start)}`,
      `Prefill 结束：${formatTime(row.prefill_end)}`,
      `完成：${formatTime(row.finish)}`,
    ].join("<br />");
  };
}

export function createGanttOption(model: GanttModel): GanttChartOption {
  const phaseIndex = new Map(GANTT_PHASES.map((phase, index) => [phase, index]));
  const data = model.segments.map((segment) => [
    segment.rowIndex,
    segment.start,
    segment.end,
    phaseIndex.get(segment.phase) ?? 0,
    segment.requestId,
  ]);
  const initialLastRow = Math.min(model.rows.length - 1, 39);

  return {
    animation: false,
    textStyle: {
      color: "#a9bad0",
    },
    grid: {
      top: 16,
      right: 48,
      bottom: 72,
      left: 84,
    },
    tooltip: {
      trigger: "item",
      appendToBody: true,
      formatter: createTooltipFormatter(model),
    },
    xAxis: {
      type: "value",
      min: model.startTime ?? 0,
      max: model.endTime ?? 0,
      name: "仿真时钟（ms）",
      nameLocation: "middle",
      nameGap: 38,
      nameTextStyle: { color: "#8ea3bd" },
      axisLine: { lineStyle: { color: "#415874" } },
      axisTick: { lineStyle: { color: "#415874" } },
      axisLabel: {
        color: "#8ea3bd",
        formatter: (value: number) => numberFormatter.format(value),
      },
      splitLine: {
        lineStyle: { color: "#1e324a" },
      },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: model.rows.map(({ id }) => `#${id}`),
      axisLine: { lineStyle: { color: "#415874" } },
      axisTick: { lineStyle: { color: "#415874" } },
      axisLabel: {
        color: "#8ea3bd",
        interval: 0,
      },
    },
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "weakFilter",
      },
      {
        type: "slider",
        xAxisIndex: 0,
        filterMode: "weakFilter",
        bottom: 18,
        height: 18,
      },
      {
        type: "inside",
        yAxisIndex: 0,
        filterMode: "filter",
        startValue: 0,
        endValue: initialLastRow,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: "slider",
        yAxisIndex: 0,
        filterMode: "filter",
        startValue: 0,
        endValue: initialLastRow,
        right: 8,
        width: 14,
      },
    ],
    series: [
      {
        name: "请求生命周期",
        type: "custom",
        dimensions: ["row", "start", "end", "phase", "requestId"],
        encode: {
          x: [1, 2],
          y: 0,
        },
        data,
        renderItem: (params, api) => {
          const rowIndex = Number(api.value(0));
          const phase = GANTT_PHASES[Number(api.value(3))];
          const start = api.coord([api.value(1), rowIndex]);
          const end = api.coord([api.value(2), rowIndex]);
          const categorySize = api.size?.([0, 1]) ?? [0, 20];
          const height =
            (Array.isArray(categorySize) ? categorySize[1] : categorySize) * 0.58;
          const coordinateSystem = params.coordSys as unknown as {
            x: number;
            y: number;
            width: number;
            height: number;
          };
          const shape = echarts.graphic.clipRectByRect(
            {
              x: start[0],
              y: start[1] - height / 2,
              width: Math.max(end[0] - start[0], 1),
              height,
            },
            coordinateSystem,
          );

          return shape && phase
            ? {
                type: "rect",
                shape,
                style: {
                  fill: PHASE_COLORS[phase],
                  opacity: 0.9,
                },
              }
            : undefined;
        },
      },
    ],
  };
}

export function RequestGantt({ requests = [] }: RequestGanttProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const model = useMemo(() => buildGanttModel(requests), [requests]);
  const hasRequests = model.rows.length > 0;
  const chartHeight = Math.min(720, Math.max(360, model.rows.length * 28 + 120));

  useEffect(() => {
    if (!hasRequests || !containerRef.current) {
      return;
    }

    const chart = echarts.init(containerRef.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, [hasRequests]);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.setOption(createGanttOption(model), true);
    }
  }, [model]);

  return (
    <section className="gantt-panel" aria-labelledby="gantt-heading">
      <div className="gantt-heading">
        <div>
          <p className="section-kicker">Request lifecycle</p>
          <h2 id="gantt-heading">请求生命周期</h2>
        </div>
        {hasRequests ? <p>共 {model.rows.length} 个请求，按到达时间排序</p> : null}
      </div>

      <div className="gantt-legend" aria-label="生命周期阶段图例">
        {GANTT_PHASES.map((phase) => (
          <span key={phase}>
            <i style={{ backgroundColor: PHASE_COLORS[phase] }} aria-hidden="true" />
            {PHASE_LABELS[phase]}
          </span>
        ))}
      </div>

      {hasRequests ? (
        <div
          ref={containerRef}
          className="gantt-chart"
          style={{ height: `${chartHeight}px` }}
          role="img"
          aria-label={`${model.rows.length} 个请求的排队、Prefill 和 Decode 生命周期甘特图`}
        />
      ) : (
        <p className="gantt-empty">运行仿真后将在此显示请求生命周期。</p>
      )}
    </section>
  );
}
