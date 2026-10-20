export interface TooltipLine {
  name: string;
  symbol: string;
  lineStyle: { color: string; type: string | number[] };
}

export function formatScanTooltip(params: unknown, series: TooltipLine[]): string;
