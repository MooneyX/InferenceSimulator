import { useId, useRef, useState } from "react";

import type { ScanRequest, SimConfig } from "../api/types";
import { serializeScanPlan, serializeSimConfig } from "../config/parameterTransfer";

export function ParameterExport({ config, axes, label = "导出本次参数", unavailable }: {
  config: SimConfig | null | undefined;
  axes?: ScanRequest["axes"] | null;
  label?: string;
  unavailable?: string;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [json, setJson] = useState("");
  const [notice, setNotice] = useState<{ message: string; error: boolean } | null>(null);
  const subject = axes ? "扫描方案" : "本次参数";

  function exportParameters() {
    try {
      if (!config) throw new Error(unavailable ?? "历史结果未保存完整参数，无法导出");
      const text = axes
        ? serializeScanPlan(config, axes.map(({ key, values }) => ({ key, text: values.map(String).join(",") })))
        : serializeSimConfig(config);
      setJson(text);
      const reminder = config.workload.mode === "trace" ? "；再次使用时请重新上传请求计划" : "";
      setNotice({ message: `已生成${subject} JSON，可手动复制${reminder}`, error: false });
      try {
        void navigator.clipboard?.writeText(text).then(() => {
          setNotice({ message: `已生成并复制${subject} JSON${reminder}`, error: false });
        }).catch(() => {});
      } catch {}
    } catch (cause) {
      setJson("");
      setNotice({ message: `导出失败：${cause instanceof Error ? cause.message : "未知错误"}`, error: true });
    }
  }

  return <div className="parameter-transfer result-parameter-export" role="group" aria-label={label}>
    <button type="button" className="secondary-button secondary-button-small" onClick={exportParameters}
      disabled={!!unavailable} title={unavailable}>{label}</button>
    {unavailable && <span className="transfer-note">{unavailable}</span>}
    {json && <>
      <label className="transfer-label" htmlFor={id}>{subject} JSON</label>
      <input id={id} ref={input} value={json} readOnly aria-describedby={`${id}-notice`}
        onFocus={() => input.current?.select()} />
    </>}
    {notice && <span id={`${id}-notice`} role={notice.error ? "alert" : "status"}
      className={`transfer-note${notice.error ? " transfer-note-error" : ""}`}>{notice.message}</span>}
  </div>;
}
