import type { SimResult } from "../api/types";
import "./WarningsPanel.css";

export interface WarningsPanelProps {
  warnings?: SimResult["warnings"];
}

export function WarningsPanel({ warnings }: WarningsPanelProps) {
  if (!warnings?.length) {
    return null;
  }

  return (
    <section className="warnings-panel" aria-labelledby="warnings-heading">
      <div className="warnings-icon" aria-hidden="true">
        !
      </div>
      <div>
        <h2 id="warnings-heading">仿真告警</h2>
        <ul>
          {warnings.map((warning, index) => (
            <li key={`${index}-${warning}`}>{warning}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
