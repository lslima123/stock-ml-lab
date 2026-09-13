import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  emphasis?: boolean;
}

export function Metric({ label, value, detail, emphasis = false }: Props) {
  return (
    <div className={`metric ${emphasis ? "metric-emphasis" : ""}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {detail ? <span className="metric-detail">{detail}</span> : null}
    </div>
  );
}
