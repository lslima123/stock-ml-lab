import type { ScopeCapability, ScopeName } from "../types";

interface Props {
  scopes: ScopeCapability[];
  value: ScopeName;
  onChange: (value: ScopeName) => void;
}

export function ScopeSelector({ scopes, value, onChange }: Props) {
  return (
    <div className="scope-grid" role="radiogroup" aria-label="Model scope">
      {scopes.map((scope) => (
        <button
          className={`scope-card ${value === scope.name ? "selected" : ""}`}
          disabled={!scope.available}
          key={scope.name}
          onClick={() => scope.available && onChange(scope.name)}
          type="button"
          role="radio"
          aria-checked={value === scope.name}
          title={scope.description}
        >
          <span className="scope-title">
            {scope.name === "local"
              ? "Local"
              : scope.name === "global"
                ? "Global"
                : "Compare"}
          </span>
          <span className="scope-description">
            {scope.name === "local"
              ? "Ticker-specific fit"
              : scope.name === "global"
                ? "Pooled multi-asset model"
                : "Same-family local vs global"}
          </span>
          <span className={`availability ${scope.available ? "available" : "reserved"}`}>
            {scope.available ? "Available" : "Reserved"}
          </span>
        </button>
      ))}
    </div>
  );
}
