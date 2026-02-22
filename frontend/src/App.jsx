import { useState, useEffect, useCallback } from "react"
import axios from "axios"
import {
  Upload, FileText, CheckCircle,
  XCircle, Clock, ChevronRight, X
} from "lucide-react"
import "./App.css"

const api = axios.create({ baseURL: "/api" })

// ── Confidence Badge ──────────────────────────────────────────
const Badge = ({ level }) => {
  const colors = {
    HIGH: { bg: "#064e3b", text: "#34d399", border: "#065f46" },
    MEDIUM: { bg: "#451a03", text: "#fb923c", border: "#78350f" },
    LOW: { bg: "#450a0a", text: "#f87171", border: "#7f1d1d" },
  }
  const c = colors[level] || colors.LOW
  return (
    <span style={{
      background: c.bg, color: c.text, border: `1px solid ${c.border}`,
      padding: "2px 8px", fontSize: "11px", fontFamily: "JetBrains Mono, monospace",
      fontWeight: 700, letterSpacing: "0.05em"
    }}>
      {level}
    </span>
  )
}

// ── Source Drawer ─────────────────────────────────────────────
const SourceDrawer = ({ spec, onClose }) => {
  if (!spec) return null
  return (
    <div style={{
      position: "fixed", right: 0, top: 0, height: "100vh", width: "420px",
      background: "#0a0c10", borderLeft: "1px solid #1e2530",
      padding: "32px 24px", zIndex: 100, overflowY: "auto",
      boxShadow: "-8px 0 40px rgba(0,0,0,0.6)"
    }}>
      <button onClick={onClose} style={{
        position: "absolute", top: 16, right: 16,
        background: "none", border: "none", color: "#6b7280", cursor: "pointer"
      }}>
        <X size={20} />
      </button>
      <div style={{
        color: "#f59e0b", fontSize: "11px",
        fontFamily: "JetBrains Mono", marginBottom: 8
      }}>
        SOURCE REFERENCE
      </div>
      <h3 style={{
        color: "#e5e7eb", fontSize: "16px",
        marginBottom: 24, lineHeight: 1.4
      }}>
        {spec.item_name}
      </h3>

      <div style={{ marginBottom: 20 }}>
        <div style={{ color: "#6b7280", fontSize: "11px", marginBottom: 6 }}>
          SPECIFICATION
        </div>
        <div style={{
          color: "#d1d5db", fontSize: "13px", lineHeight: 1.6,
          padding: "12px", background: "#111318",
          borderLeft: "3px solid #f59e0b"
        }}>
          {spec.specification_text}
        </div>
      </div>

      <div style={{ marginBottom: 20 }}>
        <div style={{ color: "#6b7280", fontSize: "11px", marginBottom: 6 }}>
          EXACT TEXT FROM DOCUMENT
        </div>
        <div style={{
          color: "#a3e635", fontSize: "12px", lineHeight: 1.6,
          padding: "12px", background: "#111318",
          fontFamily: "JetBrains Mono, monospace",
          borderLeft: "3px solid #a3e635"
        }}>
          "{spec.source?.exact_text || "N/A"}"
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          ["PAGE", spec.source?.page],
          ["UNIT", spec.unit],
          ["VALUE", spec.numeric_value],
          ["STANDARD", spec.standard_reference],
          ["MATERIAL", spec.material],
          ["TOLERANCE", spec.tolerance],
        ].map(([label, val]) => (
          <div key={label} style={{ padding: "10px 12px", background: "#111318" }}>
            <div style={{ color: "#6b7280", fontSize: "10px", marginBottom: 4 }}>
              {label}
            </div>
            <div style={{
              color: val && val !== "NOT_FOUND" ? "#e5e7eb" : "#374151",
              fontSize: "12px", fontFamily: "JetBrains Mono, monospace"
            }}>
              {val || "—"}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Specs Table ───────────────────────────────────────────────
const SpecsTable = ({ specs }) => {
  const [search, setSearch] = useState("")
  const [confFilter, setConfFilter] = useState("ALL")
  const [selected, setSelected] = useState(null)

  const filtered = specs.filter(s => {
    const matchText = !search ||
      s.item_name?.toLowerCase().includes(search.toLowerCase()) ||
      s.specification_text?.toLowerCase().includes(search.toLowerCase())
    const matchConf = confFilter === "ALL" || s.confidence === confFilter
    return matchText && matchConf
  })

  return (
    <div>
      <SourceDrawer spec={selected} onClose={() => setSelected(null)} />

      {/* Filter Bar */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center" }}>
        <input
          placeholder="Search specs..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            flex: 1, background: "#111318", border: "1px solid #1e2530",
            color: "#e5e7eb", padding: "8px 14px", fontSize: "13px",
            outline: "none", fontFamily: "DM Sans, sans-serif"
          }}
        />
        <select
          value={confFilter}
          onChange={e => setConfFilter(e.target.value)}
          style={{
            background: "#111318", border: "1px solid #1e2530",
            color: "#e5e7eb", padding: "8px 14px", fontSize: "13px",
            cursor: "pointer"
          }}
        >
          {["ALL", "HIGH", "MEDIUM", "LOW"].map(v =>
            <option key={v}>{v}</option>)}
        </select>
        <div style={{
          color: "#6b7280", fontSize: "12px", whiteSpace: "nowrap",
          fontFamily: "JetBrains Mono"
        }}>
          {filtered.length} / {specs.length} specs
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #f59e0b" }}>
              {["Item Name", "Specification", "Unit", "Value", "Standard", "Conf", "Pg", ""].map(h => (
                <th key={h} style={{
                  padding: "10px 12px", textAlign: "left",
                  color: "#f59e0b", fontFamily: "JetBrains Mono",
                  fontSize: "11px", letterSpacing: "0.08em",
                  background: "#0a0c10", whiteSpace: "nowrap"
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((spec, i) => (
              <tr key={i}
                onClick={() => setSelected(spec)}
                style={{
                  borderBottom: "1px solid #111318",
                  cursor: "pointer",
                  background: i % 2 === 0 ? "#0d1017" : "#0a0c10",
                  transition: "background 0.1s"
                }}
                onMouseEnter={e => e.currentTarget.style.background = "#151b25"}
                onMouseLeave={e => e.currentTarget.style.background =
                  i % 2 === 0 ? "#0d1017" : "#0a0c10"}
              >
                <td style={{
                  padding: "10px 12px", color: "#e5e7eb",
                  maxWidth: 160, overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: "nowrap"
                }}>
                  {spec.item_name}
                </td>
                <td style={{
                  padding: "10px 12px", color: "#9ca3af",
                  maxWidth: 280, overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: "nowrap"
                }}>
                  {spec.specification_text}
                </td>
                <td style={{
                  padding: "10px 12px", color: "#6b7280",
                  fontFamily: "JetBrains Mono"
                }}>
                  {spec.unit !== "NOT_FOUND" ? spec.unit : "—"}
                </td>
                <td style={{
                  padding: "10px 12px", color: "#6b7280",
                  fontFamily: "JetBrains Mono"
                }}>
                  {spec.numeric_value !== "NOT_FOUND" ? spec.numeric_value : "—"}
                </td>
                <td style={{
                  padding: "10px 12px", color: "#6b7280",
                  fontFamily: "JetBrains Mono", fontSize: "11px"
                }}>
                  {spec.standard_reference !== "NOT_FOUND" ? spec.standard_reference : "—"}
                </td>
                <td style={{ padding: "10px 12px" }}>
                  <Badge level={spec.confidence} />
                </td>
                <td style={{
                  padding: "10px 12px", color: "#6b7280",
                  fontFamily: "JetBrains Mono"
                }}>
                  {spec.source?.page || "—"}
                </td>
                <td style={{ padding: "10px 12px" }}>
                  <ChevronRight size={14} color="#374151" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div style={{ textAlign: "center", padding: "40px", color: "#374151" }}>
            No specs match current filters
          </div>
        )}
      </div>
    </div>
  )
}

// ── Scope Panel ───────────────────────────────────────────────
const ScopePanel = ({ scope }) => {
  const tasks = scope?.tasks || []
  const exclusions = scope?.exclusions || []
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
      <div>
        <div style={{
          color: "#f59e0b", fontSize: "11px",
          fontFamily: "JetBrains Mono", marginBottom: 16,
          letterSpacing: "0.08em"
        }}>
          TASKS ({tasks.length})
        </div>
        {tasks.length === 0 ? (
          <div style={{ color: "#374151", fontSize: "13px" }}>No tasks extracted</div>
        ) : tasks.map((t, i) => (
          <div key={i} style={{
            padding: "14px", background: "#0d1017",
            borderLeft: "3px solid #f59e0b", marginBottom: 10
          }}>
            <div style={{
              color: "#e5e7eb", fontSize: "13px",
              lineHeight: 1.5, marginBottom: 8
            }}>
              {t.task_description}
            </div>
            <div style={{
              color: "#6b7280", fontSize: "11px",
              fontFamily: "JetBrains Mono"
            }}>
              {t.responsible_party !== "NOT_FOUND" && `Party: ${t.responsible_party}`}
              {t.timeline !== "NOT_FOUND" && ` · ${t.timeline}`}
            </div>
          </div>
        ))}
      </div>
      <div>
        <div style={{
          color: "#f87171", fontSize: "11px",
          fontFamily: "JetBrains Mono", marginBottom: 16,
          letterSpacing: "0.08em"
        }}>
          EXCLUSIONS ({exclusions.length})
        </div>
        {exclusions.length === 0 ? (
          <div style={{ color: "#374151", fontSize: "13px" }}>
            No exclusions extracted
          </div>
        ) : exclusions.map((e, i) => (
          <div key={i} style={{
            padding: "14px", background: "#0d1017",
            borderLeft: "3px solid #7f1d1d", marginBottom: 10,
            color: "#9ca3af", fontSize: "13px", lineHeight: 1.5
          }}>
            {e.exclusion_description}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Job Card ──────────────────────────────────────────────────
const StatusIcon = ({ status }) => {
  if (status === "done") return <CheckCircle size={14} color="#34d399" />
  if (status === "error") return <XCircle size={14} color="#f87171" />
  if (status === "running") return <div style={{
    width: 14, height: 14, border: "2px solid #f59e0b",
    borderTopColor: "transparent", borderRadius: "50%",
    animation: "spin 0.8s linear infinite"
  }} />
  return <Clock size={14} color="#6b7280" />
}

const JobCard = ({ job, selected, onClick }) => (
  <div
    onClick={onClick}
    style={{
      padding: "14px 16px", cursor: "pointer",
      background: selected ? "#111827" : "transparent",
      borderLeft: selected ? "3px solid #f59e0b" : "3px solid transparent",
      borderBottom: "1px solid #111318",
      transition: "all 0.15s"
    }}
  >
    <div style={{
      display: "flex", alignItems: "center",
      gap: 8, marginBottom: 6
    }}>
      <StatusIcon status={job.status} />
      <div style={{
        color: "#e5e7eb", fontSize: "12px", flex: 1,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"
      }}>
        {job.filename}
      </div>
    </div>
    {(job.status === "running" || job.status === "queued") && (
      <div>
        <div style={{
          height: 2, background: "#1e2530", marginBottom: 6, overflow: "hidden"
        }}>
          <div style={{
            height: "100%", background: "#f59e0b",
            width: `${job.progress}%`, transition: "width 0.5s ease"
          }} />
        </div>
        <div style={{
          color: "#6b7280", fontSize: "11px",
          fontFamily: "JetBrains Mono"
        }}>
          {job.message}
        </div>
      </div>
    )}
    {job.status === "done" && (
      <div style={{
        color: "#34d399", fontSize: "11px",
        fontFamily: "JetBrains Mono"
      }}>
        {job.message}
      </div>
    )}
    {job.status === "error" && (
      <div style={{ color: "#f87171", fontSize: "11px" }}>{job.message}</div>
    )}
  </div>
)

// ── Upload Zone ───────────────────────────────────────────────
const UploadZone = ({ onUploaded }) => {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  const handleFile = async (file) => {
    if (!file?.name.endsWith(".pdf")) return
    setUploading(true)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await api.post("/upload", form)
      onUploaded(res.data)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault(); setDragging(false)
        handleFile(e.dataTransfer.files[0])
      }}
      onClick={() => document.getElementById("file-input").click()}
      style={{
        margin: "16px", padding: "24px",
        border: `2px dashed ${dragging ? "#f59e0b" : "#1e2530"}`,
        textAlign: "center", cursor: "pointer",
        transition: "border-color 0.2s",
        background: dragging ? "#111318" : "transparent"
      }}
    >
      <input
        id="file-input" type="file" accept=".pdf" hidden
        onChange={e => handleFile(e.target.files[0])}
      />
      <Upload size={20} color={dragging ? "#f59e0b" : "#374151"}
        style={{ margin: "0 auto 8px" }} />
      <div style={{ color: dragging ? "#f59e0b" : "#6b7280", fontSize: "12px" }}>
        {uploading ? "Uploading..." : "Drop PDF or click to upload"}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────
export default function App() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [result, setResult] = useState(null)
  const [activeTab, setActiveTab] = useState("specs")

  // Poll all active jobs
  useEffect(() => {
    const interval = setInterval(async () => {
      const active = jobs.filter(j =>
        j.status === "running" || j.status === "queued")
      for (const job of active) {
        const res = await api.get(`/jobs/${job.job_id}/status`)
        setJobs(prev => prev.map(j =>
          j.job_id === job.job_id ? res.data : j))
        if (res.data.status === "done" &&
          selectedJob?.job_id === job.job_id) {
          const r = await api.get(`/jobs/${job.job_id}/result`)
          setResult(r.data)
        }
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [jobs, selectedJob])

  const handleSelect = async (job) => {
    setSelectedJob(job)
    setResult(null)
    if (job.status === "done") {
      const r = await api.get(`/jobs/${job.job_id}/result`)
      setResult(r.data)
    }
  }

  const specs = result?.technical_specifications || []
  const scope = result?.scope_of_work || {}

  return (
    <div style={{
      display: "flex", height: "100vh",
      background: "#0f1117", color: "#e5e7eb",
      fontFamily: "DM Sans, sans-serif"
    }}>
      {/* Sidebar */}
      <div style={{
        width: 280, borderRight: "1px solid #1e2530",
        display: "flex", flexDirection: "column", flexShrink: 0
      }}>
        <div style={{
          padding: "20px 16px", borderBottom: "1px solid #1e2530"
        }}>
          <div style={{
            color: "#f59e0b", fontSize: "13px",
            fontFamily: "JetBrains Mono", letterSpacing: "0.1em",
            fontWeight: 700
          }}>
            TENDER EXTRACT PRO
          </div>
          <div style={{ color: "#374151", fontSize: "11px", marginTop: 2 }}>
            Technical Specification Extractor
          </div>
        </div>

        <UploadZone onUploaded={job => {
          setJobs(prev => [job, ...prev])
          setSelectedJob(job)
        }} />

        <div style={{ flex: 1, overflowY: "auto" }}>
          {jobs.length === 0 && (
            <div style={{
              textAlign: "center", padding: "32px 16px",
              color: "#374151", fontSize: "12px"
            }}>
              Upload a tender PDF to begin
            </div>
          )}
          {jobs.map(job => (
            <JobCard
              key={job.job_id}
              job={job}
              selected={selectedJob?.job_id === job.job_id}
              onClick={() => handleSelect(job)}
            />
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {!selectedJob ? (
          <div style={{
            flex: 1, display: "flex", alignItems: "center",
            justifyContent: "center", color: "#374151"
          }}>
            <div style={{ textAlign: "center" }}>
              <FileText size={48} style={{ margin: "0 auto 16px", opacity: 0.3 }} />
              <div>Upload a tender document to extract specifications</div>
            </div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div style={{
              padding: "20px 32px", borderBottom: "1px solid #1e2530",
              display: "flex", alignItems: "center", justifyContent: "space-between"
            }}>
              <div>
                <div style={{ color: "#e5e7eb", fontSize: "16px", fontWeight: 600 }}>
                  {selectedJob.filename}
                </div>
                {result && (
                  <div style={{
                    color: "#6b7280", fontSize: "12px",
                    fontFamily: "JetBrains Mono", marginTop: 4
                  }}>
                    {specs.length} specs · {" "}
                    {specs.filter(s => s.confidence === "HIGH").length} HIGH · {" "}
                    {(scope.tasks || []).length} tasks · {" "}
                    {(scope.exclusions || []).length} exclusions
                  </div>
                )}
              </div>
              {selectedJob.status === "running" && (
                <div style={{
                  color: "#f59e0b", fontSize: "12px",
                  fontFamily: "JetBrains Mono"
                }}>
                  {selectedJob.progress}% — {selectedJob.message}
                </div>
              )}
            </div>

            {/* Tabs */}
            {result && (
              <div style={{ borderBottom: "1px solid #1e2530", padding: "0 32px" }}>
                {["specs", "scope"].map(tab => (
                  <button key={tab} onClick={() => setActiveTab(tab)} style={{
                    background: "none", border: "none", cursor: "pointer",
                    padding: "12px 0", marginRight: 32,
                    color: activeTab === tab ? "#f59e0b" : "#6b7280",
                    fontSize: "12px", fontFamily: "JetBrains Mono",
                    letterSpacing: "0.08em",
                    borderBottom: activeTab === tab ? "2px solid #f59e0b" : "none",
                    marginBottom: -1
                  }}>
                    {tab === "specs"
                      ? `SPECIFICATIONS (${specs.length})`
                      : `SCOPE OF WORK`}
                  </button>
                ))}
              </div>
            )}

            {/* Content */}
            <div style={{ padding: "24px 32px", flex: 1 }}>
              {selectedJob.status === "error" && (
                <div style={{
                  padding: "20px", background: "#1c0a0a",
                  borderLeft: "4px solid #ef4444", color: "#f87171"
                }}>
                  <div style={{ fontWeight: 600, marginBottom: 8 }}>
                    Extraction Failed
                  </div>
                  <div style={{ fontSize: "13px" }}>{selectedJob.message}</div>
                </div>
              )}

              {(selectedJob.status === "running" ||
                selectedJob.status === "queued") && (
                  <div style={{ textAlign: "center", padding: "60px" }}>
                    <div style={{
                      width: 48, height: 48, border: "3px solid #1e2530",
                      borderTopColor: "#f59e0b", borderRadius: "50%",
                      animation: "spin 1s linear infinite",
                      margin: "0 auto 24px"
                    }} />
                    <div style={{ color: "#6b7280", marginBottom: 8 }}>
                      {selectedJob.message}
                    </div>
                    <div style={{
                      width: 300, height: 4, background: "#1e2530",
                      margin: "0 auto", overflow: "hidden"
                    }}>
                      <div style={{
                        height: "100%", background: "#f59e0b",
                        width: `${selectedJob.progress}%`,
                        transition: "width 0.5s ease"
                      }} />
                    </div>
                  </div>
                )}

              {result && activeTab === "specs" && (
                <SpecsTable specs={specs} />
              )}
              {result && activeTab === "scope" && (
                <ScopePanel scope={scope} />
              )}
            </div>
          </>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0f1117; }
        @keyframes spin { to { transform: rotate(360deg) } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0c10; }
        ::-webkit-scrollbar-thumb { background: #1e2530; }
      `}</style>
    </div>
  )
}
