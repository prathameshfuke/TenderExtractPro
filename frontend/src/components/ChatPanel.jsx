import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Send, Trash2, BookOpen, ChevronRight, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

const STARTER_QUESTIONS = [
  'What are the main technical requirements?',
  'What is explicitly excluded from scope?',
  'What delivery or completion timeline is mentioned?',
  'What are the eligibility and qualification criteria?',
  'What are the penalty or liquidated damages clauses?',
];

function ConfidenceBadge({ confidence }) {
  const level = typeof confidence === 'number'
    ? (confidence >= 0.85 ? 'HIGH' : confidence >= 0.55 ? 'MEDIUM' : 'LOW')
    : String(confidence || 'LOW').toUpperCase();
  const colors = { HIGH: '#22c55e', MEDIUM: '#f59e0b', LOW: '#ef4444' };
  const color = colors[level] || colors.LOW;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      fontSize: '10px', fontWeight: 700, letterSpacing: '0.06em',
      color, textTransform: 'uppercase',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block' }} />
      {level}
    </span>
  );
}

function CitationCard({ citation, index }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: '1px solid var(--hairline)', borderRadius: '10px',
      overflow: 'hidden', fontSize: '12.5px', background: 'var(--canvas)',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: '8px',
          padding: '8px 12px', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-muted)', textAlign: 'left',
        }}
      >
        <BookOpen size={12} style={{ flexShrink: 0, color: 'var(--accent)' }} />
        <span style={{ flex: 1, fontWeight: 600, color: 'var(--text-ink)', fontSize: '11px' }}>
          Source {index + 1}{citation.section ? ` · ${citation.section}` : ''}
        </span>
        {citation.page > 0 && (
          <span style={{ fontSize: '10px', color: 'var(--text-muted-soft)', marginRight: '6px' }}>
            p.{citation.page}
          </span>
        )}
        <ChevronRight size={12} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }} />
      </button>
      {open && (
        <div style={{
          padding: '10px 12px 12px', borderTop: '1px solid var(--hairline)',
          color: 'var(--text-muted)', lineHeight: 1.6, fontStyle: 'italic',
        }}>
          "{citation.quote || citation.text || 'No excerpt available.'}"
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === 'user';
  if (isUser) {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '20px' }}>
        <div style={{
          maxWidth: '72%', padding: '12px 16px', borderRadius: '18px 18px 4px 18px',
          background: 'var(--accent)', color: '#fff',
          fontSize: '14px', lineHeight: 1.55, fontWeight: 500,
        }}>
          {msg.text}
        </div>
      </div>
    );
  }

  const hasError = !!msg.error;
  const citations = Array.isArray(msg.citations) ? msg.citations : [];

  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', alignItems: 'flex-start' }}>
      {/* Avatar */}
      <div style={{
        width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
        background: hasError ? 'rgba(239,68,68,0.12)' : 'linear-gradient(135deg, var(--accent) 0%, var(--accent-2, #6366f1) 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        marginTop: '2px',
      }}>
        {hasError
          ? <AlertCircle size={15} color="#ef4444" />
          : <Zap size={15} color="#fff" />}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Answer text */}
        <div style={{
          background: hasError ? 'rgba(239,68,68,0.06)' : 'var(--surface-card)',
          border: `1px solid ${hasError ? 'rgba(239,68,68,0.25)' : 'var(--hairline)'}`,
          borderRadius: '4px 18px 18px 18px', padding: '14px 16px',
          fontSize: '14px', lineHeight: 1.65, color: 'var(--text-ink)',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        }}>
          {msg.text || 'No answer returned.'}
        </div>

        {/* Confidence + citation count */}
        {!hasError && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '12px',
            marginTop: '8px', padding: '0 4px',
          }}>
            {msg.confidence && <ConfidenceBadge confidence={msg.confidence} />}
            {citations.length > 0 && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted-soft)' }}>
                <CheckCircle2 size={10} style={{ verticalAlign: 'middle', marginRight: 3 }} />
                {citations.length} grounded source{citations.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
        )}

        {/* Collapsible citations */}
        {citations.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '10px' }}>
            {citations.map((cit, i) => (
              <CitationCard key={`${cit.chunk_id}-${i}`} citation={cit} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPanel({ job, messages, setMessages }) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const threadRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Load server-side history once when the panel first mounts for this job
  useEffect(() => {
    if (!job?.job_id || historyLoaded || messages.length > 0) {
      setHistoryLoaded(true);
      return;
    }
    api.get(`/jobs/${job.job_id}/history`)
      .then(res => {
        const history = res.data?.history || [];
        if (history.length > 0) {
          const restored = history.flatMap(entry => [
            { role: 'user', text: entry.question },
            {
              role: 'assistant',
              text: entry.answer?.answer || entry.answer?.error || 'No answer.',
              confidence: entry.answer?.confidence,
              citations: Array.isArray(entry.answer?.citations) ? entry.answer.citations : [],
              error: entry.answer?.error || null,
            },
          ]);
          setMessages(restored);
        }
      })
      .catch(() => {}) // silently ignore history fetch errors
      .finally(() => setHistoryLoaded(true));
  }, [job?.job_id]);

  const askQuestion = useCallback(async (rawQuestion) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed || !job?.job_id || loading) return;

    setMessages(prev => [...prev, { role: 'user', text: trimmed }]);
    setQuestion('');
    setLoading(true);

    try {
      const res = await api.post(`/jobs/${job.job_id}/ask`, { question: trimmed });
      const data = res.data || {};
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: data.answer || data.error || 'No answer returned.',
          confidence: data.confidence,
          citations: Array.isArray(data.citations) ? data.citations : [],
          error: data.error || null,
        },
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: 'Failed to reach the document Q&A engine.',
          confidence: 'LOW',
          citations: [],
          error: err?.response?.data?.error || err.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [job?.job_id, loading, setMessages]);

  const handleClearHistory = async () => {
    if (!job?.job_id) return;
    setMessages([]);
    try {
      await api.delete(`/jobs/${job.job_id}/history`);
    } catch (_) {}
  };

  const isEmpty = messages.length === 0 && !loading;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'var(--canvas)', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 24px', borderBottom: '1px solid var(--hairline)',
        background: 'var(--surface-card)', flexShrink: 0,
      }}>
        <div>
          <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '2px', color: 'var(--text-ink)' }}>
            Ask the Document
          </h3>
          <p style={{ fontSize: '12px', color: 'var(--text-muted-soft)', margin: 0 }}>
            Answers are grounded in the retrieved tender segments
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleClearHistory}
            title="Clear conversation"
            style={{
              display: 'flex', alignItems: 'center', gap: '5px',
              padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--hairline)',
              background: 'transparent', cursor: 'pointer',
              fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500,
            }}
          >
            <Trash2 size={13} />
            Clear
          </button>
        )}
      </div>

      {/* Thread */}
      <div
        ref={threadRef}
        style={{ flex: 1, overflowY: 'auto', padding: '24px', scrollBehavior: 'smooth' }}
      >
        {/* Starter chips — only when no messages */}
        {isEmpty && (
          <div style={{ marginBottom: '32px' }}>
            <p style={{ fontSize: '12px', color: 'var(--text-muted-soft)', marginBottom: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Try asking
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {STARTER_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => askQuestion(q)}
                  disabled={loading}
                  style={{
                    padding: '8px 14px', borderRadius: '20px',
                    border: '1px solid var(--hairline)',
                    background: 'var(--surface-card)', cursor: 'pointer',
                    fontSize: '13px', color: 'var(--text-muted)',
                    fontWeight: 500, lineHeight: 1.3, textAlign: 'left',
                    transition: 'border-color 0.15s, background 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--hairline)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <Message key={`${msg.role}-${i}`} msg={msg} />
        ))}

        {/* Loading indicator */}
        {loading && (
          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '24px' }}>
            <div style={{
              width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
              background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-2, #6366f1) 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Zap size={15} color="#fff" />
            </div>
            <div style={{
              padding: '14px 18px', borderRadius: '4px 18px 18px 18px',
              background: 'var(--surface-card)', border: '1px solid var(--hairline)',
              display: 'flex', alignItems: 'center', gap: '10px',
            }}>
              <span style={{ display: 'flex', gap: '4px' }}>
                {[0, 1, 2].map(i => (
                  <span key={i} style={{
                    width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)',
                    animation: `chatPulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                    display: 'inline-block',
                  }} />
                ))}
              </span>
              <span style={{ fontSize: '13px', color: 'var(--text-muted-soft)' }}>Synthesizing evidence…</span>
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <div style={{
        padding: '16px 24px', borderTop: '1px solid var(--hairline)',
        background: 'var(--surface-card)', flexShrink: 0,
      }}>
        <form
          onSubmit={e => { e.preventDefault(); askQuestion(question); }}
          style={{
            display: 'flex', gap: '10px', alignItems: 'center',
            padding: '6px 6px 6px 16px',
            background: 'var(--canvas)', borderRadius: '14px',
            border: '1px solid var(--hairline)',
            transition: 'border-color 0.2s',
          }}
          onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
          onBlur={e => e.currentTarget.style.borderColor = 'var(--hairline)'}
        >
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask anything about this tender…"
            disabled={loading}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              fontSize: '14px', color: 'var(--text-ink)',
              caretColor: 'var(--accent)',
            }}
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            style={{
              width: 38, height: 38, borderRadius: '10px', border: 'none',
              background: question.trim() && !loading ? 'var(--accent)' : 'var(--hairline)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: question.trim() && !loading ? 'pointer' : 'default',
              transition: 'background 0.2s',
              flexShrink: 0,
            }}
          >
            <Send size={16} color={question.trim() && !loading ? '#fff' : 'var(--text-muted-soft)'} />
          </button>
        </form>
      </div>

      {/* Pulse animation keyframes injected once */}
      <style>{`
        @keyframes chatPulse {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
          40% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}