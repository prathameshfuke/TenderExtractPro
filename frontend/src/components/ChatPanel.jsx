import { useState } from 'react';
import axios from 'axios';
import { Send, MessageSquareText } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

const starterQuestions = [
  'What are the main technical requirements?',
  'What is explicitly excluded from scope?',
  'What delivery or completion timeline is mentioned?',
];

export default function ChatPanel({ job }) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async (rawQuestion) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed || !job?.job_id || loading) return;

    const userMessage = { role: 'user', text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const res = await api.post(`/jobs/${job.job_id}/ask`, { question: trimmed });
      const data = res.data || {};
      const botMessage = {
        role: 'assistant',
        text: data.answer || data.error || 'No answer returned.',
        confidence: data.confidence || 'LOW',
        citations: Array.isArray(data.citations) ? data.citations : [],
        error: data.error || null,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: 'Failed to query the document.',
          confidence: 'LOW',
          citations: [],
          error: err?.response?.data?.error || err.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h3>Ask the Document</h3>
        <p>Grounded in the retrieved segments of this tender.</p>
      </div>

      <div className="chat-thread">
        {messages.length === 0 && (
          <div className="chat-starters" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '24px' }}>
            {starterQuestions.map((item) => (
              <button key={item} className="outline-pill" style={{ padding: '8px 16px', fontSize: '13px' }} type="button" onClick={() => askQuestion(item)}>
                {item}
              </button>
            ))}
          </div>
        )}

        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
            <span className="chat-role">{message.role === 'user' ? 'You' : 'TenderExtractPro'}</span>
            <div className="chat-bubble">
              <div>{message.text}</div>
              {message.error && <div className="chat-error" style={{ color: 'var(--error)', fontSize: '14px', marginTop: '8px' }}>{message.error}</div>}
              
              {Array.isArray(message.citations) && message.citations.length > 0 && (
                <div className="chat-citations" style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {message.citations.map((citation, citationIndex) => (
                    <div key={`${citation.chunk_id}-${citationIndex}`} className="source-text-box" style={{ fontSize: '13px', background: 'var(--canvas-soft)', border: '1px solid var(--hairline)' }}>
                      <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted-soft)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                        <span>Citation {citationIndex + 1}</span>
                        <span>Page {citation.page || '-'}</span>
                      </div>
                      {citation.quote || 'No quote available.'}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <span className="chat-role">TenderExtractPro</span>
            <div className="chat-bubble" style={{ color: 'var(--text-muted-soft)' }}>
              Synthesizing evidence...
            </div>
          </div>
        )}
      </div>

      <div className="chat-input-container">
        <form
          className="chat-input-bar"
          onSubmit={(event) => {
            event.preventDefault();
            askQuestion(question);
          }}
        >
          <input
            className="chat-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Search the document..."
            disabled={loading}
          />
          <button className="chat-send-btn" type="submit" disabled={loading || !question.trim()}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}