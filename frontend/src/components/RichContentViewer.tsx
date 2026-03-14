import React, { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import DOMPurify from 'dompurify';
import 'katex/dist/katex.min.css';

interface RichContentViewerProps {
    content: string;
}

interface WebSearchResult {
    title: string;
    url: string;
    snippet: string;
}

const tryParseStructuredContent = (content: string) => {
    try {
        const trimmed = content.trim();
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start !== -1 && end !== -1 && end > start) {
            return JSON.parse(trimmed.substring(start, end + 1));
        }
    } catch {
        // Fallback to markdown
    }

    return null;
};

export const RichContentViewer: React.FC<RichContentViewerProps> = ({ content }) => {
    // Sanitize content to handle common LLM output issues that break KaTeX
    const sanitizedContent = content
        // Replace non-breaking hyphens with standard hyphens
        .replace(/\u2011/g, '-')
        // Replace smart/curly quotes with standard single/double quotes
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        // Fix trailing % in math blocks that confuses KaTeX parser
        .replace(/%(\s*\$)/g, '$1');

    // Check if the content is a JSON result from tools
    const renderData = tryParseStructuredContent(sanitizedContent);

    if (renderData && renderData.type === 'web_search_results') {
        return (
            <div className="search-results-viewer" style={{ margin: '1em 0', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Results for: <span style={{ color: 'var(--primary)' }}>{renderData.query}</span>
                </div>
                {renderData.results.map((r: WebSearchResult, i: number) => (
                    <div key={i} className="search-card" style={{ background: 'var(--surface2, #1e1e1e)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                        <a href={r.url} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', fontWeight: 600, textDecoration: 'none', fontSize: '14px', display: 'block', marginBottom: '4px' }}>
                            {r.title}
                        </a>
                        <div style={{ fontSize: '11px', color: 'var(--accent, #00ff85)', opacity: 0.7, marginBottom: '6px', wordBreak: 'break-all' }}>{r.url}</div>
                        <div style={{ fontSize: '13px', lineHeight: '1.4', opacity: 0.9 }}>{r.snippet}</div>
                    </div>
                ))}
            </div>
        );
    }

    if (renderData && renderData.type === 'web_fetch_result') {
        return (
            <div className="fetch-result-viewer" style={{ margin: '1em 0', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--surface2, #1e1e1e)' }}>
                <div style={{ padding: '10px 16px', background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600 }}>📄 Page Content</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-dim)' }}>{renderData.url}</span>
                </div>
                {renderData.error ? (
                    <div style={{ padding: '20px', color: 'var(--error)', fontSize: '13px' }}>Error: {renderData.error}</div>
                ) : (
                    <div style={{ padding: '16px', fontSize: '13px', lineHeight: '1.6', maxHeight: '400px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
                        {renderData.content}
                        {renderData.truncated && (
                            <div style={{ marginTop: '12px', padding: '8px', background: 'rgba(255,184,0,0.1)', color: 'var(--warn)', borderRadius: '4px', fontSize: '11px' }}>
                                Content truncated ({renderData.total_length} chars total)
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    }

    if (renderData && (renderData.path || renderData.type === 'app_view')) {
        return (
            <div className="html-render-sandbox">
                <div className="render-header">
                    <div className="render-heading">
                        <span className="render-title">
                            <span className="render-title-icon">◈</span>
                            {renderData.title || 'V8 PLATINUM APP'}
                        </span>
                        <span className="render-subtitle">Interactive sandbox preview ready in chat</span>
                    </div>
                    <div className="render-actions">
                        <span className="render-badge">live preview</span>
                        <a href={renderData.path} target="_blank" rel="noreferrer" className="render-open-link">OPEN FULLSCREEN ↗</a>
                    </div>
                </div>
                <div className="render-meta">
                    <span className="render-meta-label">Source</span>
                    <span className="render-meta-value">{renderData.path}</span>
                </div>
                <iframe
                    src={renderData.path}
                    title={renderData.title}
                    className="render-frame"
                    sandbox="allow-scripts allow-popups allow-same-origin"
                />
            </div>
        );
    }

    return (
        <div className="rich-content-viewer">
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                    rehypeRaw,
                    [rehypeKatex, {
                        strict: 'ignore',
                        throwOnError: false,
                        trust: true
                    }]
                ]}
                components={{
                    code({ inline, className, children, ...props }: React.ComponentProps<'code'> & { inline?: boolean }) {
                        const match = /language-(\w+)/.exec(className || '');
                        const language = match ? match[1] : '';
                        const codeString = String(children).replace(/\n$/, '');

                        if (inline || !match) {
                            return (
                                <code className={className} {...props}>
                                    {children}
                                </code>
                            );
                        }

                        return <CodeBlock language={language} code={codeString} />;
                    },
                    table({ children, ...props }: React.ComponentProps<'table'>) {
                        return (
                            <div style={{ overflowX: 'auto', margin: '1em 0' }}>
                                <table {...props}>{children}</table>
                            </div>
                        );
                    }
                }}
            >
                {sanitizedContent}
            </ReactMarkdown>
        </div>
    );
};

const CodeBlock = ({ language, code }: { language: string; code: string }) => {
    const [showPreview, setShowPreview] = useState(false);
    const [copied, setCopied] = useState(false);

    // Modular Live View logic: currently HTML and SVG are natively supported in the DOM.
    // Can be easily expanded to JSON visualizations, charts, etc.
    const isPreviewable = ['html', 'svg'].includes(language?.toLowerCase());
    const previewMarkup = useMemo(() => {
        if (!isPreviewable) return '';

        const previewLanguage = language?.toLowerCase();
        const sanitizedMarkup = DOMPurify.sanitize(
            code,
            previewLanguage === 'svg'
                ? { USE_PROFILES: { svg: true, svgFilters: true } }
                : { USE_PROFILES: { html: true } }
        );

        if (previewLanguage === 'svg') {
            return `<!doctype html><html><body style="margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;background:#ffffff;">${sanitizedMarkup}</body></html>`;
        }

        return sanitizedMarkup;
    }, [code, isPreviewable, language]);

    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="code-block-container">
            <div className="code-block-header">
                <div className="code-block-language">
                    <span>{language}</span>
                    {isPreviewable && <span className="code-block-pill">preview available</span>}
                </div>
                <div className="code-block-actions">
                    {isPreviewable && (
                        <button
                            onClick={() => setShowPreview(!showPreview)}
                            className={`code-block-action ${showPreview ? 'active' : ''}`}
                        >
                            {showPreview ? 'CODE' : 'LIVE VIEW'}
                        </button>
                    )}
                    <button
                        onClick={copyToClipboard}
                        className={`code-block-action ${copied ? 'copied' : ''}`}
                    >
                        {copied ? 'COPIED!' : 'COPY'}
                    </button>
                </div>
            </div>

            {showPreview ? (
                <div className="code-preview-shell">
                    <div className="code-preview-note">Sandboxed live preview</div>
                    <iframe
                        className="code-preview-frame"
                        sandbox="allow-scripts"
                        srcDoc={previewMarkup}
                        title={`${language} live preview`}
                    />
                </div>
            ) : (
                <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={language}
                    PreTag="div"
                    customStyle={{ margin: 0, borderRadius: 0, fontSize: '13px', background: '#0d0d0d' }}
                >
                    {code}
                </SyntaxHighlighter>
            )}
        </div>
    );
};
