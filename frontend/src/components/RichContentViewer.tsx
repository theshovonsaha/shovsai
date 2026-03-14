import React, { useState } from 'react';
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
    let renderData: any = null;
    try {
        const trimmed = sanitizedContent.trim();
        // Look for the first { and last } to extract JSON even if there are prefixes/suffixes
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start !== -1 && end !== -1 && end > start) {
            const potentialJson = trimmed.substring(start, end + 1);
            renderData = JSON.parse(potentialJson);
        }
    } catch (e) {
        // Fallback to normal markdown
    }

    if (renderData && renderData.type === 'web_search_results') {
        return (
            <div className="search-results-viewer" style={{ margin: '1em 0', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Results for: <span style={{ color: 'var(--primary)' }}>{renderData.query}</span>
                </div>
                {renderData.results.map((r: any, i: number) => (
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
        const appPath = typeof renderData.path === 'string' ? encodeURI(renderData.path) : '';
        if (!appPath) {
            return (
                <div className="html-render-sandbox" style={{ margin: '1em 0', borderRadius: '12px', border: '1px solid var(--border)', background: 'var(--surface2, #1e1e1e)', padding: '16px' }}>
                    <div style={{ fontWeight: 600, marginBottom: '8px' }}>HTML Preview unavailable</div>
                    <div style={{ fontSize: '13px', opacity: 0.8 }}>
                        The app preview path is missing from the tool response.
                    </div>
                </div>
            );
        }
        return (
            <div className="html-render-sandbox" style={{ margin: '1em 0', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', boxShadow: '0 8px 30px rgba(0,0,0,0.5)' }}>
                <div className="render-header" style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 20px', background: '#0a0a0a', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--primary)', letterSpacing: '0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '18px' }}>◈</span> {renderData.title || 'V8 PLATINUM APP'}
                    </span>
                    <a href={appPath} target="_blank" rel="noreferrer" style={{ fontSize: '11px', color: 'var(--text-dim)', textDecoration: 'none', background: 'rgba(255,255,255,0.05)', padding: '4px 10px', borderRadius: '4px' }}>OPEN FULLSCREEN ↗</a>
                </div>
                <iframe
                    src={appPath}
                    title={renderData.title}
                    style={{ width: '100%', height: '600px', border: 'none', background: '#000' }}
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
                    code({ inline, className, children, ...props }: any) {
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
                    table({ children, ...props }: any) {
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

    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="code-block-container" style={{ position: 'relative', margin: '1em 0', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)' }}>
            <div className="code-block-header" style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 12px', background: 'var(--surface2, #1e1e1e)', color: 'var(--text-dim, #a0a0a0)', fontSize: '11px', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                <span>{language}</span>
                <div style={{ display: 'flex', gap: '12px' }}>
                    {isPreviewable && (
                        <button
                            onClick={() => setShowPreview(!showPreview)}
                            style={{ background: 'none', border: 'none', color: showPreview ? 'var(--primary, #fff)' : 'inherit', cursor: 'pointer', fontSize: 'inherit', fontFamily: 'inherit', padding: 0 }}
                        >
                            {showPreview ? 'CODE' : 'LIVE VIEW'}
                        </button>
                    )}
                    <button
                        onClick={copyToClipboard}
                        style={{ background: 'none', border: 'none', color: copied ? '#4caf50' : 'inherit', cursor: 'pointer', fontSize: 'inherit', fontFamily: 'inherit', padding: 0 }}
                    >
                        {copied ? 'COPIED!' : 'COPY'}
                    </button>
                </div>
            </div>

            {showPreview ? (
                <div
                    className="code-preview-area"
                    style={{ padding: '16px', background: '#fff', color: '#000', overflowX: 'auto' }}
                    dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(code) }}
                />
            ) : (
                <SyntaxHighlighter
                    style={vscDarkPlus as any}
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
