import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';

interface RichContentViewerProps {
    content: string;
}

export const RichContentViewer: React.FC<RichContentViewerProps> = ({ content }) => {
    return (
        <div className="rich-content-viewer">
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeRaw, rehypeKatex]}
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
                {content}
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
                    dangerouslySetInnerHTML={{ __html: code }}
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
