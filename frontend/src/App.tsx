import {useState} from 'react';
import type {FormEvent} from 'react';
import {Layout, LayoutContent} from '@astryxdesign/core';
import {Button} from '@astryxdesign/core/Button';
import {Card} from '@astryxdesign/core/Card';
import {Citation} from '@astryxdesign/core/Citation';
import {EmptyState} from '@astryxdesign/core/EmptyState';
import {Heading} from '@astryxdesign/core/Heading';
import {HStack} from '@astryxdesign/core/HStack';
import {StackItem} from '@astryxdesign/core/Stack';
import {Text} from '@astryxdesign/core/Text';
import {TextInput} from '@astryxdesign/core/TextInput';
import {VStack} from '@astryxdesign/core/VStack';

type Result = {
  question: string;
  answer: string | null;
  score: number | null;
  title: string;
  heading: string;
  paragraph: string;
  start: number | null;
  end: number | null;
  retrieve_ms: number;
  rerank_ms: number;
  highlight_ms: number;
  best_distance: number;
  best_rerank: number;
};

function wikiUrl(title: string, heading: string): string {
  const base = `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, '_'))}`;
  return heading ? `${base}#${encodeURIComponent(heading.replace(/ /g, '_'))}` : base;
}

function SourceParagraph({result}: {result: Result}) {
  const {paragraph, start, end} = result;
  if (start === null || end === null) {
    return <Text as="p">{paragraph}</Text>;
  }
  return (
    <Text as="p">
      {paragraph.slice(0, start)}
      <mark className="answer-mark">{paragraph.slice(start, end)}</mark>
      {paragraph.slice(end)}
    </Text>
  );
}

function AnswerView({result}: {result: Result}) {
  if (result.answer === null || result.score === null) {
    return (
      <EmptyState
        title="No answer found"
        description={`Best rerank score ${result.best_rerank.toFixed(2)}. This question is probably outside the corpus.`}
      />
    );
  }
  const ms = (n: number) => `${Math.round(n)}ms`;
  return (
    <VStack gap={3}>
      <VStack gap={1}>
        <Text type="large" weight="semibold">
          {result.answer}
        </Text>
        <Text type="supporting">
          score {result.score.toFixed(2)} · retrieve {ms(result.retrieve_ms)} · rerank{' '}
          {ms(result.rerank_ms)} · highlight {ms(result.highlight_ms)}
        </Text>
      </VStack>
      <Card>
        <VStack gap={4}>
          <SourceParagraph result={result} />
          <HStack gap={2} vAlign="center">
            <Citation
              source={{title: result.title, url: wikiUrl(result.title, result.heading)}}
              number={1}
            />
            {result.heading ? <Text type="supporting">{result.heading}</Text> : null}
          </HStack>
        </VStack>
      </Card>
    </VStack>
  );
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function onSubmit(e?: FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || isLoading) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api?q=${encodeURIComponent(q)}`);
      if (!response.ok) throw new Error(`server returned ${response.status}`);
      setResult((await response.json()) as Result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Layout
      contentWidth={720}
      content={
        <LayoutContent>
          <VStack gap={6}>
            <VStack gap={2}>
              <Heading level={1}>verbatim RAG</Heading>
              <Text color="secondary">
                Ask a question. The answer is an exact quote from the source text.
              </Text>
            </VStack>
            <form onSubmit={onSubmit}>
              <HStack gap={2} vAlign="end">
                <StackItem size="fill">
                  <TextInput
                    label="Question"
                    isLabelHidden
                    width="100%"
                    value={question}
                    onChange={setQuestion}
                    onEnter={onSubmit}
                    placeholder="What is the capital of Japan?"
                  />
                </StackItem>
                <Button
                  type="submit"
                  label="Ask"
                  variant="primary"
                  isLoading={isLoading}
                />
              </HStack>
            </form>
            {error ? <Text color="secondary">Error: {error}</Text> : null}
            {result ? <AnswerView result={result} /> : null}
          </VStack>
        </LayoutContent>
      }
    />
  );
}
